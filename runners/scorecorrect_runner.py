import numpy as np
import tqdm
from losses.dsm import dsm_score_estimation
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.refinenet_dilated_baseline import RefineNetDilated, init_net
from torchvision.utils import save_image, make_grid
from PIL import Image
import shutil
from losses.stein import keep_grad, approx_jacobian_trace, exact_jacobian_trace, stein_stats

from models import nice, utils

__all__ = ['CorrectorRunner']

class CorrectorRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        ## load pretrained flow model
        args = torch.load('models/mnist/saved_dicts')

        # model hyperparameters
        dataset = args["dataset"]
        latent = args["latent"]
        coupling = args['coupling']
        mask_config = args["mask_config"]
        full_dim = args["full_dim"]
        mid_dim = args["mid_dim"]
        hidden = args["hidden"]
            
        if latent == 'normal':
            prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif latent == 'logistic':
            prior = utils.StandardLogistic()

        flow = nice.NICE(prior=prior, 
                    coupling=coupling, 
                    in_out_dim=full_dim, 
                    mid_dim=mid_dim, 
                    hidden=hidden, 
                    mask_config=mask_config).to(self.config.device)
        flow.load_state_dict(args["model_state_dict"])

        for p in flow.parameters():
            p.requires_grad_(False)


        
        ## score function
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            # test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
            #                        transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            # test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
            #                      transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            if self.config.data.random_flip:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), download=True)
            else:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.ToTensor(),
                                 ]), download=True)

            # test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba_test'), split='test',
            #                       transform=transforms.Compose([
            #                           transforms.CenterCrop(140),
            #                           transforms.Resize(self.config.data.image_size),
            #                           transforms.ToTensor(),
            #                       ]), download=True)


        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        # test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
        #                          num_workers=4, drop_last=True)

        # test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = RefineNetDilated(self.config).to(self.config.device)
        score = init_net(score, 'zero', 0.002, [0])

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1

                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                X.requires_grad_()
                # loss = dsm_score_estimation(score, X, sigma=0.01)
                logp = flow.log_prob(X.view(X.shape[0],-1))
                stats, norms, grad_norms, logp_u = stein_stats(logp, X, score, approx_jcb=False, n_samples=1)

                loss = -1*stats.mean() + 0.5*norms.mean()/self.config.training.l2_labmda
                # score = s_flow - s_pd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                # if step % 100 == 0:
                    # score.eval()
                    # try:
                    #     test_X, test_y = next(test_iter)
                    # except StopIteration:
                    #     test_iter = iter(test_loader)
                    #     test_X, test_y = next(test_iter)

                    # test_X = test_X.to(self.config.device)
                    # test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    # if self.config.data.logit_transform:
                    #     test_X = self.logit_transform(test_X)

                    # with torch.no_grad():
                    #     test_dsm_loss = dsm_score_estimation(score, test_X, sigma=0.01)

                    # tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    # torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

                    # with torch.no_grad():
                    self.test(flow=flow,score=score,iters=step)


    def Langevin_dynamics(self, x_mod, scorenet, flow, n_steps=1000, step_lr=0.00002):
        images = []

        b,c,h,w = x_mod.shape

        x_mod = flow.sample(b).to(self.config.device).view(x_mod.shape).detach()
        x_mod = torch.autograd.Variable(x_mod, requires_grad=True)
        # with torch.no_grad():
        for _ in range(n_steps):
            print(_)
            images.append(torch.clamp(x_mod.detach(), 0.0, 1.0).to('cpu'))
            noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
            
            grad = torch.autograd.grad(flow.log_prob(x_mod.view(x_mod.shape[0],-1)).sum(), [x_mod], retain_graph=True)[0]
            grad -= scorenet(x_mod).detach()/self.config.training.l2_labmda
            x_mod.data.add_(step_lr * grad + noise)
            print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

        return images

    def test(self,flow=None,score=None,iters=None):
        if not score:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
            score = RefineNetDilated(self.config).to(self.config.device)
            score = torch.nn.DataParallel(score)

            score.load_state_dict(states[0])

        if not flow:
            ## load pretrained flow model
            args = torch.load('models/mnist/saved_dicts')

            # model hyperparameters
            dataset = args["dataset"]
            latent = args["latent"]
            coupling = args['coupling']
            mask_config = args["mask_config"]
            full_dim = args["full_dim"]
            mid_dim = args["mid_dim"]
            hidden = args["hidden"]
                
            if latent == 'normal':
                prior = torch.distributions.Normal(
                    torch.tensor(0.).to(device), torch.tensor(1.).to(device))
            elif latent == 'logistic':
                prior = utils.StandardLogistic()

            flow = nice.NICE(prior=prior, 
                        coupling=coupling, 
                        in_out_dim=full_dim, 
                        mid_dim=mid_dim, 
                        hidden=hidden, 
                        mask_config=mask_config).to(self.config.device)
            flow.load_state_dict(args["model_state_dict"])

            for p in flow.parameters():
                p.requires_grad_(False)
      

        grid_size = 5
        
        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)



        score.eval()

        imgs = []
        if self.config.data.dataset == 'MNIST' or self.config.data.dataset == 'FashionMNIST':
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'MNIST':
                dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                                transform=transform)
            else:
                dataset = FashionMNIST(os.path.join(self.args.run, 'datasets', 'fmnist'), train=True, download=True,
                                       transform=transform)

            dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            samples, _ = next(data_iter)
            samples = samples.cuda()

            samples = torch.rand(grid_size**2, 1, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)
            all_samples = self.Langevin_dynamics(samples, score,flow, 5, 0.01)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 1 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))



        elif self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
            samples, _ = next(iter(dataloader))

            samples = torch.rand(grid_size ** 2, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            all_samples = self.Langevin_dynamics(samples, score,flow, 1000, 0.00002)

            # for i, sample in enumerate(tqdm.tqdm(all_samples)):
            #     sample = sample.view(100, self.config.data.channels, self.config.data.image_size,
            #                          self.config.data.image_size)

            #     if self.config.data.logit_transform:
            #         sample = torch.sigmoid(sample)

            #     torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))


            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)), nrow=10)
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)

            dataloader = DataLoader(dataset, batch_size=grid_size ** 2, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            samples, _ = next(data_iter)
            samples = samples.cuda()
            samples = torch.rand_like(samples)

            all_samples = self.Langevin_dynamics(samples, score,flow, 1000, 0.00002)

            # for i, sample in enumerate(tqdm.tqdm(all_samples)):
            #     sample = sample.view(100, self.config.data.channels, self.config.data.image_size,
            #                          self.config.data.image_size)

            #     if self.config.data.logit_transform:
            #         sample = torch.sigmoid(sample)

            #     torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))


            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)), nrow=10)
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))
            
        # imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
        shutil.rmtree(self.args.image_folder)
        if iters:
            imgs[0].save("{}_movie.gif".format(iters), save_all=True, append_images=imgs[1:], duration=1, loop=0)
            try:
                imgs[0].save("/content/drive/MyDrive/{}_movie.gif".format(iters), save_all=True, append_images=imgs[1:], duration=1, loop=0)
            except:
                pass
        else:
            imgs[0].save("movie.gif", save_all=True, append_images=imgs[1:], duration=1, loop=0)
