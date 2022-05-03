import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
from losses.stein import annealed_ssc
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image

__all__ = ['ScoreCorrectRunner']


class ScoreCorrectRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

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
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)

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

            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba_test'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=True)

        elif self.config.data.dataset == 'SVHN':
            dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                           transform=tran_transform)
            test_dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn_test'), split='test', download=True,
                                transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)



        ## load base score
        basescore = CondRefineNetDilated(self.config).to(self.config.device)
        basescore = torch.nn.DataParallel(basescore)

        if self.config.data.dataset == 'MNIST':
            states = torch.load(os.path.join("run/logs/mnist", 'checkpoint.pth'), map_location=self.config.device)
        elif self.config.data.dataset == 'CIFAR10':
            states = torch.load(os.path.join("run/logs/cifar10", 'checkpoint.pth'), map_location=self.config.device)
        else:
            raise NotImplementedError('dataset {} ckpt not found'.format(self.config.data.dataset))

        basescore.load_state_dict(states[0])
        for p in basescore.parameters():
            p.requires_grad_(False)
        basescore.eval()
        print(" base score loaded")


        ## initialize score
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)
        print('score model initialized')

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = -1

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)


        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    score_fn = lambda X, labels: basescore(X,labels).detach() - score(X, labels)/self.config.training.lam 
                    loss = anneal_dsm_score_estimation(score_fn, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    score_fn = lambda X, labels: basescore(X,labels).detach() - score(X, labels)/self.config.training.lam 
                    loss = anneal_sliced_score_estimation_vr(score_fn, X, labels, sigmas,n_particles=self.config.training.n_particles)
                elif self.config.training.algo == 'ssc':
                    loss = annealed_ssc(basescore, score, X, sigmas, labels=None, lam=1.0, anneal_power=2.0, hook=None, n_particles=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

                    score.eval()

                    ## Different part from NeurIPS 2019.
                    ## Random state will be affected because of sampling during training time.
                    init_samples = torch.rand(36, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    if self.config.data.logit_transform:
                        sample = torch.sigmoid(sample)

                    all_samples = self.anneal_Langevin_caliberation(init_samples,basescore, score, sigmas.cpu().numpy(), self.config.training.lam,
                                                            self.config.sampling.n_steps_each,
                                                            self.config.sampling.step_lr,
                                                            denoise=self.config.sampling.denoise)

                    # making last sample
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    if self.config.data.logit_transform:
                        sample = torch.sigmoid(sample)
                    
                    image_grid = make_grid(sample, 6)
                    save_image(image_grid,os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                    torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))

                    # making gif
                    imgs = []
                    for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                        if i%10 == 0 or i== len(all_samples):
                          sample = sample.view(sample.shape[0], self.config.data.channels, self.config.data.image_size,self.config.data.image_size)
                          if self.config.data.logit_transform:
                              sample = torch.sigmoid(sample)

                          image_grid = make_grid(sample, nrow=int(np.sqrt(sample.shape[0])))
                          im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                          imgs.append(im)
                        else:
                          pass
                    imgs[0].save(os.path.join(self.args.log_sample_path, "{}_movie.gif".format(step)), save_all=True, append_images=imgs[1:], duration=1, loop=0)

                    del imgs
                    del all_samples

    def anneal_Langevin_caliberation(self, x_mod, basescorenet, resscorenet, sigmas, lam, n_steps_each=100, step_lr=0.00002,denoise=True):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = basescorenet(x_mod, labels) - resscorenet(x_mod, labels)/lam
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),grad.abs().max()))

            if denoise:
                x_mod = x_mod + step_size * (basescorenet(x_mod, labels) - resscorenet(x_mod, labels)/lam)
                images.append(x_mod.to('cpu'))                

            return images
