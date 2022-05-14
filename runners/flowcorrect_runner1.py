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
from models.refinenet_dilated_baseline import RefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
import shutil


import argparse
import numpy as np
import os
import time
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
import math
from models import FlowPlusPlus
from models import EBM_res as EBM
from tqdm import tqdm
from matplotlib import pyplot as plt

__all__ = ['FloppCorrectRunner']


class FloppCorrectRunner():
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
        
        # Model
        print('Building model..')
        flow_net = FlowPlusPlus(scales=[(0, 4), (2, 3)],
                           in_shape=(3, 32, 32),
                           mid_channels=self.config.flow_model.num_channels,
                           num_blocks=self.config.flow_model.num_blocks,
                           num_dequant_blocks=self.config.flow_model.num_dequant_blocks,
                           #num_dequant_blocks=-1,
                           num_components=self.config.flow_model.num_components,
                           use_attn=self.config.flow_model.use_attn,
                           drop_prob=self.config.flow_model.drop_prob)
        flow_net = flow_net.to(self.config.device)
        flow_net = torch.nn.DataParallel(flow_net)
        cudnn.benchmark = True
        
        loss_fn = util.NLLLoss().to(device)
        flow_param_groups = util.get_param_groups(flow_net, self.config.flow_training.weight_decay, norm_suffix='weight_g')
        flow_optimizer = optim.Adam(flow_param_groups, lr=self.config.flow_training.lr_flow)
        warm_up = self.config.flow_training.warm_up * self.config.training.batch_size
        flow_scheduler = sched.LambdaLR(flow_optimizer, lambda s: min(1., s / warm_up))
        
        
        
            
        score = RefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = -1
        
        flow_losses = []
        sbm_losses = []
        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1

                score.train()
                flow_net.train()
                
                x_list = []
                # train flow model
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)
                    
                if epoch == 0 and step < 200:
                    if step == -1:
                        print('Data dependent initialization for flow parameter at the begining of training')
                    x_list.append(x.clone()) # use more data to do data dependent initialization
                    if len(x_list) >= 20:
                        x_list = torch.cat(x_list, dim=0)
                        with torch.no_grad():
                            flow_net(x_list.detach(), reverse=False)
                        x_list = []
                    counter += 1
                    if counter == 199:
                        print('Begin training')
                    continue
                    
                flow_optimizer.zero_grad()
                z, sldj = flow_net(X.detach(), reverse=False)
                flow_loss = loss_fn(z, sldj)
                flow_loss.backward()
                if self.config.flow_training.max_grad_norm > 0:
                    util.clip_grad_norm(flow_optimizer, self.config.flow_training.max_grad_norm)
                    
                flow_optimizer.step()
                flow_scheduler.step(step*self.config.training.batch_size)
                flow_losses.append(flow_loss.item())
                
                # train residual score
                
                flow_net.eval()
                for p in flow_net.parameters():
                    p.requires_grad_(False)
                    
                X.requires_grad_(True)
                total_score = lambda X: torch.autograd.grad(-loss_fn(*flow_net(X, reverse=False)), X, retain_graph=True, create_graph=True)[0] - score(X)/self.config.training.lam
                loss = dsm_score_estimation(total_score, X, sigma=0.01)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                flow_net.train()
                for p in flow_net.parameters():
                    p.requires_grad_(True)

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, flow_loss: {},stein loss: {}".format(step, flow_loss.item(), loss.item()))

                if step >= self.config.training.n_iters:
                    return 0
                
                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    # torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
                    
                    # visualize and save
                    grid_size = 6
                    if not os.path.exists(self.args.image_folder):
                        os.makedirs(self.args.image_folder)
                    score.eval()
                    flow_net.eval()

                    imgs = []
                    if self.config.data.dataset == 'MNIST' or self.config.data.dataset == 'FashionMNIST':
                        samples = torch.rand(grid_size**2, 1, self.config.data.image_size, self.config.data.image_size,device=self.config.device)
                    else: 
                        samples = torch.rand(grid_size**2, 3, self.config.data.image_size, self.config.data.image_size,device=self.config.device)

                    all_samples = self.Langevin_dynamics_flowscore(samples,flow_net, score, 1000, 0.00002)

                    for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                        sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                             self.config.data.image_size)
                        if self.config.data.logit_transform:
                            sample = torch.sigmoid(sample)
                        if i % 10 == 0:
                            im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                            imgs.append(im)

                    image_grid = make_grid(all_samples[-1], nrow=grid_size)
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(step)))
                    imgs[0].save("movie.gif", save_all=True, append_images= imgs[1:], duration=1, loop=0)

    def Langevin_dynamics_flowscore(self, x_mod, flow, resscore, n_steps=1000, step_lr=0.00002):
        images = []
        
        x_mod.requires_grad_(True)
        for _ in range(n_steps):
            images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
            noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
            x_mod.grad.zero_()
            (-loss_fn(*flow(x_mod, reverse=False))).backward()
            grad = x_mod.grad.data - resscore(x_mod).detach()/self.config.training.lam
            x_mod.data.add_(step_lr*grad + noise)
            images.append(x_mod.clone().detach())
            # print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images
