import sys
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scene.entropy_models import EntropyBottleneck
import math

class rdloss(torch.nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, return_type="all"):
        super().__init__()

        self.metric = torch.nn.MSELoss()
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, y_hat,y_likelihoods, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in y_likelihoods
        )
        out["mse_loss"] = self.metric(y_hat, target)
        distortion = out["mse_loss"]
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]

class Motion_Grid(nn.Module):
    def __init__(self,q = 1):
        super(Motion_Grid, self).__init__() 
        self.device = torch.device("cuda")
        self.grid = self._init_grid()
        self.tinymlp = nn.Sequential(nn.Linear(20, 64), nn.LeakyReLU(), nn.Linear(64, 7))
        self.scale = 1.0
        self.entropy_bottleneck0 = EntropyBottleneck(channels=32,entropy_coder='rangecoder').to(self.device)
        self.entropy_bottleneck1 = EntropyBottleneck(channels=64,entropy_coder='rangecoder').to(self.device)
        self.entropy_bottleneck2 = EntropyBottleneck(channels=128,entropy_coder='rangecoder').to(self.device)
        self.entropy_bottleneck0.update()
        self.entropy_bottleneck1.update()
        self.entropy_bottleneck2.update()
        self.criterion = rdloss(lmbda=0.001)
        self.num_frequencies = 3
        self.q = q
        self.is_train = True

    def _init_grid(self):
        grid = []
        grid.append(nn.Parameter(torch.randn(1,4,32,32,32),requires_grad=True))
        grid.append(nn.Parameter(torch.randn(1,4,64,64,64),requires_grad=True))
        grid.append(nn.Parameter(torch.randn(1,2,128,128,128),requires_grad=True))

        return nn.ParameterList(grid)
    
    def positional_encoding(self, x):
        pe = []
        for i in range(self.num_frequencies):
            pe.append(torch.sin((2 ** i) * np.pi * x))
            pe.append(torch.cos((2 ** i) * np.pi * x))
        pes = torch.cat(pe, dim=-1) # N , 3*2*num_frequencies
        return pes
    
    def save(self, path):
        ckpt = self.state_dict()
        for i in range(len(self.grid)):
            ckpt.pop(f'grid.{i}')
        torch.save(ckpt, path)
        for i in range(len(self.grid)):
            getattr(self,'entropy_bottleneck'+str(i)).update()
        self.compress(path.replace('.pth','_compress.pth'))
    
    def load(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt,strict=False)
        self.decompress(path.replace('.pth','_compress.pth'))

    def compress(self, path):
        for i in range(len(self.grid)):
            getattr(self,'entropy_bottleneck'+str(i)).compress_range(self.grid[i].data.squeeze()*self.q,path = path.replace('.pth','_grid'+str(i)))


    def decompress(self, path):
        for i in range(len(self.grid)):
            getattr(self,'entropy_bottleneck'+str(i)).file_path = path.replace('.pth','_compressgrid'+str(i)+'.bin')
            self.grid[i].data = torch.tensor(getattr(self,'entropy_bottleneck'+str(i)).decompress_range(path.replace('.pth','_grid'+str(i))),dtype=torch.float32).view(self.grid[i].shape).to(self.device)/self.q

    def get_optparam_groups(self):
        grad_vars = []
        grad_vars += [{'params': self.grid, 'lr': 5e-2}]
        grad_vars += [{'params': self.tinymlp.parameters(),'lr': 1e-2}]
        grad_vars += [{'params': self.entropy_bottleneck0.parameters(),'lr': 1e-2}]
        grad_vars += [{'params': self.entropy_bottleneck1.parameters(),'lr': 1e-2}]
        grad_vars += [{'params': self.entropy_bottleneck2.parameters(),'lr': 1e-2}]
        return grad_vars
    
    def interpolate(self, input):
        output = []
        if self.is_train:
            for i in range(3):
                noise_shape = self.grid[i].shape
                half = 0.5 / self.q
                noise = torch.rand(noise_shape, device=self.device)/self.q - half
                output.append(F.grid_sample(self.grid[i]+noise, input[:,:,:,:,i*6:i*6+3], mode='bilinear', align_corners=False))
                output.append(F.grid_sample(self.grid[i]+noise, input[:,:,:,:,i*6+3:i*6+6], mode='bilinear', align_corners=False))
        else:
            for i in range(3):
                output.append(F.grid_sample(self.grid[i], input[:,:,:,:,i*6:i*6+3], mode='bilinear', align_corners=False))
                output.append(F.grid_sample(self.grid[i], input[:,:,:,:,i*6+3:i*6+6], mode='bilinear', align_corners=False))

        return torch.cat(output, dim=1).view(20,-1).T
    
    def train_entropy(self,q=1):
        codec_loss = 0.0
        for i in range(len(self.grid)):
            y_hat, y_likelihoods = getattr(self,"entropy_bottleneck"+str(i))(self.grid[i].squeeze(0)*q)
            codec_loss += self.criterion(y_hat,y_likelihoods, self.grid[i].squeeze(0)*q)['loss']
        return codec_loss
    
    def normalize(self, x):
        x_min, _ = x.min(dim=0, keepdim=True)
        x_max, _ = x.max(dim=0, keepdim=True)
        x = 2 * (x - x_min) / (x_max - x_min) - 1
        return x

    def forward(self, x):
        x = self.positional_encoding(x)
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x = self.interpolate(x).reshape(-1,20)
        x = self.tinymlp(x)
        return x
    
