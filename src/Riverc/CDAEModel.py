"""
Auto-Encoder Architecture
"""

from torch import linalg as LA

import torch as T
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear


import os.path as osp
SEED = 1234

T.manual_seed(SEED)
T.backends.cudnn.deterministic = True


class AutoEncoder(nn.Module):

    def __init__(self, hp, args):
        super(AutoEncoder, self).__init__()

        inDim = outDim = hp.imDim
        self.latentDim = hp.latentDim
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(inDim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32), 
        #     nn.ReLU(),
        #     nn.Linear(32, hp.latentDim)) 
        # self.decoder = nn.Sequential(
        #     nn.Linear(hp.latentDim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, outDim))

        if hp.AE_Model == 1:
            self.encoder = nn.Sequential(
                nn.Linear(inDim, 256),
                nn.Softplus(),
                nn.Linear(256, 128), 
                nn.Softplus(),
                nn.Linear(128, hp.latentDim)) 
            self.decoder = nn.Sequential(
                nn.Linear(hp.latentDim, 128),
                nn.Softplus(),
                nn.Linear(128, 256),
                nn.Softplus(),
                nn.Linear(256, outDim)) 
        
        if hp.AE_Model == 2:
            self.encoder = nn.Sequential(
                nn.Linear(inDim, 256),
                nn.SiLU(),
                nn.Linear(256, 128), 
                nn.SiLU(),
                nn.Linear(128, hp.latentDim)) 
            self.decoder = nn.Sequential(
                nn.Linear(hp.latentDim, 128),
                nn.SiLU(),
                nn.Linear(128, 256),
                nn.SiLU(),
                nn.Linear(256, outDim)) 

        if hp.AE_Model == 3:
            self.encoder = nn.Sequential(
                nn.Linear(inDim, 256),
                nn.SELU(),
                nn.Linear(256, 128), 
                nn.SELU(),
                nn.Linear(128, hp.latentDim)) 
            self.decoder = nn.Sequential(
                nn.Linear(hp.latentDim, 128),
                nn.SELU(),
                nn.Linear(128, 256),
                nn.SELU(),
                nn.Linear(256, outDim)) 

        if hp.AE_Model == 4:
            self.encoder = nn.Sequential(
                nn.Linear(inDim, 500),
                nn.SiLU(),
                nn.Linear(500, 250), 
                nn.SiLU(),
                nn.Linear(250, hp.latentDim)) 
            self.decoder = nn.Sequential(
                nn.Linear(hp.latentDim, 250),
                nn.SiLU(),
                nn.Linear(250, 500),
                nn.SiLU(),
                nn.Linear(500, outDim)) 

        if hp.AE_Model == 5:
            self.encoder = nn.Sequential(
                nn.Linear(inDim, 500),
                nn.PReLU(),
                nn.Linear(500, 250), 
                nn.PReLU(),
                nn.Linear(250, hp.latentDim)) 
            self.decoder = nn.Sequential(
                nn.Linear(hp.latentDim, 250),
                nn.PReLU(),
                nn.Linear(250, 500),
                nn.PReLU(),
                nn.Linear(500, outDim)) 

        # self.encoder = nn.Sequential(
        #     nn.Linear(inDim, 64),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Linear(64, hp.latentDim)) 
        # self.decoder = nn.Sequential(
        #     nn.Linear(hp.latentDim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, outDim)) 

        # self.encoder = nn.Sequential(
        #     nn.Linear(inDim, 64),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(64, 32), 
        #     nn.BatchNorm1d(num_features=32),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(32, hp.latentDim)) 
        # self.decoder = nn.Sequential(
        #     nn.Linear(hp.latentDim, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, outDim)) 
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(inDim, 64),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32), 
        #     nn.BatchNorm1d(num_features=32),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(32, hp.latentDim)) 
        # self.decoder = nn.Sequential(
        #     nn.Linear(hp.latentDim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, outDim)) 

    def forward(self, x):
        """
        Args:
            x (Tensor): (currentBatchSize, imDim) or (currentBatchSize, latentDim)
        Returns:
            nextLatentVec (Tensor): (currentBatchSize, imDim)
        """
        if x.shape[-1] == self.latentDim:
            return self.decoder(x)
        else:
            y = self.encoder(x)
            z = self.decoder(y)
            return z,y
