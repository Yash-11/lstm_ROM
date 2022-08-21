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



class AutoEncoder(nn.Module):

    def __init__(self, hp, args):
        super(AutoEncoder, self).__init__()

        inDim = outDim = hp.imDim
        self.latentDim = hp.latentDim
        
        self.encoder = nn.Sequential(
            nn.Linear(inDim, 64),
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, hp.latentDim)) 
        self.decoder = nn.Sequential(
            nn.Linear(hp.latentDim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, outDim)) 

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