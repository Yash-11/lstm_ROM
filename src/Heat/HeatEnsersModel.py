"""
Network Architecture
"""

import pandas as pd
from torch import linalg as LA
from typing import Union, Tuple, Optional, List

import torch as T
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear

import pdb
import os.path as osp


class Model(nn.Module):

    def __init__(self, hp, hiddenDim, args):
        super(Model, self).__init__()

        self.info = args.logger.info
        self.device = args.device
        self.args = args
        self.hiddenDim = hp.hiddenDim

        self.rl = nn.LSTM(hp.latentDim, hp.hiddenDim, num_layers=1, batch_first=True)
        self.l1 = nn.Linear(hp.hiddenDim, hp.latentDim)


    def reset_hidden_states(self, for_batch=None):
        if for_batch is not None:
            batch_size = for_batch.shape[0]

        # Initialize recurrent hidden states
        self.rl_h = self.init_hidden(batch_size=batch_size)
        self.rl_c = self.init_hidden(batch_size=batch_size)

        if for_batch is not None:
            device = for_batch.device
      
            self.rl_h = self.rl_h.to(device)
            self.rl_c = self.rl_c.to(device)


    def init_hidden(self, batch_size):
        layers =1
        return T.randn(layers, batch_size, self.hiddenDim)

    def forward(self, snapshot_Seq):
        """
        Args:
            snapshot_Seq (Tensor): (currentBatchSize, seqLen, hiddenDim)
        Returns:
            nextLatentVec (Tensor): (currentBatchSize, seqLen, hiddenDim)
        """
        self.reset_hidden_states(for_batch=snapshot_Seq)

        # rl_out (currentBatchSize, seqLen, hiddenDim)
        # self.rl_h ((1, hiddenDim), (1, hiddenDim))
        # pdb.set_trace()
        rl_out, self.rl_h = self.rl(snapshot_Seq, (self.rl_h, self.rl_c))
        
        
        # rl_out, self.rl_h = self.rl(snapshot_Seq)
        
        # nextLatentVec = rl_out[:, -1]
        nextLatentVec = self.rl_h[0][0]
        nextLatentVec = self.l1(nextLatentVec)

        return nextLatentVec

