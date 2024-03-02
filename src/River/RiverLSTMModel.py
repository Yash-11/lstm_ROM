"""
Network Architecture
"""

# import pandas as pd
from torch import linalg as LA
from typing import Union, Tuple, Optional, List

import torch as T
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear


import os.path as osp
SEED = 1234

T.manual_seed(SEED)
T.backends.cudnn.deterministic = True

class Model(nn.Module):

    def __init__(self, hp, args):
        super(Model, self).__init__()

        self.info = args.info
        self.device = args.device
        self.args = args
        self.latentDim = hp.latentDim
        self.hiddenDim = self.latentDim
        self.num_lstm_layers = hp.num_lstm_layers
        self.resetHidden = True

        self.rl = nn.LSTM(self.latentDim, self.hiddenDim, 
                        num_layers=self.num_lstm_layers, 
                        batch_first=True)
        # self.l1 = nn.Sequential(
        #     nn.Linear(hp.hiddenDim, hp.latentDim))

        self.loss = T.nn.MSELoss(reduction = 'sum')


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
        layers = self.num_lstm_layers
        T.manual_seed(1234)
        _ = T.nn.init.orthogonal_(T.zeros(layers, batch_size, self.hiddenDim))
        # print(_[:, :2, :5])
        return _


    def forward(self, snapshot_Seq):
        """
        Args:
            snapshot_Seq (Tensor): (currentBatchSize, seqLen, hiddenDim)
        Vars:
            output (Tensor): (currentBatchSize, seqLen, hiddenDim)
            h_n (Tensor): (num_layers, currentBatchSize, hiddenDim)
            c_n (Tensor): (num_layers, currentBatchSize, hiddenDim)
            hc (tuple): (h_n, c_n)
            Outputs (tuple): (output, (h_n, c_n))
        Returns:
            (currentBatchSize, hiddenDim)
        """
        self.reset_hidden_states(for_batch=snapshot_Seq) if self.resetHidden else None

        output, hc = self.rl(snapshot_Seq, (self.rl_h, self.rl_c))
        # out = self.l1(hc[0][-1])
        # return out

        return hc[0][-1]


    def loss_fn(self, pred, target):
        return self.loss(pred, target)
