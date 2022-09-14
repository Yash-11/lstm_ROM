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
from torch.nn.utils import weight_norm


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

        num_inputs = hp.seq_len
        num_channels = hp.num_channels
        output_size = 1
        kernel_size = hp.kernel_size
        dropout = hp.dropout
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1,
                                     padding=hp.padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.conv1d = nn.Conv1d(num_channels[-1], output_size, kernel_size=1)
        # self.linear = nn.Sequential(nn.Linear(num_channels[-1], 32),
        #                             nn.ReLU(),
        #                             nn.Linear(32, hp.output_size))

        self.loss = T.nn.MSELoss(reduction = 'sum')

    def forward(self, x):
        """
        Args:
            snapshot_Seq (Tensor): (currentBatchSize, seqLen, hiddenDim) #(currentBatchSize, hiddenDim, seqLen)
        Returns:
            output (Tensor): (currentBatchSize, hiddenDim)
        """
        # (currentBatchSize, seqLen, latenDim)
        out = self.network(x)  # (currentBatchSize, 1, latenDim)
        out = self.conv1d(out)
        out = T.tanh(out)
        return out[:, 0]


    def loss_fn(self, pred, target):
        return self.loss(pred, target)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, 
                                           dilation=dilation))
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, 
                                           dilation=dilation))
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU(init=0.5)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 
        out = self.net(x)
        # pdb.set_trace()
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
