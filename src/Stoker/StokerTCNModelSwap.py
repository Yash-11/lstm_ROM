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
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        # self.linear = nn.Conv1d(output_size, output_size)
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
        # out = self.conv1d(out[:, :, -1:])[:, :, 0]
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
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()#nn.ReLU()
        # nn.PReLU
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, 
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()#nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()#nn.ReLU()
        self.init_weights()

    # def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    #     super(TemporalBlock, self).__init__()
    #     self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
    #                                        stride=stride, padding=padding, 
    #                                        dilation=dilation)
    #     self.chomp1 = Chomp1d(padding)
    #     self.relu1 = nn.ReLU()#.nn.Softplus()#nn.SELU()
    #     self.dropout1 = nn.Dropout(dropout)

    #     self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
    #                                        stride=stride, padding=padding, 
    #                                        dilation=dilation)
    #     self.chomp2 = Chomp1d(padding)
    #     self.relu2 = nn.ReLU()#.nn.Softplus()#nn.SELU()
    #     self.dropout2 = nn.Dropout(dropout)

    #     self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
    #                              self.conv2, self.chomp2, self.relu2, self.dropout2)
    #     self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    #     self.relu = nn.ReLU()#.nn.Softplus()#nn.SELU()
    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.xavier_uniform_(m.weight)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    # def init_weights(self):
    #     nn.init.xavier_uniform_(self.conv1.weight)
    #     nn.init.xavier_uniform_(self.conv1.weight)
    #     if self.downsample is not None:
    #         nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        # return out
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
