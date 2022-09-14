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

        if hp.AE_Model == 1:
            self.n_chLatent = 64
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(32, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(8, 1, kernel_size=3, stride=1, padding=1),
            )

        if hp.AE_Model == 2:
            self.n_chLatent = 32
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(32, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(8, 1, kernel_size=3, stride=1, padding=1),
            )
        if hp.AE_Model == 3:
            self.n_chLatent = 8
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(32, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(8, 1, kernel_size=3, stride=1, padding=1),
            )
        if hp.AE_Model == 4:
            self.n_chLatent = 1
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                
                nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(32, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(8, 1, kernel_size=3, stride=1, padding=1),
            )
        if hp.AE_Model == 5:
            self.n_chLatent = 1
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(8, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(                
                nn.ConvTranspose1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(32, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(8, 1, kernel_size=3, stride=1, padding=1),
            )

        if hp.AE_Model == 6:
            self.n_chLatent = 1
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(                
                nn.ConvTranspose1d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            )
        if hp.AE_Model == 7:
            self.n_chLatent = 1
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=5, stride=2, padding=2),

                nn.Conv1d(8, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=5, stride=2, padding=2),

                nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=5, stride=2, padding=2),

                nn.Conv1d(32, 1, kernel_size=5, stride=1, padding=2),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                
                nn.ConvTranspose1d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(32, 8, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(8, 1, kernel_size=5, stride=1, padding=2),
            )
        if hp.AE_Model == 8:
            self.n_chLatent = 1
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(8),
                nn.SiLU(),
                nn.AvgPool1d(kernel_size=5, stride=2, padding=2),

                nn.Conv1d(8, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.AvgPool1d(kernel_size=5, stride=2, padding=2),

                nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.AvgPool1d(kernel_size=5, stride=2, padding=2),

                nn.Conv1d(32, 1, kernel_size=5, stride=1, padding=2),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Upsample(scale_factor=2),
                
                nn.ConvTranspose1d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(32, 8, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(8),
                nn.SiLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose1d(8, 1, kernel_size=5, stride=1, padding=2),
            )

    def _forward(self, x):

        if x.shape[-1] != 1000:
            # x [1, currentBatchSize, latentDim*n_chLatent]
            x = x[0]
            n = x.shape[0]
            x = x.reshape((n, self.n_chLatent, -1))

            # x [currentBatchSize, n_chLatent, latentDim]
            # out [currentBatchSize, 1, imDim]
            out = self.decoder(x)
            out = out[:, 0][None]

            # out [1, currentBatchSize, imDim]
            return out
        else:
            # x [currentBatchSize, imDim]
            n = x.shape[0]
            y = self.encoder(x[:, None])
            z = self.decoder(y)
            return z[:, 0],y.reshape((n, -1))

    def forward(self, x):

        if x.shape[-1] != 1000:
            # x [currentBatchSize, latentDim*n_chLatent]
            x1 = x.reshape((tuple(x.shape[:-1]) + (self.n_chLatent, -1)))

            # x [currentBatchSize, n_chLatent, latentDim]
            # out [currentBatchSize, 1, imDim]
            out = self.decoder(x1)
            out = out[:, 0]

            # out [currentBatchSize, imDim]
            return out
        else:
            # x [currentBatchSize, imDim]
            n = x.shape[0]
            y = self.encoder(x[:, None])
            z = self.decoder(y)
            return z[:, 0],y.reshape((n, -1))
