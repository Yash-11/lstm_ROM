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

        self.H = hp.imH
        self.W = hp.imW
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
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
                # nn.BatchNorm2d(32),
                # nn.ReLU(),
                # nn.AvgPool2d(kernel_size=2, stride=2),

                # nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
            )

            self.decoder = nn.Sequential(
                # nn.ConvTranspose2d(1, 32, kernel_size=5, stride=1, padding=2),
                # nn.BatchNorm2d(32),
                # nn.ReLU(),
                # nn.Upsample(scale_factor=2),
                
                nn.ConvTranspose2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=2),
            )
        if hp.AE_Model == 8:
            self.n_chLatent = 1
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),

                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),

                nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                
                nn.ConvTranspose2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=2),
            )
        if hp.AE_Model == 9:
            self.n_chLatent = 1
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                
                nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),

                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            )


    def forward(self, x):

        if x.shape[-1] != self.W:
            # only decode

            # x [currentBatchSize, h, w]
            x = x[:, None]

            # x [currentBatchSize, n_chLatent, h, w]
            # out [currentBatchSize, 1, H, W]
            out = self.decoder(x)
            out = out[:, 0]

            # out [currentBatchSize, H, W]
            return out
        else:
            # auto encoder

            # x [currentBatchSize, H, W]
            n = x.shape[0]
            y = self.encoder(x[:, None])  # [currentBatchSize, 1, h, w]
            z = self.decoder(y)  # [currentBatchSize, 1, H, W]

            # return z[:, 0], y.reshape((n, -1))
            return z[:, 0], y[:, 0]
