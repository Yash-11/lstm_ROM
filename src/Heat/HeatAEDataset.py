"""
Auto-Encoder Dataset
"""

import pdb
import logging
from numpy import random
from torch.utils.data.dataset import Dataset
import numpy as np
from numpy.random import choice
import torch as T

import os
from os.path import dirname, realpath, join, exists
import sys
filePath = realpath(__file__)
srcDir = dirname(dirname(filePath))
sys.path.append(srcDir)

  


class AEDatasetClass(Dataset):

    def __init__(self, rawData, use, experPaths, hyperParams, device='cpu', info=print):
        """
        Vars:
            self.rawData.data (Tensor): (latentDim, timeSteps)
        """
        self.use = use
        self.device = device
        self.info = info

        self.rawData = rawData
        self.ep = experPaths
        self.hp = hyperParams


        rawData = self.rawData.data.T  # (maxNumTimeSteps, imDim)
        rawData, self.hp.meanAE, self.hp.stdAE = self.normalize(rawData)

        self.dataTrainX = rawData  # (maxNumTimeSteps, imDim)
        self.dataTrainY = rawData  # (maxNumTimeSteps, imDim)

        self.dataTestX = rawData[10:11]  # (numSampTest, imDim)
        self.dataTestY = rawData[10:11]  # (numSampTest, imDim)

        self.dataEncodeX = rawData  # (maxNumTimeSteps, imDim)


    def normalize(self, data):
        # (maxNumTimeSteps, imDim)
        m = T.mean(data).item()
        s = T.std(data).item()
        return (data - m)/s, m, s

    
    def __len__(self):
        len = self.hp.maxNumTimeSteps
        return len


    def __getitem__(self, idx):
        d = self.device
        if self.use == 'train':
            return self.dataTrainX[idx].to(d), self.dataTrainY[idx].to(d)
        elif self.use == 'test':
            return self.dataTestX[idx].to(d), self.dataTestY[idx].to(d)
        elif self.use == 'encode':
            return self.dataEncodeX[idx].to(d)