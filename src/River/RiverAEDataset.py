"""
Auto-Encoder Dataset
"""


import logging
import random
from torch.utils.data.dataset import Dataset
import numpy as np
from numpy.random import choice
import torch as T
import pdb

import os
from os.path import dirname, realpath, join, exists
import sys
filePath = realpath(__file__)
srcDir = dirname(dirname(filePath))
sys.path.append(srcDir)

  


class AEDatasetClass(Dataset):

    def __init__(self, rawData, use, experPaths, hyperParams, device='cpu', info=print):
        """
        Args:
            rawData (class): class with data as attribute
        """
        self.use = use
        self.device = device
        self.info = info

        self.rawData = rawData
        self.ep = experPaths
        self.hp = hyperParams


        rawData = self.rawData.data  # (maxNumTimeSteps, H, W)
        rawData, self.hp.meanAE, self.hp.stdAE = self.normalize(rawData)
        len = rawData.shape[0]

        idxTr = list(range(len))
        idxVal = random.sample(idxTr, self.hp.numSampValidAE)
        for i in idxVal: idxTr.remove(i)
        
        self.dataTrainX = rawData[idxTr]  # (numSampTrainAE, H, W)
        self.dataTrainY = rawData[idxTr]  # (numSampTrainAE, H, W)

        self.dataValidX = rawData[idxVal]  # (numSampValidAE, H, W)
        self.dataValidY = rawData[idxVal]  # (numSampValidAE, H, W)

        nTest = self.hp.numSampTestAE
        idx = np.arange(0, len, int((len-1e-1)//nTest))[1:nTest+1]
        # print(idx)
        self.dataTestX = rawData[idx]  # (numSampTestAE, H, W)
        self.dataTestY = rawData[idx]  # (numSampTestAE, H, W)

        self.dataEncodeX = rawData  # (maxNumTimeSteps, H, W)
        # pdb.set_trace()


    def normalize(self, data):
        # (maxNumTimeSteps, h, w)
        m = T.mean(data).item()
        s = T.std(data).item()
        return (data - m)/s, m, s

    
    def __len__(self):
        
        if self.use == 'train': len = self.hp.numSampTrainAE
        elif self.use == 'test': len = self.hp.numSampTestAE
        elif self.use == 'encode': len = self.hp.maxNumTimeSteps
        elif self.use == 'valid': len = self.hp.numSampValidAE
        return len


    def __getitem__(self, idx):
        d = self.device
        if self.use == 'train':
            return self.dataTrainX[idx].to(d), self.dataTrainY[idx].to(d)
        elif self.use == 'test':
            return self.dataTestX[idx].to(d), self.dataTestY[idx].to(d)
        elif self.use == 'encode':
            return self.dataEncodeX[idx].to(d)