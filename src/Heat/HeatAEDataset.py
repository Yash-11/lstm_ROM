"""
Auto-Encoder Dataset
"""


import logging
import random
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


        rawData = self.rawData.data  # (maxNumTimeSteps, imDim)
        rawData, self.hp.meanAE, self.hp.stdAE = self.normalize(rawData)

        ln = rawData.shape[0]
        idxTr = list(range(ln))
        idxVal = random.sample(idxTr, self.hp.numSampData-self.hp.numSampValidAE)
        for i in idxVal: idxTr.remove(i)
        
        self.dataTrainX = rawData  # (maxNumTimeSteps, imDim)
        self.dataTrainY = rawData  # (maxNumTimeSteps, imDim)

        self.dataValidX = rawData[idxVal]  # (numSampValidAE, imDim)
        self.dataValidY = rawData[idxVal]  # (numSampValidAE, imDim)
        
        self.dataTestX = rawData[10:11]  # (numSampTest, imDim)
        self.dataTestY = rawData[10:11]  # (numSampTest, imDim)

        self.dataEncodeX = rawData  # (maxNumTimeSteps, imDim)


    def normalize(self, data):
        # (maxNumTimeSteps, imDim)
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