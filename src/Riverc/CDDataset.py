"""
Dataset
"""


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

  


class DatasetClass(Dataset):

    def __init__(self, rawData, use, experPaths, hyperParams, device='cpu', info=print):
        """
        Args:
            rawData.data (Tensor): (timeSteps, H, W)
            use (str): 'train' or 'test' or 'valid'
            experPaths (class): useful paths
        """

        self.use = use
        self.device = device
        self.info = info

        self.rawData = rawData
        self.ep = experPaths
        self.hp = hyperParams


        seq_len = self.hp.seq_len
        numSampTrain = self.hp.numSampTrain
        numSampTest = self.hp.numSampTest
        numSampValid = self.hp.numSampValid
        timeStepsUnroll = self.hp.timeStepsUnroll


        if self.hp.reduce:
            rawData = self.rawData.LatentVecs  # (timeSteps, h, w)
        else:
            rawData = self.rawData.data  # (timeSteps, H, W)
        
        h, w = rawData.shape[1], rawData.shape[2]
        rawData, self.max, self.min = self.rescale(rawData, a = -0.5, b = 0.5)

        rawDataTrain = rawData[0 : numSampTrain +seq_len]
        rawDataTest = rawData[numSampTrain : numSampTrain + timeStepsUnroll+seq_len+1]
        rawDataValid = rawDataTest
        
        self.dataTrainX = T.zeros((numSampTrain, seq_len, h, w))
        self.dataTrainY = T.zeros((numSampTrain, h, w))
        
        self.dataValidX = T.zeros((numSampValid, seq_len, h, w))
        self.dataValidY = T.zeros((numSampValid, h, w))
    
        self.dataTestX = T.zeros((numSampTest, seq_len, h, w))
        self.dataTestY = T.zeros((numSampTest, timeStepsUnroll, h, w))

        for i in range(numSampTrain):
            
            self.dataTrainX[i] = T.stack([rawDataTrain[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataTrainY[i] = rawDataTrain[i+seq_len,:]

        for i in range(numSampValid):
            
            self.dataValidX[i] = T.stack([rawDataValid[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataValidY[i] = rawDataValid[i+seq_len,:]
        
        for i in range(numSampTest):
            
            # self.dataTestX[i] = T.stack([rawDataTest[i+sl,:] for sl in range(seq_len)], dim=0)
            # self.dataTestY[i] = T.stack([rawDataTest[i+seq_len+sl,:] for sl in range(timeStepsUnroll)], dim=0)
            
            self.dataTestX[i] = T.stack([rawData[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataTestY[i] = T.stack([rawData[i+seq_len+sl,:] for sl in range(timeStepsUnroll)], dim=0)


    def rescale(self, data, a=-1, b=1):
        min = data.min(); max = data.max()
        data_ = (data - min)*(b-a)/(max-min)+a
        return data_, max, min

    
    def __len__(self):
        if self.use == 'train': len = self.hp.numSampTrain
        elif self.use == 'test': len = self.hp.numSampTest
        elif self.use == 'valid': len = self.hp.numSampValid
        return len

        
    def __getitem__(self, idx):
        d = self.device

        if self.use == 'train':
            return self.dataTrainX[idx].to(d), self.dataTrainY[idx].to(d)
        else:
            return self.dataTestX[idx].to(d), self.dataTestY[idx].to(d)
