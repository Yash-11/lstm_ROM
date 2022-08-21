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
        Vars:
            self.rawData.data (Tensor): (latentDim, timeSteps)
        """
        self.use = use
        self.device = device
        self.info = info

        self.rawData = rawData
        self.ep = experPaths
        self.hp = hyperParams


        seq_len = self.hp.seq_len
        numSampTrain = self.hp.numSampTrain
        numSampValid = self.hp.numSampValid
        numSampTest = self.hp.numSampTest
        latentDim = self.hp.latentDim
        timeStepsUnroll = self.hp.timeStepsUnroll


        # rawData = self.rawData.data.T  # (timeSteps, latentDim) 
        rawData = self.rawData.LatentVecs  # (timeSteps, latentDim)
        rawData, self.max, self.min = self.rescale(rawData, a = -1, b = 1)

        rawDataTrain = rawData[0 : numSampTrain +seq_len]
        rawDataTest = rawData[numSampTrain +seq_len : numSampTrain +seq_len + timeStepsUnroll+seq_len+1]
        rawDataValid = rawDataTest
        
        self.dataTrainX = T.zeros((numSampTrain, seq_len, latentDim))
        self.dataTrainY = T.zeros((numSampTrain, latentDim))
        self.dataValidX = T.zeros((numSampValid, seq_len, latentDim))
        self.dataValidY = T.zeros((numSampValid, latentDim))
        self.dataTestX = T.zeros((numSampTest, seq_len, latentDim))
        self.dataTestY = T.zeros((numSampTest, timeStepsUnroll, latentDim))

        for i in range(numSampTrain):
            
            self.dataTrainX[i] = T.stack([rawDataTrain[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataTrainY[i] = rawDataTrain[i+seq_len,:]

        for i in range(numSampValid):
            
            self.dataValidX[i] = T.stack([rawDataValid[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataValidY[i] = rawDataValid[i+seq_len,:]

        for i in range(numSampTest):
            
            self.dataTestX[i] = T.stack([rawDataTest[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataTestY[i] = T.stack([rawDataTest[i+seq_len+sl,:] for sl in range(timeStepsUnroll)], dim=0)
            # self.dataTestX[i] = T.stack([rawDataTrain[i+sl,:] for sl in range(seq_len)], dim=0)
            # self.dataTestY[i] = T.stack([rawDataTrain[i+seq_len+sl,:] for sl in range(timeStepsUnroll)], dim=0)


    def rescale(self, data, a=-1, b=1):
        min = data.min(); max = data.max()
        data_ = (data - min)*(b-a)/(max-min)+a
        return data_, max, min

    
    def __len__(self):
        len = self.hp.numSampTrain if self.use == 'train' else self.hp.numSampTest
        return len


    def __getitem__(self, idx):
        d = self.device

        if self.use == 'train':
            return self.dataTrainX[idx].to(d), self.dataTrainY[idx].to(d)
        else:
            return self.dataTestX[idx].to(d), self.dataTestY[idx].to(d)
