"""
Dataset
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
        numSampTest = self.hp.numSampTest
        latentDim = self.hp.latentDim


        rawData = self.rawData.data.T  # (timeSteps, latentDim)
        rawDataTrain = rawData[0 : numSampTrain +seq_len]
        rawDataTest = rawData[numSampTrain +seq_len : numSampTrain +seq_len + numSampTest +seq_len]
        
        self.dataTrainX = T.zeros((numSampTrain, seq_len, latentDim))
        self.dataTrainY = T.zeros((numSampTrain, latentDim))
        self.dataTestX = T.zeros((numSampTest, seq_len, latentDim))
        self.dataTestY = T.zeros((numSampTest, latentDim))

        for i in range(numSampTrain):
            
            self.dataTrainX[i] = T.stack([rawDataTrain[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataTrainY[i] = rawDataTrain[i+seq_len,:]

        for i in range(numSampTest):
            
            self.dataTestX[i] = T.stack([(rawDataTest[i+sl,:]) for sl in range(seq_len)], dim=0)
            self.dataTestY[i] = rawDataTest[i+seq_len,:]



    def __len__(self):
        len = self.hp.numSampTrain if self.use == 'train' else self.hp.numSampTest
        return len


    def __getitem__(self, idx):
        d = self.device

        if self.use == 'train':
            return self.dataTrainX[idx].to(d), self.dataTrainY[idx].to(d)
        else:
            return self.dataTestX[idx].to(d), self.dataTestY[idx].to(d)
