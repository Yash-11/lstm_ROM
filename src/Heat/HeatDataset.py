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
        timeStepsUnroll = self.hp.timeStepsUnroll


        # rawData = self.rawData.data.T  # (timeSteps, latentDim) 
        rawData = self.rawData.LatentVecs  # (timeSteps, latentDim)
        rawData, self.max, self.min = self.rescale(rawData, a = -1, b = 1)

        rawDataTrain = rawData[0 : numSampTrain +seq_len]
        rawDataTest = rawData[numSampTrain +seq_len : numSampTrain +seq_len + timeStepsUnroll+seq_len+1]
        pdb.set_trace()
        
        self.dataTrainX = T.zeros((numSampTrain, seq_len, latentDim))
        self.dataTrainY = T.zeros((numSampTrain, latentDim))
        self.dataTestX = T.zeros((numSampTest, seq_len, latentDim))
        self.dataTestY = T.zeros((numSampTest, timeStepsUnroll, latentDim))

        for i in range(numSampTrain):
            
            self.dataTrainX[i] = T.stack([rawDataTrain[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataTrainY[i] = rawDataTrain[i+seq_len,:]

        for i in range(numSampTest):
            
            self.dataTestX[i] = T.stack([rawDataTest[i+sl,:] for sl in range(seq_len)], dim=0)
            self.dataTestY[i] = T.stack([rawDataTest[i+seq_len+sl,:] for sl in range(timeStepsUnroll)], dim=0)
            # self.dataTestX[i] = T.stack([rawDataTrain[i+sl,:] for sl in range(seq_len)], dim=0)
            # self.dataTestY[i] = T.stack([rawDataTrain[i+seq_len+sl,:] for sl in range(timeStepsUnroll)], dim=0)


        # rawData = self.rawData.LatentVecs  # (timeSteps, latentDim)
        # rawData, self.max, self.min = self.rescale(rawData, a = -1, b = 1)

        
        # test_init_seq = 20
        # rawDataTrain = rawData[0 : numSampTrain*seq_len+1]
        # rawDataTest = rawData[numSampTrain*seq_len+1 : numSampTrain*seq_len+1 + timeStepsUnroll+test_init_seq+1]
        
        # self.dataTrainX = T.zeros((numSampTrain, seq_len, latentDim))
        # self.dataTrainY = T.zeros((numSampTrain, seq_len, latentDim))
        # self.dataTestX = T.zeros((numSampTest, test_init_seq, latentDim))
        # self.dataTestY = T.zeros((numSampTest, timeStepsUnroll, latentDim))
        # # self.dataTestX = T.zeros((numSampTest, seq_len, latentDim))
        # # self.dataTestY = T.zeros((numSampTest, seq_len, latentDim))

        # for i in range(numSampTrain):

        #     self.dataTrainX[i] = rawDataTrain[i*seq_len:i*seq_len+seq_len,:]
        #     self.dataTrainY[i] = rawDataTrain[i*seq_len+1:i*seq_len+seq_len+1,:]

        # for i in range(numSampTest):

        #     # self.dataTestX[i] = T.stack([rawDataTest[i+sl, :] for sl in range(test_init_seq)], dim=0)
        #     # self.dataTestY[i] = rawDataTest[i + test_init_seq: i + test_init_seq +timeStepsUnroll,:]
        #     # self.dataTestX[i] = rawDataTrain[i*seq_len:i*seq_len+seq_len,:]
        #     # self.dataTestY[i] = rawDataTrain[i*seq_len+1:i*seq_len+seq_len+1,:]

        #     # pdb.set_trace()
        #     i = 75
        #     self.dataTestX[0] = T.stack([rawDataTrain[i+sl, :] for sl in range(test_init_seq)], dim=0)
        #     self.dataTestY[0] = rawData[i + test_init_seq: i + test_init_seq +timeStepsUnroll,:]


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
