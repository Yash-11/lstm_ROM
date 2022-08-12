
""" 
Plot predictions and reults.
"""


#%%
import pdb
import h5py
import torch as T
import numpy as np
from numpy.random import choice
from numpy.linalg import norm

from os.path import dirname, realpath, join
import sys

filePath = realpath(__file__)
experDir = dirname(realpath(__file__))
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

from src.Utils import Arguments, loadRunArgs, Dict2Class
from src.Paths import Paths
from src.Heat.HeatPlots import Plots
from src.Heat.HeatLoadData import LoadData


args = Arguments()
pathDict = {'run': 'firstTry1', 'data': f'./data'}
experPaths = Paths(experDir, args.os, pathDict)
hp = loadRunArgs(experPaths.run)

rawData = LoadData(hp, experPaths, args)

#%% ------------ load saved predictions during testing ---------------------

try:
    # predData = {}
    name = f'predHDataTest_epoch{hp.loadWeightsEpoch}_.hdf5'
    predData = h5py.File(join(experPaths.run, name), 'r')
    print(f'loaded pred data')
except:
    print(f'{join(experPaths.run, name)}')
    raise Exception(FileNotFoundError)


#%%
pred = predData['pred'][0]
target = predData['target'][0]
loss = np.mean(np.abs((pred - target)/target), 1)*100

timeStepsUnroll = hp.numSampTrain +hp.seq_len*2+ np.arange(0, hp.timeStepsUnroll, 10)

savePath = join(experPaths.run, f'PercentError')
plotParams = {'xlabel':'Time Step', 'ylabel': 'Percentage Error', 
            'xticks':np.arange(0, hp.timeStepsUnroll, 10), 'yticks':np.arange(20),
            'xticklabels':timeStepsUnroll, 'yticklabels':np.arange(20),
            'title':'Predicted Error', 'savePath':savePath}
plotData = loss

Plots().plotPercentError(plotData, Dict2Class(plotParams))

#%% -------------------- plot Pred with no noise ---------------------------

savePath = join(experPaths.run, f'ConvectionDiffusionpredPlot{0}_epoch{hp.loadWeightsEpoch}')
#choice([1, 2, 3], size=hp.numSampTest, replace=True)
plotParams = {'tStepModelPlot':[2]*hp.numSampTest, 'imDim': hp.imDim, 'tStepPlot':slice(0, hp.numSampTest, 2)}
plotData = predData[f'Sensors{16}_SNRdb{None}']
Plots().plotPred(plotData, Dict2Class(plotParams), savePath)

