
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
from src.Burgers.BurgersPlots import Plots
from src.Burgers.BurgersLoadData import LoadData


args = Arguments()
pathDict = {'run': 'firstTry', 'data': f'./data'}
experPaths = Paths(experDir, args.os, pathDict)
hp = loadRunArgs(experPaths.run)

rawData = LoadData(hp, experPaths, args)

#%% ------------ load saved predictions during testing ---------------------

try:
    # predData = {}
    name = f'predHDataTest_epoch{hp.loadWeightsEpoch}_.hdf5'
    predData = h5py.File(join(experPaths.run, name), 'r')
    print(f'loaded pred data {name}')
except:
    print(f'{join(experPaths.run, name)}')
    raise Exception(FileNotFoundError)


#%%
pred = predData['pred'][0]
target = predData['target'][0]
# pdb.set_trace()
loss = np.mean(np.abs((pred - target)/target), 1)*100

timeStepsUnroll = hp.numSampTrain +hp.seq_len*2+ np.arange(0, hp.timeStepsUnroll, 10)

savePath = join(experPaths.run, f'PercentError')
plotParams = {'xlabel':'Time Step', 'ylabel': 'Percentage Error', 
            'xticks':np.arange(0, hp.timeStepsUnroll, 10), 'yticks':np.arange(300),
            'xticklabels':timeStepsUnroll, 'yticklabels':np.arange(300),
            'title':'Predicted Error', 'savePath':savePath}
plotData = loss

Plots().plotPercentError(plotData, Dict2Class(plotParams))

#%% -------------------- plot Pred with no noise ---------------------------

from src.Burgers.BurgersPlots import Plots
savePath = join(experPaths.run, f'BurgerspredPlot{0}_epoch{hp.loadWeightsEpoch}')
#choice([1, 2, 3], size=hp.numSampTest, replace=True)
plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
plotData = {'pred': pred, 'target': target}
Plots().plotPred(plotData, Dict2Class(plotParams), savePath) 

savePath = join(experPaths.run, f'BurgerspredimPlot{0}_epoch{hp.loadWeightsEpoch}')
Plots().implotPred(plotData, Dict2Class(plotParams), savePath)

#%% --------------------- calculate L2 Error -------------------------------

idx = 2; M=1
plotData = np.zeros((len(SensorsLs), M, hp.numSampTest, len(SNRdbLs)))

for s, Sensors in enumerate(SensorsLs):
    for i, SNRdb in enumerate(SNRdbLs):
        
        pred = predData[f'Sensors{Sensors}_SNRdb{SNRdb}']['pred'][:, idx]  # (numSampTest, M, numNodes)
        target = predData[f'Sensors{Sensors}_SNRdb{SNRdb}']['target'][:, idx]

        l2Error = np.zeros(pred.shape[:-1])
        for j in range(hp.numSampTest):
            for k in range(pred.shape[1]):
                l2Error[j, k] = norm(target[j, k] - pred[j, k]) / norm(target[j, k])
    
        plotData[s, 0, :, i] = l2Error[:, 0].reshape((-1))
    

#%% --------------------- violin plot of L2 error --------------------------

for s, Sensors in enumerate(SensorsLs):
    savePath = join(experPaths.run, f'ConvectionDiffusionViolinPlot{0}Sensors{Sensors}_epoch{hp.loadWeightsEpoch}') 
    plotParams = {
        'xticks': [10, 20, 30, 40],
        'xticklabels': [10, 20, 60, 'None'],
        'xticksPlot': [[10, 20, 30, 40]],
        'ylabel': 'Error',
        'xlabel': 'SNRdb',
        'title': f'Sensors: {Sensors}',
        'label': ['U'],
        'facecolor': ['green']
    }
    Plots().violinplot(plotData[s], Dict2Class(plotParams), savePath)


# %%
