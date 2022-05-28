
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
from src.ConvectionDiffusion.ConvectionDiffusionPlots import Plots
from src.ConvectionDiffusion.ConvectionDiffusionLoadData import LoadData


args = Arguments()
pathDict = {'run': 'firstTry', 'data': f'../../src/ConvectionDiffusion/solver/pseudoSpectral_ConvectionDiffusion'}
experPaths = Paths(experDir, args.os, pathDict)
hp = loadRunArgs(experPaths.run)

rawData = LoadData(args, experPaths)
SensorsLs = hp.numSensorTestLs
SNRdbLs = hp.noiseLs


#%% ------------ load saved predictions during testing ---------------------

try:
    predData = {}
    for j, Sensors in enumerate(SensorsLs):
        for i, SNRdb in enumerate(SNRdbLs):
            name = f'predDataTest_epoch{hp.loadWeightsEpoch}_Sensors{Sensors}_SNRdb{SNRdb}.hdf5'
            predData[f'Sensors{Sensors}_SNRdb{SNRdb}'] = h5py.File(join(experPaths.run, name), 'r')
    print(f'loaded pred data')
except:
    print(f'{join(experPaths.run, name)}')
    raise Exception(FileNotFoundError)


#%% -------------------- plot Pred with no noise ---------------------------

savePath = join(experPaths.run, f'ConvectionDiffusionpredPlot{0}_epoch{hp.loadWeightsEpoch}')
#choice([1, 2, 3], size=hp.numSampTest, replace=True)
plotParams = {'tStepModelPlot':[2]*hp.numSampTest, 'imDim': rawData.imDim, 'tStepPlot':slice(0, hp.numSampTest, 2)}
plotData = predData[f'Sensors{16}_SNRdb{None}']
Plots().plotPred(plotData, Dict2Class(plotParams), savePath)


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
