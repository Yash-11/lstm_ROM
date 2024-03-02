 #%%


import logging
import torch as T
import itertools
import json
import random
import string
import h5py
import numpy as np
import pandas as pd
import numpy.linalg as LA

import sys
from os.path import dirname, realpath, join

filePath = realpath(__file__)
experDir = dirname(realpath(__file__))
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

#%%'
from src.Utils import Arguments, Dict2Class, startSavingLogs, save_args, loadRunArgs
from src.EnsemblePipeline import ModelPipeline
from src.AEPipelineEnsemble import AEPipeline
from src.Paths import Paths

from src.CD.CDLoadData import LoadData
from src.CD.CDPlots import Plots

from src.CD.CDAEDataset import AEDatasetClass
from src.CD.CDCAEModel import AutoEncoder

from src.CD.CDDataset import DatasetClass
from src.CD.CDCNNEnsembleModel import Model


T.manual_seed(0)
np.random.seed(0)
random.seed(0)


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/CDTCN'
    ep.run = f'{runName}'


def results(runName, minValidEpoch):
    
    args = Arguments()
    logger = logging.getLogger('my_module')
    logger.setLevel(logging.DEBUG)

    # set hyper params for run
    class HyperParams: pass
    hp = HyperParams()

    experPaths = Paths(experDir, args.os)
    addPaths(experPaths, runName)

    # load saved hyper params for testing from old runs
    hp = loadRunArgs(experPaths.run)
    hp.loadWeightsEpoch = minValidEpoch

    startSavingLogs(args, experPaths.run, logger)
    rawData = LoadData(hp, experPaths, args)    

    # test 
    if hp.reduce:
        aePipeline = AEPipeline(AutoEncoder, hp, experPaths, rawData, AEDatasetClass, args)
        rawData.loadLatentVecs()
    
    modelPipeline = ModelPipeline(Model, hp, experPaths, rawData, DatasetClass, args)
    hp.predData_Info = f'_'

    modelPipeline.test()
    if hp.reduce: aePipeline.decodeLatentVecDistributions()
    # if hp.reduce: aePipeline.decodeLatentVecs()

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)


    # --------------------------------------------------------------------------
    #                  load saved predictions during testing

    loadWeightsEpoch = '_'.join(str(e) for e in hp.loadWeightsEpoch)
    # loadWeightsEpoch = 7950
    try:
        if hp.reduce: 
            name = f'predHDataTest_epoch{loadWeightsEpoch}_.hdf5'
        else:
            name = f'predDataTest_epoch{loadWeightsEpoch}_.hdf5'
        predData = h5py.File(join(experPaths.run, name), 'r')
        print(f'loaded pred data {name}')
    except:
        print(f'{join(experPaths.run, name)}')
        raise Exception(FileNotFoundError)

    # --------------------------------------------------------------------------
    #                        l2 relative error plot

    pred = predData['pred'][:]
    target = predData['target'][:]
    var = predData['var'][:]
    var = var**0.5
    # var = np.zeros_like(pred)
    aa = pred[:].reshape((hp.timeStepsUnroll, -1))
    bb = target[:].reshape((hp.timeStepsUnroll, -1))

    # loss = LA.norm((pred - target), axis=1) / LA.norm(target, axis=1) *100
    loss = LA.norm((aa - bb), axis=1) / LA.norm(bb, axis=1) *100

    timeStepsUnroll = hp.numSampTrain +hp.seq_len*2+ np.arange(0, hp.timeStepsUnroll, 10)

    savePath = join(experPaths.run, f'CAE_CNNEns_Stok_Error_epoch{loadWeightsEpoch}')
    plotParams = {'xlabel':'Time Step', 'ylabel': 'Percentage Error', 
                'xticks':np.arange(0, hp.timeStepsUnroll, 10), 'yticks':np.arange(300),
                'xticklabels':timeStepsUnroll, 'yticklabels':np.arange(300),
                'title':'Predicted Error', 'savePath':savePath}
    plotData = loss

    Plots().plotPercentError(plotData, Dict2Class(plotParams))

    # --------------------------------------------------------------------------
    #                        image plot for prediction

    # savePath = join(experPaths.run, f'CAE_CNNEns_Stok_predPlot{0}_epoch{loadWeightsEpoch}')
    # plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
    # plotData = {'pred': pred, 'target': target}
    # savePath = join(experPaths.run, f'CAE_CNNEns_Stok_predimPlot{0}_epoch{loadWeightsEpoch}')
    # Plots().implotPred(plotData, Dict2Class(plotParams), savePath)

    # --------------------------------------------------------------------------
    #                        graph plot for prediction

    for timestepplot in [60, 110, 160, 210, 240]:
        savePath = join(experPaths.run, f'CAE_CNNEns_Stok_predgraphPlot{timestepplot}_epoch{loadWeightsEpoch}')
        plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
        plotData = {'pred': pred, 
                    'target': target,
                    'var': var}

        Plots().plotPredSingleVar(plotData, Dict2Class(plotParams), savePath, timestepplot)


path = join(experDir, f'minLoss.csv')
df =  pd.read_csv(path)
df = df.reset_index()

# %% ---------------------------------------------------------------------------
#                       test all combinations in minLoss.csv

for index, row in df.iterrows():
    results(row['name'], row['minValidEpoch'])

# %% ---------------------------------------------------------------------------
#                           test particular run

# name = 'results_CAE_CNNEns_sql20_krs3_lr5e-05_trSmp250_ch100100100_bs15_8PA67'
# minValidEpoch = []

# for i in range(10):
#     name1 = name+f'model{i}'
#     minValidEpoch_i = df.loc[df['name'] == name1, 'minValidEpoch'].values[0] 
#     minValidEpoch.append(int(minValidEpoch_i))

# results(name, minValidEpoch)

# results('resultsOP_CAE_CNNEns_sql20_krs3_lr5e-05_trSmp250_ch100100100_bs15_8PA67', [2500])
