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
from src.Pipeline import ModelPipeline
from src.AEPipeline import AEPipeline
from src.Paths import Paths

from src.Stoker.StokerLoadData import LoadData
from src.Stoker.StokerPlots import Plots

from src.Stoker.StokerAEDataset import AEDatasetClass
from src.Stoker.StokerCAEModel import AutoEncoder

from src.Stoker.StokerDataset import DatasetClass
from src.Stoker.StokerTCNModel import Model


T.manual_seed(0)
np.random.seed(0)
random.seed(0)


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/StokerTCN'
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

    # modelPipeline.test()
    # if hp.reduce: aePipeline.decodeLatentVecs()

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)


    # --------------------------------------------------------------------------
    #                  load saved predictions during testing

    try:
        if hp.reduce:
            name = f'predHDataTest_epoch{hp.loadWeightsEpoch}_.hdf5'
        else:
            name = f'predDataTest_epoch{hp.loadWeightsEpoch}_.hdf5'
        predData = h5py.File(join(experPaths.run, name), 'r')
        print(f'loaded pred data {name}')
    except:
        print(f'{join(experPaths.run, name)}')
        raise Exception(FileNotFoundError)

    pred = predData['pred'][0]
    target = predData['target'][0]
    loss = LA.norm((pred - target), axis=1) / LA.norm(target, axis=1) *100
    timeStepsUnroll = hp.numSampTrain +hp.seq_len*2+ np.arange(0, hp.timeStepsUnroll, 10)

    # --------------------------------------------------------------------------
    #                        image plot for prediction

    savePath = join(experPaths.run, f'CAE_TCN_Stok_imgPred{0}_epoch{hp.loadWeightsEpoch}')
    plotParams = {'tStepModelPlot':[2]*hp.numSampTest, 'v_min':hp.dataMin, 'v_max':hp.dataMax}
    plotData = {'data': pred}
    Plots().imgPlot(plotData, Dict2Class(plotParams), savePath)

    savePath = join(experPaths.run, f'CAE_TCN_Stok_imgTar{0}_epoch{hp.loadWeightsEpoch}')
    plotParams = {'tStepModelPlot':[2]*hp.numSampTest, 'v_min':hp.dataMin, 'v_max':hp.dataMax}
    plotData = {'data': target}
    Plots().imgPlot(plotData, Dict2Class(plotParams), savePath)
    exit()
    
    # --------------------------------------------------------------------------
    #                        l2 relative error plot

    savePath = join(experPaths.run, f'CAE_TCN_Stok_Error_epoch{hp.loadWeightsEpoch}')
    plotParams = {'xlabel':'Time Step', 'ylabel': 'Percentage Error', 
                'xticks':np.arange(0, hp.timeStepsUnroll, 10), 'yticks':np.arange(300),
                'xticklabels':timeStepsUnroll, 'yticklabels':np.arange(300),
                'title':'Predicted Error', 'savePath':savePath}
    plotData = loss

    Plots().plotPercentError(plotData, Dict2Class(plotParams))

    # --------------------------------------------------------------------------
    #                        image plot for prediction

    savePath = join(experPaths.run, f'CAE_TCN_Stok_predPlot{0}_epoch{hp.loadWeightsEpoch}')
    plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
    plotData = {'pred': pred, 'target': target}
    savePath = join(experPaths.run, f'CAE_TCN_Stok_predimPlot{0}_epoch{hp.loadWeightsEpoch}')
    Plots().implotPred(plotData, Dict2Class(plotParams), savePath)

    # --------------------------------------------------------------------------
    #                        graph plot for prediction

    for timestepplot in [50, 100, 150]:
      savePath = join(experPaths.run, f'CAE_TCN_Stok_predgraphPlot{timestepplot}_epoch{hp.loadWeightsEpoch}')
      plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
      plotData = {'pred': pred[hp.numSampTrain+hp.seq_len-1:], 'target': target[hp.numSampTrain+hp.seq_len-1:]}
      # plotData = {'pred': pred, 'target': target}
  
      Plots().plotPredSingle(plotData, Dict2Class(plotParams), savePath, timestepplot)


path = join(experDir, f'minLoss.csv')
df =  pd.read_csv(path)
df = df.reset_index()

# %% ---------------------------------------------------------------------------
#                       test all combinations in minLoss.csv

# for index, row in df.iterrows():
#     results(row['name'], row['minValidEpoch'])

# %% ---------------------------------------------------------------------------
#                           test particular run

name = 'results_CAE_TCN_ld125_sql20_krs3_lr0.0003_ch200200200_bs15_QJP0S'
# minValidEpoch = df.loc[df['name'] == name, 'minValidEpoch'].values[0] 
# results(name, int(minValidEpoch))
results(name, 1050)
