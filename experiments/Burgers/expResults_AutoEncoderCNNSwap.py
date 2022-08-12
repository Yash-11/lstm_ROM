 #%%

import pdb
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
# from src.Pipeline import ModelPipeline
from src.EnsemblePipeline import ModelPipeline
from src.AEPipeline import AEPipeline
from src.Paths import Paths

from src.Burgers.BurgersDataset import DatasetClass
from src.Burgers.BurgersLoadData import LoadData
from src.Burgers.BurgersPlots import Plots

from src.Burgers.BurgersAEDataset import AEDatasetClass
from src.Burgers.BurgersCAEModel import AutoEncoder


T.manual_seed(0)
np.random.seed(0)
random.seed(0)


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/BurgersTCN'
    ep.run = f'{runName}'

def round_neatest(x, s=50):
    r = x%s
    if r<=s//2: return x-r 
    elif r>s//2: return x+(s-r)

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
    if hp.reduce: aePipeline.decodeLatentVecs()

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

    # --------------------------------------------------------------------------
    #                        l2 relative error plot

    pred = predData['pred'][0]
    target = predData['target'][0]
    
    loss = LA.norm((pred - target), axis=1) / LA.norm(target, axis=1) *100

    timeStepsUnroll = hp.numSampTrain +hp.seq_len*2+ np.arange(0, hp.timeStepsUnroll, 10)

    savePath = join(experPaths.run, f'Error_epoch{hp.loadWeightsEpoch}')
    plotParams = {'xlabel':'Time Step', 'ylabel': 'Percentage Error', 
                'xticks':np.arange(0, hp.timeStepsUnroll, 10), 'yticks':np.arange(300),
                'xticklabels':timeStepsUnroll, 'yticklabels':np.arange(300),
                'title':'Predicted Error', 'savePath':savePath}
    plotData = loss

    Plots().plotPercentError(plotData, Dict2Class(plotParams))

    # --------------------------------------------------------------------------
    #                        image plot for prediction

    savePath = join(experPaths.run, f'BurgerspredPlot{0}_epoch{hp.loadWeightsEpoch}')
    plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
    plotData = {'pred': pred, 'target': target}
    savePath = join(experPaths.run, f'BurgerspredimPlot{0}_epoch{hp.loadWeightsEpoch}')
    Plots().implotPred(plotData, Dict2Class(plotParams), savePath)

    # --------------------------------------------------------------------------
    #                        graph plot for prediction

    for timestepplot in [20, 40, 60]:
      savePath = join(experPaths.run, f'BurgerspredsinglesnapPlot{timestepplot}_epoch{hp.loadWeightsEpoch}')
      plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
      plotData = {'pred': pred[hp.numSampTrain+hp.seq_len-1:], 'target': target[hp.numSampTrain+hp.seq_len-1:]}
      # plotData = {'pred': pred, 'target': target}
  
      Plots().plotPredSingle(plotData, Dict2Class(plotParams), savePath, timestepplot)


# %% ---------------------------------------------------------------------------
#                           load required model

# from src.Burgers.BurgersCNNModel import Model
# from src.Burgers.BurgersLSTMModel import Model
from src.Burgers.BurgersCNNEnsembleModel import Model


# %% ---------------------------------------------------------------------------
#                       test all combinations in minLoss.csv

# path = join(experDir, f'minLoss.csv')
# df =  pd.read_csv(path)
# df = df.reset_index()

# for index, row in df.iterrows():
#     runName = row['name']
#     minValidEpoch = round_neatest(row['minValidEpoch'])
#     results(runName, minValidEpoch)

# %% ---------------------------------------------------------------------------
#                           test particular run

# results('50results_AE_CNNSwap_ld4_sql10_krs3_lr0.0003_trSmp150_ch5050_bs16_8PA67', 1300)
# results('50results_AE_CNNSwap_ld4_sql10_krs3_lr0.0003_trSmp150_ch5050_bs16_8PA67', 600)
# results('results_ld6_sql20_layers1_lr0.0003_trSmp250_bs108PA67', 1000)

results('NO_reluBN_results_AE_CNNSwap_Re600_ld4_sql10_krs3_lr0.0003_trSmp150_ch5050_bs16_8PA67', 3000)
results('silu_results_AE_CNNSwap_Re600_ld4_sql10_krs3_lr0.0003_trSmp150_ch5050_bs16_8PA67', 3000)