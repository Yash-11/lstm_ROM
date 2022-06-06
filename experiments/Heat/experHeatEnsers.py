"""
Main script for experiment 'Basic'
for setting params and training/testing
"""

import pdb
import logging
import torch as T

import sys
from os.path import dirname, realpath, join

filePath = realpath(__file__)
experDir = dirname(realpath(__file__))
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

from src.Utils import Parser, startSavingLogs, save_args
from src.Pipeline import ModelPipeline
from src.AEPipeline import AEPipeline
from src.Paths import Paths

from src.Heat.HeatDataset import DatasetClass
from src.Heat.HeatAEDataset import AEDatasetClass
from src.Heat.HeatLSTMModel import Model
from src.Heat.HeatAEModel import AutoEncoder
from src.Heat.HeatLoadData import LoadData



# def setHyperParams(hp):
#     # model 
#     hp.hiddenDim = 128
#     hp.latentDim = 81
#     hp.seq_len = 7

#     # training
#     hp.numIters = 5001
#     hp.lr = 0.0005
#     hp.batchSizeTrain = 45
#     hp.epochStartTrain = 0

#     # testing
#     hp.loadWeightsEpoch = 5000
#     hp.batchSizeTest = 1
#     hp.timeStepsUnroll = 25

#     # data
#     hp.numSampTrain = 45
#     hp.numSampTest = 1

#     # logging
#     hp.save = 1
#     hp.show = 0
#     hp.saveLogs = 1
#     hp.saveInterval = 20
#     hp.logInterval = 100
#     hp.checkpointInterval = 5000

def setHyperParams(hp):
    # model 
    hp.hiddenDim = 8
    hp.latentDim = 8
    hp.seq_len = 5

    # training
    hp.numIters = 4001
    hp.lr = 0.00034
    hp.batchSizeTrain = 5
    hp.epochStartTrain = 0

    # testing
    hp.loadWeightsEpoch = 4000
    hp.batchSizeTest = 1
    hp.timeStepsUnroll = 100

    # data
    hp.numSampTrain = 40
    hp.numSampTest = 1

    # logging
    hp.save = 1
    hp.show = 0
    hp.saveLogs = 1
    hp.saveInterval = 20
    hp.logInterval = 100
    hp.checkpointInterval = 500

    # AEtraining
    hp.numItersAE = 1001
    hp.lrAE = 0.0003
    hp.batchSizeTrainAE = 50
    hp.epochStartTrainAE = 0

    # AEtesting
    hp.loadAEWeightsEpoch = 1000
    hp.batchSizeTestAE = 1
    hp.batchSizeEncode = 500

    # AEdata
    hp.numSampTrainAE = 500
    hp.numSampTestAE = 1

    # logging
    hp.logIntervalAE = 50
    hp.checkpointIntervalAE = 500


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'../../src/Heat/data'
    ep.code = f'../../src/Heat'
    ep.run = f'{runName}'



if __name__ == '__main__':

    args = Parser().parse()
    logger = logging.getLogger('my_module')
    logger.setLevel(logging.DEBUG)

    # set useful paths to instance `experPaths`
    runName ='firstTry'
    experPaths = Paths(experDir, args.os)
    addPaths(experPaths, runName)

    # set hyper params for run
    class HyperParams: pass
    hp = HyperParams()
    setHyperParams(hp)
    
    # load saved hyper params for testing from old runs
    # hp = loadRunArgs(experPaths.run)
    
    startSavingLogs(args, experPaths.run, logger)
    rawData = LoadData(hp, experPaths, args)

    aePipeline = AEPipeline(AutoEncoder, hp, experPaths, rawData, AEDatasetClass, args)
    # aePipeline.train()
    # aePipeline.test()
    # aePipeline.generateLatentVecs()

    rawData.loadLatentVecs()
    modelPipeline = ModelPipeline(Model, hp, experPaths, rawData, DatasetClass, args)

    # train and test
    modelPipeline.train()

    hp.predData_Info = f'_'
    # modelPipeline.test()
    # aePipeline.decodeLatentVecs()

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)    

    