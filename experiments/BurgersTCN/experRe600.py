

#%%

import pdb
import logging
import torch as T

import sys
from os.path import dirname, realpath, join

filePath = realpath(__file__)
experDir = dirname(realpath(__file__))
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

#%%'
from src.Utils import Arguments, startSavingLogs, save_args
from src.Pipeline import ModelPipeline
from src.AEPipeline import AEPipeline
from src.Paths import Paths

from src.Burgers.BurgersTCNDataset import DatasetClass
from src.Burgers.BurgersTCNAEDataset import AEDatasetClass
from src.Burgers.BurgersTCNModel import Model
from src.Burgers.BurgersTCNAEModel import AutoEncoder
from src.Burgers.BurgersTCNLoadData import LoadData
from src.Burgers.BurgersTCNPlots import Plots


#%%
def setHyperParams(hp):
#     # model 
#     hp.latentDim = 10
#     hp.seq_len = 20
#     hp.num_inputs = hp.latentDim
#     hp.num_channels = [36, 36, 36]
#     hp.output_size = hp.latentDim
#     hp.kernel_size = 3
#     hp.dropout = 0.1

#     # training
#     hp.numIters = 5001
#     hp.lr = 0.0003
#     hp.batchSizeTrain = 25
    
#     hp.epochStartTrain = 0000

#     # testing
#     hp.loadWeightsEpoch = 3000
#     hp.batchSizeTest = 1
#     hp.timeStepsUnroll = 220

#     # data
#     hp.numSampTrain = 150
#     hp.numSampTest = 1
#     hp.Re = 600

#     # logging
#     hp.save = 1
#     hp.show = 0
#     hp.saveLogs = 1
#     hp.saveInterval = 20
#     hp.logInterval = 100
#     hp.checkpointInterval = 500

#     # AEtraining
#     hp.numItersAE = 3001
#     hp.lrAE = 0.00004
#     hp.batchSizeTrainAE = 50
#     hp.epochStartTrainAE = 0

#     # AEtesting
#     hp.loadAEWeightsEpoch = 3000
#     hp.batchSizeTestAE = 1
#     hp.batchSizeEncode = 500

#     # AEdata
#     hp.numSampTrainAE = 250
#     hp.numSampTestAE = 1

#     # logging
#     hp.logIntervalAE = 50
#     hp.checkpointIntervalAE = 1000


    # model 
    hp.latentDim = 10
    hp.seq_len = 20
    hp.num_inputs = hp.latentDim
    hp.num_channels = [36, 36, 36]
    hp.output_size = hp.latentDim
    hp.kernel_size = 3
    hp.dropout = 0.1

    # training
    hp.numIters = 5001
    hp.lr = 0.0003
    hp.batchSizeTrain = 25
    
    hp.epochStartTrain = 0000

    # testing
    hp.loadWeightsEpoch = 4000
    hp.batchSizeTest = 1
    hp.timeStepsUnroll = 470

    # data
    hp.numSampTrain = 200
    hp.numSampTest = 1
    hp.Re = 600

    # logging
    hp.save = 1
    hp.show = 0
    hp.saveLogs = 1
    hp.saveInterval = 20
    hp.logInterval = 100
    hp.checkpointInterval = 500

    # AEtraining
    hp.numItersAE = 4001
    hp.lrAE = 0.00004
    hp.batchSizeTrainAE = 100
    hp.epochStartTrainAE = 0

    # AEtesting
    hp.loadAEWeightsEpoch = 4000
    hp.batchSizeTestAE = 1
    hp.batchSizeEncode = 500

    # AEdata
    hp.numSampTrainAE = 500
    hp.numSampTestAE = 1

    # logging
    hp.logIntervalAE = 50
    hp.checkpointIntervalAE = 1000


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/BurgersTCN'
    ep.run = f'{runName}'


#%%
args = Arguments()#.parse()
logger = logging.getLogger('my_module')
logger.setLevel(logging.DEBUG)

# set hyper params for run
class HyperParams: pass
hp = HyperParams()
setHyperParams(hp)

# set useful paths to instance `experPaths`
runName = f'resultsRe{hp.Re}'
experPaths = Paths(experDir, args.os)
addPaths(experPaths, runName)

# load saved hyper params for testing from old runs
# hp = loadRunArgs(experPaths.run)

startSavingLogs(args, experPaths.run, logger)
rawData = LoadData(hp, experPaths, args)


#%%

Plots().Simulation(rawData.data, 0, join(experPaths.run, 'sim'))


#%%
aePipeline = AEPipeline(AutoEncoder, hp, experPaths, rawData, AEDatasetClass, args)
aePipeline.train()
# aePipeline.test()
aePipeline.generateLatentVecs()

rawData.loadLatentVecs()
modelPipeline = ModelPipeline(Model, hp, experPaths, rawData, DatasetClass, args)

# train and test
modelPipeline.train()

hp.predData_Info = f'_'
modelPipeline.test()
aePipeline.decodeLatentVecs()

# save hyper params for the run
sv_args = hp
save_args(sv_args, experPaths.run)
# %%