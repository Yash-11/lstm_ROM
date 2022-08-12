

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
from src.Utils import Arguments, startSavingLogs, save_args, loadRunArgs, Dict2Class
from src.Pipeline import ModelPipeline
from src.AEPipeline import AEPipeline
from src.Paths import Paths

from src.Burgers.BurgersTCNDataset import DatasetClass
from src.Burgers.BurgersTCNAEDataset import AEDatasetClass
from src.Burgers.BurgersTCNModel import Model
from src.Burgers.BurgersAEModel import AutoEncoder
from src.Burgers.BurgersLoadData import LoadData
from src.Burgers.BurgersPlots import Plots


#%%
def setHyperParams(hp):
    # model 
    hp.latentDim = 6
    hp.seq_len = 13
    hp.num_inputs = hp.latentDim
    hp.num_channels = [32, 32]#[32, 32, 32]
    hp.output_size = hp.latentDim
    hp.kernel_size = 3
    hp.dropout = 0.3

    # training
    hp.numIters = 5001
    hp.lr = 0.0003#0.0002
    hp.batchSizeTrain = 16
    
    hp.epochStartTrain = 000

    # testing
    hp.loadWeightsEpoch = 800
    hp.batchSizeTest = 1
    hp.timeStepsUnroll = 235

    # data
    hp.numSampData = 250
    hp.numSampTrain = 150
    hp.numSampValid = 50
    hp.numSampTest = 1
    hp.Re = 600

    # logging
    hp.save = 1
    hp.show = 0
    hp.saveLogs = 1
    hp.saveInterval = 20
    hp.logInterval = 100
    hp.checkpointInterval = 100

    # AEtraining
    hp.numItersAE = 3001
    hp.lrAE = 1e-03
    hp.batchSizeTrainAE = 250
    hp.epochStartTrainAE = 0

    # AEtesting
    hp.loadAEWeightsEpoch = 3000
    hp.batchSizeTestAE = 1
    hp.batchSizeEncode = 250

    # AEdata
    hp.numSampTrainAE = 250
    hp.numSampTestAE = 5

    # logging
    hp.logIntervalAE = 200
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
runName = f'resultsRe{hp.Re}Best'
experPaths = Paths(experDir, args.os)
addPaths(experPaths, runName)

# load saved hyper params for testing from old runs
# hp = loadRunArgs(experPaths.run)
# hp.numSampData = 500
# hp.lrAE =6e-04
# hp.batchSizeTrainAE = 250
# hp.numItersAE = 2001
# hp.loadAEWeightsEpoch = 2000
# hp.loadWeightsEpoch = 800

# hp.num_channels = [5, 5]
# hp.batchSizeTrain = 25
# hp.seq_len = 3
# hp.lr = 0.001
# hp.loadWeightsEpoch = 1000

startSavingLogs(args, experPaths.run, logger)
rawData = LoadData(hp, experPaths, args)


#%%

# Plots().Simulation(rawData.data, 0, join(experPaths.run, 'sim'))


#%%
aePipeline = AEPipeline(AutoEncoder, hp, experPaths, rawData, AEDatasetClass, args)
# aePipeline.train()
# aePipeline.test()
# LatentVecs = aePipeline.generateLatentVecs()  # (numSampTrainAE, latentDim)

# # plot latent vector with time
# savePath = join(experPaths.run, f'LatentVecs{hp.loadWeightsEpoch}')
# plotParams = {'savePath':savePath}
# Plots().latentPlot(LatentVecs, Dict2Class(plotParams))

rawData.loadLatentVecs()
modelPipeline = ModelPipeline(Model, hp, experPaths, rawData, DatasetClass, args)

# train and test 
# modelPipeline.train()

hp.predData_Info = f'_'
LatentVecsPred = modelPipeline.test()
aePipeline.decodeLatentVecs()

# plot latent vector with time
savePath = join(experPaths.run, f'LatentVecsPred{hp.loadWeightsEpoch}')
plotParams = {'savePath':savePath}
Plots().latentPlot(LatentVecsPred, Dict2Class(plotParams))

# save hyper params for the run
sv_args = hp
save_args(sv_args, experPaths.run)
# %%
