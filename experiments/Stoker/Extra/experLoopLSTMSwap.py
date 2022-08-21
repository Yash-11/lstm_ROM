

# %% ---------------------------------------------------------------------------


import logging
import torch as T
import itertools
import json
import random
import string
import h5py
import numpy as np

import sys
from os.path import dirname, realpath, join, exists

filePath = realpath(__file__)
experDir = dirname(realpath(__file__))
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

# %% ---------------------------------------------------------------------------
#                          Import useful modules

from src.Utils import Arguments, Dict2Class, startSavingLogs, save_args
from src.Pipeline import ModelPipeline
from src.AEPipeline import AEPipeline
from src.Paths import Paths

from src.Stoker.StokerDataset import DatasetClass
from src.Stoker.StokerLSTMModelSwap import Model
from src.Stoker.StokerLoadData import LoadData
from src.Stoker.StokerPlots import Plots

from src.Stoker.StokerAEDataset import AEDatasetClass
from src.Stoker.StokerAEModel import AutoEncoder

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
T.manual_seed(SEED)
T.backends.cudnn.deterministic = True

class ParamsManager:

    def __init__(self, ep):
        
        # model 
        self.seq_len = [20, 10, 5]
        self.latentDim = [6, 10, 16, 32]
        self.num_lstm_layers = [1, 2, 3]        

        # training
        self.numIters = [3001]
        self.lr = [3e-4]
        self.batchSizeTrain = [10]
        self.epochStartTrain = [0000]
        self.weight_decay = [1e-5]

        # testing
        self.loadWeightsEpoch = [500]
        self.batchSizeTest = [1]
        self.timeStepsUnroll = [450]

        # data
        self.numSampData = [500]
        self.numSampTrain = [250]
        self.numSampValid = [100]
        self.numSampTest = [1]
        self.reduce = [True]

        # logging
        self.save = [1]
        self.show = [0]
        self.saveLogs = [1]
        self.saveInterval = [20]
        self.logInterval = [50]
        self.checkpointInterval = [100]

        # AEtraining
        self.numItersAE = [2001]
        self.lrAE = [0.0003]
        self.batchSizeTrainAE = [50]
        self.epochStartTrainAE = [0]

        # AEtesting
        self.loadAEWeightsEpoch = [2000]
        self.batchSizeTestAE = [1]
        self.batchSizeEncode = [500]

        # AEdata
        self.numSampTrainAE = [400]
        self.numSampTestAE = [1]
        self.numSampValidAE = [100]

        # logging
        self.logIntervalAE = [100]
        self.checkpointIntervalAE = [1000]
        
        params = self.__dict__
        with open(join(ep.experDir, "AllParams.json"), 'w') as file:
            json.dump(params, file, indent=4)


    def iterateComb(self, ep):

        ls = []
        path = join(ep.experDir, "paramCombDone.json")


        # iterate over combinations of hyperparams
        for instance in itertools.product(*self.__dict__.values()):                
            hpDict = dict(zip(self.__dict__.keys(), instance))

            # load paramCombDone.json
            if not exists(path):
                ls = []
            else:
                with open(path, 'r') as args_file:
                    ls = json.load(args_file)

            # train combination if not trained before
            if not hpDict in ls:
                addName(hpDict)
                
                automation(Dict2Class(hpDict), ep)

                # update paramCombDone.json
                ls.append(hpDict)
                with open(path, 'w') as file:
                    json.dump(ls, file, indent=4)

def HyperParams():

    class HyperParams: pass
    hp = HyperParams()

    # model 
    hp.latentDim = 100
    hp.seq_len = 20
    hp.num_inputs = hp.latentDim
    hp.num_channels = [64, 64, 1]
    hp.output_size = hp.latentDim
    hp.kernel_size = 3
    hp.dropout = 0.2

    # training
    hp.numIters = 4001
    hp.lr = 0.0003
    hp.batchSizeTrain = 25
    
    hp.epochStartTrain = 0000

    # testing
    hp.loadWeightsEpoch = 200
    hp.batchSizeTest = 1
    hp.timeStepsUnroll = 230

    # data
    hp.numSampTrain = 280
    hp.numSampValid = 100
    hp.numSampTest = 1
    hp.numSampData = 500
    hp.reduce = False

    # logging
    hp.save = 1
    hp.show = 0
    hp.saveLogs = 1
    hp.saveInterval = 20
    hp.logInterval = 25
    hp.checkpointInterval = 50


def addName(hpDict):
    """
    Give name to particular hyperparam combination.
    """
    ld = hpDict['latentDim']
    sql = hpDict['seq_len']
    lr = hpDict['lr']
    trs = hpDict['numSampTrain']
    bs = hpDict['batchSizeTrain']
    num_lstm_layers = hpDict['num_lstm_layers']

    rnd = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    runName = f'results_ld{ld}_sql{sql}_layers{num_lstm_layers}_lr{lr}_trSmp{trs}_bs{bs}{rnd}'
    hpDict["runName"] = runName


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/StokerTCN'
    ep.run = f'{runName}'


def automation(hp, experPaths):

    # set useful paths to instance `experPaths`
    addPaths(experPaths, hp.runName)

    startSavingLogs(args, experPaths.run, logger)
    rawData = LoadData(hp, experPaths, args)

    if hp.reduce:
        aePipeline = AEPipeline(AutoEncoder, hp, experPaths, rawData, AEDatasetClass, args)
        aePipeline.train()
        # aePipeline.test()
        LatentVecs = aePipeline.generateLatentVecs()  # (numSampTrainAE, latentDim)
        rawData.loadLatentVecs()
    
    # train
    modelPipeline = ModelPipeline(Model, hp, experPaths, rawData, DatasetClass, args)
    modelPipeline.train()

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)


# %% ---------------------------------------------------------------------------
args = Arguments()
logger = logging.getLogger('my_module')
logger.setLevel(logging.DEBUG)
experPaths = Paths(experDir, args.os)

# %% ---------------------------------------------------------------------------
#                   Train all combinations of hyperParams
manager = ParamsManager(experPaths)
manager.iterateComb(experPaths)

# %% ---------------------------------------------------------------------------
#                      Train particular hyperParam comb

# hp = HyperParams()
# hpDict = hp.__dict__
# addName(hpDict)
# automation(Dict2Class(hpDict), experPaths)




