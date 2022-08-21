

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
from src.BnnPipeline import ModelPipeline
from src.Paths import Paths

from src.Stoker.StokerDataset import DatasetClass
from src.Stoker.StokerCNNModel import Model
from src.Stoker.StokerLoadData import LoadData
from src.Stoker.StokerPlots import Plots

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
T.manual_seed(SEED)
T.backends.cudnn.deterministic = True

class ParamsManager:

    def __init__(self, ep):
        
        # model 
        self.seq_len = [20, 10]
        # self.num_inputs = self.latentDim
        self.num_channels = [[100, 100, 100], [200, 200, 200], [50, 50, 50]]
        # self.output_size = [50]
        self.kernel_size = [9, 5, 11]
        self.dropout = [0]
        

        # training
        self.numIters = [3001]
        self.lr = [1e-3]
        self.batchSizeTrain = [10]
        self.epochStartTrain = [0000]

        # testing
        self.loadWeightsEpoch = [3000]
        self.batchSizeTest = [1]
        self.timeStepsUnroll = [450]
        self.nPredSampsValid = [30]
        self.nPredSampsTest = [30]

        # data
        self.numSampData = [500]
        self.numSampTrain = [250]
        self.numSampValid = [100]
        self.numSampTest = [1]
        self.reduce = [False]

        # logging
        self.save = [1]
        self.show = [0]
        self.saveLogs = [1]
        self.saveInterval = [20]
        self.logInterval = [50]
        self.checkpointInterval = [200]
        
        params = self.__dict__
        with open(join(ep.experDir, "AllParams.json"), 'w') as file:
            json.dump(params, file, indent=4)


    def iterateComb(self, ep):

        ls = []
        path = join(ep.experDir, "paramCombDone.json")
        iii = 0
         

        # iterate over combinations of hyperparams
        for instance in itertools.product(*self.__dict__.values()):
            # if iii == 0: 
            #     iii=9
            #     continue
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
    hp.lr = 0.001
    hp.batchSizeTrain = 25
    
    hp.epochStartTrain = 0000

    # testing
    hp.test_samples = 30
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
    sql = hpDict['seq_len']
    lr = hpDict['lr']
    trs = hpDict['numSampTrain']
    bs = hpDict['batchSizeTrain']
    ch = hpDict['num_channels']
    krs = hpDict['kernel_size']
    chn = ""
    for c in ch:
        chn+=str(c)

    rnd = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    runName = f'results_lstmSwap_sql{sql}_krs{krs}_lr{lr}_trSmp{trs}_ch{chn}_bs{bs}_{rnd}'
    hpDict["runName"] = runName


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/StokerTCN'
    ep.run = f'{runName}'


def automation(hp, experPaths):
    # set hyper params for run
    hp.padding = (hp.kernel_size - 1)//2

    # set useful paths to instance `experPaths`
    addPaths(experPaths, hp.runName)

    startSavingLogs(args, experPaths.run, logger)
    rawData = LoadData(hp, experPaths, args)

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

hp = HyperParams()
hpDict = hp.__dict__
addName(hpDict)
# automation(Dict2Class(hpDict), experPaths)




