

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

import sys
from os.path import dirname, realpath, join, exists

filePath = realpath(__file__)
experDir = dirname(realpath(__file__))
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

#%%'
from src.Utils import Arguments, Dict2Class, startSavingLogs, save_args
from src.Pipeline import ModelPipeline
from src.AEPipeline import AEPipeline
from src.Paths import Paths

from src.Burgers.BurgersTCNDatasetSwap import DatasetClass
from src.Burgers.BurgersTCNModelSwap import Model
from src.Burgers.BurgersLoadData import LoadData
from src.Burgers.BurgersPlots import Plots

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
        self.num_channels = [[200, 200], [100, 100], [50, 50]]
        # self.output_size = [50]
        self.kernel_size = [4, 3]
        self.dropout = [0]

        # training
        self.numIters = [601]
        self.lr = [3e-5]
        self.batchSizeTrain = [16]
        
        self.epochStartTrain = [0000]

        # testing
        self.loadWeightsEpoch = [500]
        self.batchSizeTest = [1]
        self.timeStepsUnroll = [230]

        # data
        self.numSampData = [250]
        self.numSampTrain = [150]
        self.numSampValid = [50]
        self.numSampTest = [1]
        self.Re = [300, 600]

        # logging
        self.save = [1]
        self.show = [0]
        self.saveLogs = [1]
        self.saveInterval = [20]
        self.logInterval = [20]
        self.checkpointInterval = [50]
        
        params = self.__dict__
        with open(join(ep.experDir, "AllParams.json"), 'w') as file:
            json.dump(params, file, indent=4)


    def run(self, ep):

        ls = []

        # load paramCombDone
        path = join(ep.experDir, "paramCombDone.json")
         

        for instance in itertools.product(*self.__dict__.values()):
            hpDict = dict(zip(self.__dict__.keys(), instance))

            if not exists(path):
                ls = []
            else:
                with open(path, 'r') as args_file:
                    ls = json.load(args_file)

            if not hpDict in ls:
                self.addName(hpDict)
                automation(Dict2Class(hpDict))

                ls.append(hpDict)
                with open(path, 'w') as file:
                    json.dump(ls, file, indent=4)

    def addName(self, hpDict):
        Re = hpDict['Re']
        sql = hpDict['seq_len']
        lr = hpDict['lr']
        trs = hpDict['numSampTrain']
        bs = hpDict['batchSizeTrain']
        ch = hpDict['num_channels']
        chn = ""
        for c in ch:
            chn+=str(c)

        rnd = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        runName = f'Re{Re}_sql{sql}_lr{lr}_trSmp{trs}_ch{chn}__bs{bs}_{rnd}'
        hpDict["runName"] = runName


#%%
def setHyperParams(hp, ParamsDict):
    for key, value in ParamsDict.items():
        setattr(hp, key, value)

def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/BurgersTCN'
    ep.run = f'{runName}'


def automation(hp):
    # set hyper params for run
    # setHyperParams(hp)

    # set useful paths to instance `experPaths`
    experPaths = Paths(experDir, args.os)
    addPaths(experPaths, hp.runName)

    startSavingLogs(args, experPaths.run, logger)
    rawData = LoadData(hp, experPaths, args)

    modelPipeline = ModelPipeline(Model, hp, experPaths, rawData, DatasetClass, args)

    # train and test 
    modelPipeline.train()

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)
    # %%

#%%
args = Arguments()
logger = logging.getLogger('my_module')
logger.setLevel(logging.DEBUG)

experPaths = Paths(experDir, args.os)
manager = ParamsManager(experPaths)
manager.run(experPaths)



