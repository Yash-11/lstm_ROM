

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

from src.Burgers.BurgersTCNDataset import DatasetClass
from src.Burgers.BurgersTCNAEDataset import AEDatasetClass
from src.Burgers.BurgersLSTMModel import Model
from src.Burgers.BurgersAEModel import AutoEncoder
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
        self.latentDim = [6, 8, 10, 16, 32]
        self.seq_len = [5, 10, 20]
        self.num_lstm_layers = [1, 2, 3]

        # training
        self.numIters = [5001]
        self.lr = [3e-4]
        self.batchSizeTrain = [10]
        
        self.epochStartTrain = [0000]

        # testing
        self.loadWeightsEpoch = [2000]
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
        self.logInterval = [100]
        self.checkpointInterval = [100]

        # AEtraining
        self.numItersAE = [101]
        self.lrAE = [0.00004]
        self.batchSizeTrainAE = [50]
        self.epochStartTrainAE = [0]

        # AEtesting
        self.loadAEWeightsEpoch = [100]
        self.batchSizeTestAE = [1]
        self.batchSizeEncode = [250]

        # AEdata
        self.numSampTrainAE = [200]
        self.numSampTestAE = [1]
        self.numSampValidAE = [50]

        # logging
        self.logIntervalAE = [100]
        self.checkpointIntervalAE = [100]
        
        params = self.__dict__
        with open(join(ep.experDir, "AllParams.json"), 'w') as file:
            json.dump(params, file, indent=4)


    def run(self, ep):

        ls = []

        # load paramCombDone
        path = join(ep.experDir, "paramCombDone.json")
        # if not exists(path):
        #     with open(path, 'w') as file: json.dump(ls, file, indent=4)
         

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
        ld = hpDict['latentDim']
        sql = hpDict['seq_len']
        lr = hpDict['lr']
        trs = hpDict['numSampTrain']
        bs = hpDict['batchSizeTrain']
        num_lstm_layers = hpDict['num_lstm_layers']

        rnd = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        runName = f'Re{Re}ld{ld}_sql{sql}_layers{num_lstm_layers}_lr{lr}_trSmp{trs}_bs{bs}{rnd}'
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


    #%%
    # Plots().Simulation(rawData.data, 0, join(experPaths.run, 'sim'))

    #%%
    aePipeline = AEPipeline(AutoEncoder, hp, experPaths, rawData, AEDatasetClass, args)
    aePipeline.train()
    # aePipeline.test()
    LatentVecs = aePipeline.generateLatentVecs()  # (numSampTrainAE, latentDim)

    rawData.loadLatentVecs()
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



