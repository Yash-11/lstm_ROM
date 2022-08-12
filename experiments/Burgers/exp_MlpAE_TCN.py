

# %% ---------------------------------------------------------------------------

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

# %% ---------------------------------------------------------------------------
#                          Import useful modules

from src.Utils import Arguments, Dict2Class, startSavingLogs, save_args
from src.Pipeline import ModelPipeline
from src.AEPipeline import AEPipeline
from src.Paths import Paths

from src.Burgers.BurgersLoadData import LoadData
from src.Burgers.BurgersPlots import Plots

from src.Burgers.BurgersAEDataset import AEDatasetClass
from src.Burgers.BurgersAEModel import AutoEncoder

from src.Burgers.BurgersDataset import DatasetClass
from src.Burgers.BurgersTCNModel import Model

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
T.manual_seed(SEED)
T.backends.cudnn.deterministic = True

class ParamsManager:

    def __init__(self, ep):
        
        # model 
        self.latentDim = [4, 8, 16, 32]
        self.seq_len = [10, 20]
        self.num_channels = [[32, 32], [16, 16], [25, 25, 25]]
        self.kernel_size = [3]
        self.dropout = [0]
        self.AE_Model = [1]

        # training
        self.numIters = [4001]
        self.lr = [3e-4]
        self.batchSizeTrain = [16]
        self.epochStartTrain = [0000]
        self.weight_decay = [1e-5]

        # testing
        self.loadWeightsEpoch = [0]
        self.batchSizeTest = [1]
        self.timeStepsUnroll = [230]

        # data
        self.numSampData = [250]
        self.numSampTrain = [150]
        self.numSampValid = [50]
        self.numSampTest = [1]
        self.reduce = [True]
        self.Re = [300, 600]

        # logging
        self.save = [1]
        self.show = [0]
        self.saveLogs = [1]
        self.saveInterval = [20]
        self.logInterval = [50]
        self.checkpointInterval = [50]

        # AEtraining
        self.numItersAE = [3001]
        self.lrAE = [0.0003]
        self.batchSizeTrainAE = [50]
        self.epochStartTrainAE = [0]

        # AEtesting
        self.loadAEWeightsEpoch = [3000]
        self.batchSizeTestAE = [1]
        self.batchSizeEncode = [250]

        # AEdata
        self.numSampTrainAE = [200]
        self.numSampTestAE = [50]
        self.numSampValidAE = [50]

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
    hp.seq_len = 13
    hp.num_channels = [12, 12]
    hp.output_size = 6
    hp.kernel_size = 3
    hp.latentDim = 32
    hp.dropout = 0
    hp.AE_Model = 4
    
    # training
    hp.numIters = 4001
    hp.lr = 5e-4
    hp.batchSizeTrain = 25
    hp.epochStartTrain = 0000

    # testing
    hp.loadWeightsEpoch = 2500
    hp.batchSizeTest = 1
    hp.timeStepsUnroll = 450

    # data
    hp.numSampData = 500
    hp.numSampTrain = 250
    hp.numSampValid = 100
    hp.numSampTest = 1
    hp.reduce = True
    hp.Re = 300

    # logging
    hp.save = 1
    hp.show = 0
    hp.saveLogs = 1
    hp.saveInterval = 20
    hp.logInterval = 10
    hp.checkpointInterval = 100

    # AEtraining
    hp.numItersAE = 2001
    hp.lrAE = 0.0004
    hp.weight_decay = 1e-4
    hp.batchSizeTrainAE = 50
    hp.epochStartTrainAE = 0

    # AEtesting
    hp.loadAEWeightsEpoch = 2000
    hp.batchSizeTestAE = 1
    hp.batchSizeEncode = 500

    # AEdata
    hp.numSampTrainAE = 400
    hp.numSampTestAE = 1
    hp.numSampValidAE = 100

    # logging
    hp.logIntervalAE = 100
    hp.checkpointIntervalAE = 1000
    return hp


def addName(hpDict):
    """
    Give name to particular hyperparam combination.
    """
    Re = hpDict['Re']
    ld = hpDict['latentDim']
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
    runName = f'results_MlpAE_TCN_Re{Re}_ld{ld}_sql{sql}_krs{krs}_lr{lr}_trSmp{trs}_ch{chn}_bs{bs}_{rnd}'
    hpDict["runName"] = runName


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/BurgersTCN'
    ep.run = f'{runName}'


def resultsAE(hp):
    
    try:
        info = hp.predData_Info if hasattr(hp, 'predData_Info') else ''
        name = f'predLatentDataTest_epoch{hp.loadAEWeightsEpoch}{info}.hdf5'
        predData = h5py.File(join(experPaths.run, name), 'r')
        print(f'loaded pred data {name}')
    except:
        print(f'{join(experPaths.run, name)}')
        raise Exception(FileNotFoundError)

    pred = predData['pred']
    target = predData['target']

    for i in [0, 15, 30]:
        savePath = join(experPaths.run, f'BurgersAEpredsinglesnapPlot{i}_epoch{hp.loadWeightsEpoch}')
        plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
        plotData = {'pred': pred, 'target': target}
        Plots().plotPredSingleAE(plotData, Dict2Class(plotParams), savePath, i)



def automation(hp, experPaths):

    # set useful paths to instance `experPaths`
    addPaths(experPaths, hp.runName)

    startSavingLogs(args, experPaths.run, logger)
    rawData = LoadData(hp, experPaths, args)

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)
    
    if hp.reduce:
        aePipeline = AEPipeline(AutoEncoder, hp, experPaths, rawData, AEDatasetClass, args)
        aePipeline.train()
        aePipeline.test()
        resultsAE(hp)
        LatentVecs = aePipeline.generateLatentVecs()  # (numSampTrainAE, latentDim)
        rawData.loadLatentVecs()
    
    # train
    modelPipeline = ModelPipeline(Model, hp, experPaths, rawData, DatasetClass, args)
    modelPipeline.train()


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




