

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
from src.EnsemblePipeline import ModelPipeline
from src.AEPipelineEnsemble import AEPipeline
from src.Paths import Paths

from src.CD.CDLoadData import LoadData
from src.CD.CDPlots import Plots

from src.CD.CDAEDataset import AEDatasetClass
from src.CD.CDCAEModel import AutoEncoder

from src.CD.CDDataset import DatasetClass
from src.CD.CDCNNEnsembleModel import Model

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
T.manual_seed(SEED)
T.backends.cudnn.deterministic = True

class ParamsManager:

    def __init__(self, ep):
        
        # model 
        self.seq_len = [30]
        self.num_channels = [[250, 250, 250]]
        self.kernel_size = [3]
        self.latentDim = [4]
        self.dropout = [0]
        self.AE_Model = [9]
        self.n_modelEnsemble = [10]
        self.epsilonLatentVar = [1e-7]

        # training
        self.numIters = [1202]
        self.lr = [3e-4] # 5e-5
        self.batchSizeTrain = [10]
        self.epochStartTrain = [0000]
        self.weight_decay = [5e-5]

        # testing
        self.loadWeightsEpoch = [1200]
        self.batchSizeTest = [1]
        self.timeStepsUnroll = [270]

        # data
        self.numSampData = [300]
        self.numSampTrain = [180]
        self.numSampValid = [10]
        self.numSampTest = [1]
        self.reduce = [True]

        # logging
        self.save = [1]
        self.show = [0]
        self.saveLogs = [1]
        self.saveInterval = [20]
        self.logInterval = [10]
        self.checkpointInterval = [50]

        # AEtraining
        self.numItersAE = [11]
        self.lrAE = [3e-4]
        self.batchSizeTrainAE = [20]
        self.epochStartTrainAE = [0]

        # AEtesting
        self.loadAEWeightsEpoch = [10]
        self.batchSizeTestAE = [1]
        self.batchSizeEncode = [1]
        self.sampling = ['unscented_transform']

        # AEdata
        self.numSampTrainAE = [250]
        self.numSampTestAE = [50]
        self.numSampValidAE = [50]

        # logging
        self.logIntervalAE = [10]
        self.checkpointIntervalAE = [10]
        
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
    hp.seq_len = 20
    hp.num_channels = [100, 100, 100]
    hp.kernel_size = 3
    hp.latentDim = 125
    hp.dropout = 0
    hp.AE_Model = 7
    hp.n_modelEnsemble = 15
    hp.epsilonLatentVar = 1e-7

    # training
    hp.numIters = 8002
    hp.lr = 5e-6
    hp.batchSizeTrain = 16
    hp.epochStartTrain = 0000
    hp.weight_decay = 1e-5

    # testing
    hp.loadWeightsEpoch = 500
    hp.batchSizeTest = 1
    hp.timeStepsUnroll = 450

    # data
    hp.numSampData = 500
    hp.numSampTrain = 250
    hp.numSampValid = 100
    hp.numSampTest = 1
    hp.reduce = True

    # logging
    hp.save = 1
    hp.show = 0
    hp.saveLogs = 1
    hp.saveInterval = 20
    hp.logInterval = 50
    hp.checkpointInterval = 50

    # AEtraining
    hp.numItersAE = 4002
    hp.lrAE = 0.0003
    hp.batchSizeTrainAE = 50
    hp.epochStartTrainAE = 0

    # AEtesting
    hp.loadAEWeightsEpoch = 3000
    hp.batchSizeTestAE = 100
    hp.batchSizeEncode = 500
    hp.sampling = 'unscented_transform'

    # AEdata
    hp.numSampTrainAE = 400
    hp.numSampTestAE = 100
    hp.numSampValidAE = 100

    # logging
    hp.logIntervalAE = 100
    hp.checkpointIntervalAE = 100
    return hp


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
    AE_Model = hpDict['AE_Model']
    chn = ""
    for c in ch:
        chn+=str(c)

    rnd = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    runName = f'results_CAE_CNNEns_sql{sql}_krs{krs}_lr{lr}_trSmp{trs}_ch{chn}_bs{bs}_{rnd}'
    hpDict["runName"] = runName


def addPaths(ep, runName):
    ep.weights = f'{runName}/checkpoints'
    ep.data = f'./data'
    ep.code = f'../../src/CDTCN'
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

    for i in [0, 30, 20]:
        savePath = join(experPaths.run, f'CDAEpredsinglesnapPlot{i}_epoch{hp.loadWeightsEpoch}')
        plotParams = {'tStepModelPlot':[2]*hp.numSampTest}
        plotData = {'pred': pred, 'target': target}
        Plots().plotPredSingleAE(plotData, Dict2Class(plotParams), savePath, i)


def automation(hp, experPaths):
    # set hyper params for run
    hp.padding = (hp.kernel_size - 1)//2

    # set useful paths to instance `experPaths`
    addPaths(experPaths, hp.runName)

    startSavingLogs(args, experPaths.run, logger)
    args.info(f'{hp.runName}')
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

    # save hyper params for the run
    sv_args = hp
    save_args(sv_args, experPaths.run)
    
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




