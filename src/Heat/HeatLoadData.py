"""
Loads data
"""

import pdb
import json
import numpy as np
import torch as T
from scipy.io import loadmat

import sys
from os.path import dirname, realpath, join, exists

filePath = realpath(__file__)
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

from src.Utils import Dict2Class, awgn


class LoadData:
    """ Loads data present in /data folder and saves in self.data. 
    Var: 
        loadRun: runs to be loaded
    Returns:
        self.data (Tensor): (timeStep, 2, numNodes)
    """

    def __init__(self, hp, experPaths, args):

        self.hp = hp
        self.info = args.logger.info if hasattr(args, 'logger') else print
        self.dataDir = experPaths.data
        self.experPaths = experPaths
        
        self.loadVertexValues()        
        self.info(f'data loaded \ndata shape: {self.data.shape}\n')

    
    def loadDataParams(self):
        path = join(self.dataDir, f'dataParams.json')
        with open(path, 'r') as file:  
            dict = json.load(file)
        return Dict2Class(dict)


    def loadVertexValues(self):
        """ 
        Vars:
            self.data (Tensor): (latentDim, timeSteps)
                timeStep: num steps
        """

        data = loadmat(join(self.dataDir, 'heatS500.mat'))
        self.data = T.tensor(data['solution'], dtype=T.float32)        
        self.hp.imDim = self.data.shape[0]
        self.hp.maxNumTimeSteps = self.data.shape[1]

    def loadLatentVecs(self):
        path = join(self.experPaths.run, 'LatentVecs.npy')
        self.LatentVecs = T.tensor(np.load(path), dtype=T.float32) 

        self.info(f'Latent Vectors loaded \nshape: {self.LatentVecs.shape}\n')


if __name__ == '__main__':
    
    from src.Utils import Parser
    from src.Paths import Paths

    # ---------------------------- Argument Parser -----------------------------
    args = Parser().parse()
    pathDict = {'data': 'data'}
    experDir = dirname(realpath(__file__))
    experPaths = Paths(experDir, args.os, pathDict=pathDict)

    # -------------------------- load and plot data ----------------------------
    rawData = LoadData(args, experPaths)