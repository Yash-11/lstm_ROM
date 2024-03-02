"""
Loads data
"""


import json
import numpy as np
import torch as T
from scipy.io import loadmat

import sys
from os.path import dirname, realpath, join, exists

filePath = realpath(__file__)
projectDir = dirname(dirname(dirname(filePath)))
sys.path.append(projectDir)

from src.Utils import Dict2Class


class LoadData:
    """ Loads data present in /data folder and saves in self.data. 
    Var: 
        loadRun: runs to be loaded
    """

    def __init__(self, hp, experPaths, args):

        self.hp = hp
        self.info = args.info if hasattr(args, 'info') else print
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
            self.data (Tensor): (timeSteps, H, W)
                timeStep: num steps
        """

        data = np.load(join(self.dataDir, f'river_data1.npy'))
        self.data = T.tensor(data, dtype=T.float32)
        self.data = T.permute(self.data, (2, 0, 1))
        self.hp.imH = self.data.shape[1]
        self.hp.imW = self.data.shape[2]
        self.hp.maxNumTimeSteps = self.data.shape[0]
        
        self.hp.dataMin = self.data.min().item()
        self.hp.dataMax = self.data.max().item()


    def loadLatentVecs(self):
        path = join(self.experPaths.run, 'LatentVecs.npy')
        self.LatentVecs = T.tensor(np.load(path), dtype=T.float32) 

        self.info(f'Latent Vectors loaded \nshape: {self.LatentVecs.shape}\n')

