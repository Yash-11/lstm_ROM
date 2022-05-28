"""" 
Provide Paths to directory
for weights, data etc.
"""

import pdb
import os.path as osp
import os
from os.path import dirname, realpath


class Paths(object):
    """ 
    Store paths to folder of attributes w.r.t experDir.   
    Paths are store in :obj:`pathDict`
           
    Returns: path of attributes after making its dir    
    """

    def __init__(self, experDir, OS, pathDict={}):
        super().__setattr__('pathDict', dict())
        super().__setattr__('experDir', experDir)
        super().__setattr__('OS', OS)
        for key in pathDict:
            setattr(self, key, pathDict[key])
        

    def __getattr__(self, name: str):
        
        if name in self.pathDict:
            path = self.pathDict[name]
            if not os.path.exists(path):
                os.makedirs(path)
                # print(f'path made {path}')
            return path
        else:
            raise AttributeError(f'{self.__class__.__name__}.{name} is invalid.')


    def __setattr__(self, name: str, value) -> None:
        value = self.fullPath(value)
        self.pathDict[name] = value
    

    def fullPath(self, relativePath):
        fullPath = osp.join(self.experDir, relativePath)
        return self.OSpath(fullPath, self.OS)


    def OSpath(self, fullPath, OS):
        if OS == 'Windows': return fullPath.replace('/', '\\')
        if OS == 'Linux': return fullPath 



if __name__ == '__main__':
    """ Path function Test """

    pathDict = {'torch_data': 'torch_data'}
    experDir = dirname(realpath(__file__))

    experPaths = Paths(experDir, 'Windows', pathDict=pathDict)    

    experPaths.weights = 'run/checkpoints'
    print(experPaths.weights)
    
    experPaths.heaven = 'earth'
    print(experPaths.heaven)

    print(experPaths.__dict__)
    print('\n\n\n')
    print(experPaths.hell)

