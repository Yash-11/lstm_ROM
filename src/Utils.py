
import argparse
import json
import torch as T
import logging
from prettytable import PrettyTable
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal

import os
import sys
from os.path import dirname, realpath, join


class Dict2Class(object):
	
	def __init__(self, my_dict):
		for key in my_dict:
			setattr(self, key, my_dict[key])


def count_parameters(model, logger):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        # logger.info(f"name: {name} value{parameter}\n")
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


def save_args(sv_args, sv_dir):
    del_args = ['logger', 'device']
    [delattr(sv_args, g) for g in del_args if hasattr(sv_args, g)]
    with open(join(sv_dir, "args.json"), 'w') as args_file:
        json.dump(vars(sv_args), args_file, indent=4)


def loadRunArgs(saveDir):
    path = join(saveDir, "args.json")
    with open(path, 'r') as args_file:
        dir = json.load(args_file)
    return Dict2Class(dir)


def startSavingLogs(args, logsPath, logger):
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if args.save_logs:
        path = join(logsPath, 'errorLogs.log')

        fh = logging.FileHandler(path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    logger.info(f'\n*********************************************************************************************\n')

    # args.logger = logger
    args.info = logger.info if args.verbose == 1 else logger.debug


class Arguments:

    def __init__(self):
        
        self.prjct_dir = dirname(dirname(__file__))

        print('project directory: {}'.format(self.prjct_dir))
        if sys.platform == "linux" or sys.platform == "linux2": operating_sys = 'Linux'
        elif sys.platform == "win32": operating_sys = 'Windows'
        else: raise Exception('os not supported')

        # self.device = T.device('cuda:' + str(0) if T.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.os = operating_sys
        self.seed = 0
        self.save_logs = True
        self.verbose = 1


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='ensers')
        prjct_dir = dirname(dirname(__file__))

        print('project directory: {}'.format(prjct_dir))
        if sys.platform == "linux" or sys.platform == "linux2": operating_sys = 'Linux'
        elif sys.platform == "win32": operating_sys = 'Windows'
        else: raise Exception('os not supported')
        device = T.device('cuda:' + str(0) if T.cuda.is_available() else 'cpu')

        self.add_argument('--prjct_dir', type=str, default=prjct_dir)
        self.add_argument('--device', type=str, default=device)
        self.add_argument('--os', type=str, default=operating_sys)
        self.add_argument('--seed', type=str, default=0)
        self.add_argument('--save_logs', action='store_true', default=True)

    def parse(self):
        args = self.parse_args()
        return args
