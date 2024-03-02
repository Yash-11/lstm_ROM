"""
class for model training and testing 
"""

from genericpath import exists

from turtle import pd
from unicodedata import decimal
import h5py
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import tyxe

import pyro
import pyro.distributions as dist

import os
from os.path import dirname, realpath, join

from zmq import device


class ModelPipeline():

    def __init__(self, Model, hyperParams, experPaths, rawData, dataset, args):
        """
        ARGS:
            n_sensor (int): number of sensors
        """
        self.args = args
        self.info = args.info

        self.hp = hyperParams
        self.rawData = rawData
        self.dataset = dataset
        self.path = experPaths

        self.model = Model(hyperParams, args).to(args.device)
        self.loss = T.nn.MSELoss(reduction = 'sum')

        self.info(self)
        self.info(self.model)

    
    def saveModel(self, epoch, losses):
        PATH = join(self.path.weights, f'weights_epoch{epoch}.tar')
        state = {'epoch': epoch,
                 'model_state_dict': self.model.state_dict(),
                 'losses': losses
                 }
        T.save(state, PATH)
        self.info(f'model saved at epoch {epoch}')

        plt.figure()
        plt.plot(losses['train'], label='train')
        plt.plot(losses['valid'], label='valid')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Training Loss')
        plt.legend()
        # plt.yscale("log")
        plt.savefig(join(self.path.run, f'Loss_plot.png'))
        plt.close()
        plt.close('all')


    def loadModel(self, model, epoch, losses=None):
        """Loads pre-trained network from file"""
        try:
            PATH = join(self.path.weights, f'weights_epoch{epoch}.tar')
            checkpoint = T.load(PATH, map_location=T.device(self.args.device))
            checkpoint_epoch = checkpoint['epoch']
            self.info(f'Found model at epoch: {checkpoint_epoch}')
        except FileNotFoundError:
            if epoch>0: raise FileNotFoundError(f'Error: Could not find PyTorch network at epoch: {epoch}')
            return

        model.load_state_dict(checkpoint['model_state_dict'])
        if (not losses is None):
            losses = checkpoint['losses']
        return model, losses


    def __repr__(self):
        # numParameters = count_parameters(self.model, self.args.logger)
        description = f'\n\n\
            {self.hp}'
        return description


    def saveMinLoss(self, losses):
        minTrainLoss = min(losses['train'])
        dd = self.hp.checkpointInterval
        minValidLoss = min(losses['valid'][::dd])
        minTrainEpoch = losses['train'].index(minTrainLoss)        
        minValidEpoch = losses['valid'].index(minValidLoss)

        info = {'name': self.hp.runName, 'minValidLoss': minValidLoss, 'minValidEpoch': minValidEpoch, 
        'minTrainLoss': minTrainLoss, 'minTrainEpoch': minTrainEpoch}

        path = join(self.path.experDir, f'minLoss.csv')

        try:
            df =  pd.read_csv(path)
        except:
            df = pd.DataFrame(columns=['name', 'minValidLoss', 'minValidEpoch', 'minTrainLoss', 'minTrainEpoch'])
        
        df = df.append(info, ignore_index=True)
        # print(df.head)
        df = df.sort_values(by=['minValidLoss'])
        df.to_csv(path, index=False)


    def callback(self, b, i, avg_elbo):
        hp = self.hp

        x = self.dataValidX.to(self.args.device)
        y = self.dataValidY.to(self.args.device)
        
        if (i%400==0) and i>0:
            err, ll = b.evaluate(x, y, num_predictions=hp.nPredSampsValid)
            err = err.item()
            self.test(self.bnn, i)
        else:
            err = 0.0
        valid_loss = err / len(self.dataValidX)

        self.losses['train'].append(avg_elbo)
        self.losses['valid'].append(valid_loss)

        i = i+hp.epochStartTrain
        if i % hp.logInterval == 0: self.info(f"    ({i:02.0f}); ELBO={avg_elbo:.8f}; Valid_loss: {valid_loss:.8f}")

        # ------------------- Save model periodically ----------------------
        if (i % hp.checkpointInterval == 0) and (i > 0):
            self.saveModel(i, self.losses)
    
    
    def train(self):
        hp = self.hp
        device = self.args.device

        train_dataset = self.dataset(self.rawData, 'train', self.path, hp, device=self.args.device, info=self.info)
        train_loader = DataLoader(train_dataset, batch_size=hp.batchSizeTrain, shuffle=True)  
        
        self.dataValidX = train_dataset.dataValidX
        self.dataValidY = train_dataset.dataValidY

        likelihood = tyxe.likelihoods.HomoskedasticGaussian(len(train_loader.sampler), scale=0.1)
        prior = tyxe.priors.IIDPrior(dist.Normal(T.zeros(1, device=device), T.ones(1, device=device)))
        guide_factory = tyxe.guides.AutoNormal

        
        self.optimizer = pyro.optim.Adam({"lr": hp.lr})
        self.losses = {'train': [], 'valid': []}

        # ------------------- load model from checkpoint -----------------------
        if hp.epochStartTrain:
            self.losses = self.loadModel(hp.epochStartTrain, self.losses)

        self.bnn = tyxe.VariationalBNN(self.model, prior, likelihood, guide_factory)
        # with tyxe.poutine.local_reparameterization():
        nepochs = hp.numIters - hp.epochStartTrain+1
        self.bnn.fit(train_loader, self.optimizer, nepochs, callback=self.callback, device=device)

        self.saveMinLoss(self.losses)
        

    
    def savePredictions(self, predData, epoch, trainBool):
        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        predData_Path = join(self.path.run, f'predDataTest_epoch{epoch}{info}.hdf5')

        predArray, targetArray = predData

        with h5py.File(predData_Path, 'w') as f:
            f.create_dataset('pred', data=predArray.detach().cpu().numpy())
            f.create_dataset('target', data=targetArray.detach().cpu().numpy())
        print(f'pred data saved at {predData_Path}')
    

    def rescale(self, x, max, min):
        a = -0.5; b = 0.5
        x = (x-a)*(max- min)/(b-a)+min
        return x
    
    
    @T.no_grad()
    def test(self, bnn, epoch):
        hp = self.hp
        device = self.args.device
        
        test_dataset = self.dataset(self.rawData, 'test', self.path, hp, device=self.args.device, info=self.info)
        test_loader = DataLoader(test_dataset, batch_size=hp.batchSizeTest, shuffle=False)

        # likelihood = tyxe.likelihoods.HomoskedasticGaussian(hp.numSampTrain, scale=0.1)
        # prior = tyxe.priors.IIDPrior(dist.Normal(T.zeros(1, device=device), T.ones(1, device=device)))
        # guide_factory = tyxe.guides.AutoNormal
        # bnn = tyxe.VariationalBNN(self.model, prior, likelihood, guide_factory)
        
        # # # ------------------------ Load saved weights --------------------------
        # epoch = hp.loadWeightsEpoch
        # bnn, _ = self.loadModel(bnn, epoch)

        predMuLs = []; predSigLs= []; dataLs = []

        for batchIdx, data in enumerate(test_loader):

            input = data[0]  # (currentBatchSize, seq_len, latentDim)
            target = data[1]  # (currentBatchSize, timeStepsUnroll, latentDim)

            last_N_seqs = input
            flag = 0
            for i in range(hp.timeStepsUnroll):
                # pdb.set_trace()
                # ([currentBatchSize, latentDim], [currentBatchSize, latentDim])
                predMu_i, predSig_i = bnn.predict(last_N_seqs, num_predictions=hp.nPredSampsTest)
                # pdb.set_trace()
                last_N_seqs = T.cat((last_N_seqs[:, 1:], predMu_i[:, None]), dim=1)

                if flag:
                    predMu = T.cat((predMu, predMu_i[:, None]), dim=1)  
                else:
                    predMu = predMu_i[:, None]
                    flag = 1

            predMu = self.rescale(predMu, test_dataset.max, test_dataset.min)
            target = self.rescale(target, test_dataset.max, test_dataset.min)

            loss = T.mean(T.abs(predMu - target)/target, 2)*100
            # self.info(f'({batchIdx}) Testing loss: {loss}')

            predMuLs.append(predMu)
            dataLs.append(target)
        
        pdb.set_trace()
        predData = T.cat(predMuLs, dim=0), T.cat(dataLs, dim=0)  # (numSampTest, timeStepModel, M, numNodes)

        self.savePredictions(predData, epoch, False)
        return predData[0][0].detach().cpu().numpy()