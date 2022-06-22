"""
class for model training and testing 
"""

import pdb
from turtle import pd
from unicodedata import decimal
import h5py
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import os
from os.path import dirname, realpath, join


class ModelPipeline():

    def __init__(self, Model, hyperParams, experPaths, rawData, dataset, args):
        """
        ARGS:
            n_sensor (int): number of sensors
        """
        self.args = args
        self.info = args.logger.info

        self.hp = hyperParams
        self.rawData = rawData
        self.dataset = dataset
        self.path = experPaths

        self.model = Model(hyperParams, args).to(args.device)

        self.info(self)
        self.info(self.model)

    
    def saveModel(self, epoch, optimizer, scheduler, losses):
        PATH = join(self.path.weights, f'weights_epoch{epoch}.tar')
        state = {'epoch': epoch,
                 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(), 
                 'losses': losses
                 }
        T.save(state, PATH)
        print(f'model saved at epoch {epoch}')

        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Training Loss')
        plt.yscale("log")
        plt.savefig(join(self.path.run, f'Loss_plot.png'))
        # plt.close()


    def loadModel(self, epoch, optimizer=None, scheduler=None, losses=None):
        """Loads pre-trained network from file"""
        try:
            PATH = join(self.path.weights, f'weights_epoch{epoch}.tar')
            checkpoint = T.load(PATH, map_location=T.device(self.args.device))
            checkpoint_epoch = checkpoint['epoch']
            print(f'Found model at epoch: {checkpoint_epoch}')
        except FileNotFoundError:
            if epoch>0: raise FileNotFoundError(f'Error: Could not find PyTorch network at epoch: {epoch}')
            return

        # Load optimizer/scheduler
        if (not optimizer is None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if (not scheduler is None):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if (not losses is None):
            losses = checkpoint['losses']
        return optimizer, scheduler, losses


    def __repr__(self):
        # numParameters = count_parameters(self.model, self.args.logger)
        description = f'\n\n\
            {self.hp}'
        return description

    
    def trainingEpoch(self, optimizer, train_loader, epoch):
        self.model.train()
        running_loss = 0

        for batchIdx, data in enumerate(train_loader):
            optimizer.zero_grad()

            
            input = data[0]  # (currentBatchSize, seq_len, latentDim)
            target = data[1]  # (currentBatchSize, latentDim)

            pred = self.model(input)  # (currentBatchSize, latentDim)
            loss = self.model.loss_fn(pred, target)
                                              
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        return running_loss



    def train(self):
        hp = self.hp

        train_dataset = self.dataset(self.rawData, 'train', self.path, hp, device=self.args.device, info=self.info)
        train_loader = DataLoader(train_dataset, batch_size=hp.batchSizeTrain, shuffle=True)  

        optimizer = T.optim.Adam(self.model.parameters(), lr=hp.lr,  weight_decay=1e-5)
        lr_lambda = lambda epoch: 1 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        losses = []

        # ------------------- load model from checkpoint -----------------------
        if hp.epochStartTrain:
            optimizer, scheduler, losses = self.loadModel(hp.epochStartTrain, optimizer, scheduler, losses)

        for epoch in range(hp.epochStartTrain+1, hp.numIters):         

            # for g in optimizer.param_groups:
            #     g['lr'] = 0.0001
            epochLr = optimizer.param_groups[0]['lr']  
            if epoch % hp.logInterval == 0: self.info(f'\n \n({epoch:02.0f}), lr: {epochLr:.6f}')
            
            loss = self.trainingEpoch(optimizer, train_loader, epoch)
            losses.append(loss.item())

            # -------------------------- adjusted lr ---------------------------
            scheduler.step()

            # ------------------- Save model periodically ----------------------
            if (epoch % hp.checkpointInterval == 0) and (epoch > 0):
                self.saveModel(epoch, optimizer, scheduler, losses)

            # ------------------------ print progress --------------------------
            if epoch % hp.logInterval == 0: self.info(f'   ({epoch}) Training loss: {loss:.8f}')

    
    def savePredictions(self, predData, epoch, trainBool):
        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        predData_Path = join(self.path.run, f'predDataTest_epoch{epoch}{info}.hdf5')

        predArray, targetArray = predData

        with h5py.File(predData_Path, 'w') as f:
            f.create_dataset('pred', data=predArray.detach().cpu().numpy())
            f.create_dataset('target', data=targetArray.detach().cpu().numpy())
        print(f'pred data saved at {predData_Path}')
    

    def rescale(self, x, max, min):
        a = -1; b = 1
        x = (x-a)*(max- min)/(b-a)+min
        return x
    
    
    def test(self):
        hp = self.hp
        
        test_dataset = self.dataset(self.rawData, 'test', self.path, hp, device=self.args.device, info=self.info)
        test_loader = DataLoader(test_dataset, batch_size=hp.batchSizeTest, shuffle=False)

        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadWeightsEpoch
        self.loadModel(epoch)
        self.model.eval()

        predLs = []; dataLs = []

        for batchIdx, data in enumerate(test_loader):

            input = data[0]  # (currentBatchSize, seq_len, latentDim)
            target = data[1]  # (currentBatchSize, timeStepsUnroll, latentDim)

            last_N_seqs = input
            flag = 0
            for i in range(hp.timeStepsUnroll):
                pred_i = self.model(last_N_seqs)  # (currentBatchSize, latentDim)
                # pred_i = pred_i[0][0]
                # pred_i = _[:, -1]
                last_N_seqs = T.cat((last_N_seqs[:, 1:], pred_i[:, None]), dim=1)

                if flag:
                    pred = T.cat((pred, pred_i[:, None]), dim=1)  
                else:
                    pred = pred_i[:, None]
                    flag = 1

            pred = self.rescale(pred, test_dataset.max, test_dataset.min)
            target = self.rescale(target, test_dataset.max, test_dataset.min)

            loss = T.mean(T.abs(pred - target)/target, 2)*100
            self.info(f'({batchIdx}) Testing loss: {loss}')

            predLs.append(pred)
            dataLs.append(target)
        
        predData = T.cat(predLs, dim=0), T.cat(dataLs, dim=0)  # (numSampTest, timeStepModel, M, numNodes)

        self.savePredictions(predData, epoch, False) 