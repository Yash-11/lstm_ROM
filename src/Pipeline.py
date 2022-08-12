"""
class for model training and testing 
"""

from genericpath import exists
import pdb
from turtle import pd
from unicodedata import decimal
import h5py
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import pandas as pd
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
        self.info = args.info

        self.hp = hyperParams
        self.rawData = rawData
        self.dataset = dataset
        self.path = experPaths

        self.model = Model(hyperParams, args).to(args.device)
        self.loss = T.nn.MSELoss(reduction = 'sum')

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
        self.info(f'model saved at epoch {epoch}')

        plt.figure()
        plt.plot(losses['train'], label='train')
        plt.plot(losses['valid'], label='valid')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Training Loss')
        plt.legend()
        plt.yscale("log")
        plt.savefig(join(self.path.run, f'Loss_plot.png'))
        plt.close()
        plt.close('all')


    def loadModel(self, epoch, optimizer=None, scheduler=None, losses=None):
        """Loads pre-trained network from file"""
        try:
            PATH = join(self.path.weights, f'weights_epoch{epoch}.tar')
            checkpoint = T.load(PATH, map_location=T.device(self.args.device))
            checkpoint_epoch = checkpoint['epoch']
            self.info(f'Found model at epoch: {checkpoint_epoch}')
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



    def saveMinLoss(self, losses):
        minTrainLoss = min(losses['train'])
        minValidLoss = min(losses['valid'])
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

    
    
    def train(self):
        hp = self.hp

        train_dataset = self.dataset(self.rawData, 'train', self.path, hp, device=self.args.device, info=self.info)
        train_loader = DataLoader(train_dataset, batch_size=hp.batchSizeTrain, shuffle=True)  

        optimizer = T.optim.Adam(self.model.parameters(), lr=hp.lr,  weight_decay=1e-4)
        lr_lambda = lambda epoch: 1 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        losses = {'train': [], 'valid': []}

        # ------------------- load model from checkpoint -----------------------
        if hp.epochStartTrain:
            optimizer, scheduler, losses = self.loadModel(hp.epochStartTrain, optimizer, scheduler, losses)

        for epoch in range(hp.epochStartTrain+1, hp.numIters):         

            self.model.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 0.0001
            epochLr = optimizer.param_groups[0]['lr']  
            
            loss = self.trainingEpoch(optimizer, train_loader, epoch)/hp.numSampTrain
            if T.isnan(loss):break
            losses['train'].append(loss.item())

            # -------------------------- adjusted lr ---------------------------
            scheduler.step()

            # -------------------------- validate -------------------------------
            self.model.eval()
            pred = self.model(train_dataset.dataValidX.to(self.args.device))
            valid_loss = self.loss(train_dataset.dataValidY, pred.cpu())/hp.numSampValid
            losses['valid'].append(valid_loss.item())

            # ------------------- Save model periodically ----------------------
            if (epoch % hp.checkpointInterval == 0) and (epoch > 0):
                self.saveModel(epoch, optimizer, scheduler, losses)

            # ------------------------ print progress --------------------------
            if epoch % hp.logInterval == 0: self.info(f'({epoch}) Training loss: {loss:.8f} Validation loss: {valid_loss:.8f}')
        
        self.saveMinLoss(losses)

    
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
                # pdb.set_trace()
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

        # pdb.set_trace()
        self.savePredictions(predData, epoch, False)
        return predData[0][0].detach().cpu().numpy()