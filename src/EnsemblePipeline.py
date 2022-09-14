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

        self.models = []
        for i in range(self.hp.n_modelEnsemble):
            self.models.append(Model(hyperParams, args).to(args.device))
        self.loss = T.nn.MSELoss(reduction = 'sum')

        self.info(self)
        self.info(self.models[0])

    
    def saveModel(self, model_idx, epoch, optimizer, scheduler, losses):
        model = self.models[model_idx]

        PATH = join(self.path.weights, f'weights{model_idx}_epoch{epoch}.tar')
        state = {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(), 
                 'losses': losses
                 }
        T.save(state, PATH)
        self.info(f'model saved at epoch {epoch}')

        plt.figure()
        plt.plot(losses['train'], label='train')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(join(self.path.run, f'Loss_plot{model_idx}.png'))
        plt.close()

        plt.figure()
        plt.plot(losses['valid'], label='valid')        
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Validation Loss')
        plt.legend()
        plt.yscale("log")
        plt.savefig(join(self.path.run, f'ValidLoss_plot{model_idx}.png'))        
        plt.close()

        plt.close('all')


    def loadModel(self, model_idx, epoch, optimizer=None, scheduler=None, losses=None):
        """Loads pre-trained network from file"""
        try:
            PATH = join(self.path.weights, f'weights{model_idx}_epoch{epoch}.tar')
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
        self.models[model_idx].load_state_dict(checkpoint['model_state_dict'])
        if (not losses is None):
            losses = checkpoint['losses']
        return optimizer, scheduler, losses


    def __repr__(self):
        # numParameters = count_parameters(self.model, self.args.logger)
        description = f'\n\n\
            {self.hp}'
        return description

    
    def trainingEpoch(self, model_idx, optimizer, train_loader, epoch):
        model = self.models[model_idx]
        model.train()
        running_loss = 0

        for batchIdx, data in enumerate(train_loader):
            optimizer.zero_grad()

            
            input = data[0]  # (currentBatchSize, seq_len, latentDim)
            target = data[1]  # (currentBatchSize, latentDim)

            mu, var = model(input)  # (currentBatchSize, latentDim)
            loss = model.loss_fn(mu, target, var)
                                              
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        return running_loss



    def saveMinLoss(self, model_idx, losses):
        minTrainLoss = min(losses['train'])
        dd = self.hp.checkpointInterval
        minValidLoss = min(losses['valid'][::dd])
        minTrainEpoch = losses['train'].index(minTrainLoss)        
        minValidEpoch = losses['valid'].index(minValidLoss)

        info = {'name': self.hp.runName+f'model{model_idx}', 'minValidLoss': minValidLoss, 'minValidEpoch': minValidEpoch, 
        'minTrainLoss': minTrainLoss, 'minTrainEpoch': minTrainEpoch}

        path = join(self.path.experDir, f'minLoss.csv')

        try:
            df =  pd.read_csv(path)
        except:
            df = pd.DataFrame(columns=['name', 'minValidLoss', 'minValidEpoch', 'minTrainLoss', 'minTrainEpoch'])
        
        df = df.append(info, ignore_index=True)
        df = df.sort_values(by=['minValidLoss'])
        df.to_csv(path, index=False)

    
    def train(self):
        hp = self.hp

        train_dataset = self.dataset(self.rawData, 'train', self.path, hp, device=self.args.device, info=self.info)
        train_loader = DataLoader(train_dataset, batch_size=hp.batchSizeTrain, shuffle=True) 

        for model_idx, model in enumerate(self.models):
            self.trainModel(model_idx, train_dataset, train_loader)
    
    
    def trainModel(self, model_idx, train_dataset, train_loader):
        hp = self.hp 
        model = self.models[model_idx]

        optimizer = T.optim.Adam(model.parameters(), lr=hp.lr,  weight_decay=1e-4)
        lr_lambda = lambda epoch: 1 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        losses = {'train': [], 'valid': []}

        # ------------------- load model from checkpoint -----------------------
        if hp.epochStartTrain:
            optimizer, scheduler, losses = self.loadModel(model_idx, hp.epochStartTrain, optimizer, scheduler, losses)

        for epoch in range(hp.epochStartTrain+1, hp.numIters):         

            model.train()
            epochLr = optimizer.param_groups[0]['lr']  
            
            loss = self.trainingEpoch(model_idx, optimizer, train_loader, epoch)/hp.numSampTrain
            if T.isnan(loss):break
            losses['train'].append(loss.item())

            # -------------------------- adjusted lr ---------------------------
            scheduler.step()

            # -------------------------- validate -------------------------------
            model.eval()
            pred, var = model(train_dataset.dataValidX.to(self.args.device))
            valid_loss = self.loss(train_dataset.dataValidY, pred.cpu())/hp.numSampValid
            losses['valid'].append(valid_loss.item())

            # ------------------- Save model periodically ----------------------
            if (epoch % hp.checkpointInterval == 0) and (epoch > 0):
                self.saveModel(model_idx, epoch, optimizer, scheduler, losses)

            # ------------------------ print progress --------------------------
            if epoch % hp.logInterval == 0: self.info(f'({epoch}) Training loss: {loss:.8f} Validation loss: {valid_loss:.8f}')
        
        self.saveMinLoss(model_idx, losses)


    def savePredictions(self, predData, epoch, trainBool):
        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        predData_Path = join(self.path.run, f'predDataTest_epoch{epoch}{info}.hdf5')

        predArray, targetArray, varArray, rescaleMinMax = predData

        with h5py.File(predData_Path, 'w') as f:
            f.create_dataset('pred', data=predArray.detach().cpu().numpy())
            f.create_dataset('target', data=targetArray.detach().cpu().numpy())
            f.create_dataset('var', data=varArray.detach().cpu().numpy())
            f.create_dataset('rescaleMinMax', data=rescaleMinMax)
        print(f'pred data saved at {predData_Path}')
    
    
    def test(self):
        hp = self.hp
        
        test_dataset = self.dataset(self.rawData, 'test', self.path, hp, device=self.args.device, info=self.info)
        test_loader = DataLoader(test_dataset, batch_size=hp.batchSizeTest, shuffle=False)

        # ------------------------ Load saved weights --------------------------
        for model_idx, epoch_i in enumerate(hp.loadWeightsEpoch):
            print(model_idx)
            self.loadModel(model_idx, epoch_i)
            self.models[model_idx].eval()

        predMuLs = []; predVarLs= []; dataLs = []

        for batchIdx, data in enumerate(test_loader):

            input = data[0]  # (currentBatchSize, seq_len, latentDim)
            target = data[1]  # (currentBatchSize, timeStepsUnroll, latentDim)

            last_N_seqsMu = input

            flag = 0
            for i in range(hp.timeStepsUnroll):

                pred_i = 0
                var_i = 0

                for model_idx, model in enumerate(self.models[:]):
                    mean_ij, var_ij = model(last_N_seqsMu)  # (currentBatchSize, latentDim)

                    pred_i = pred_i + mean_ij /hp.n_modelEnsemble
                    var_i = var_i + (var_ij+mean_ij**2)  /hp.n_modelEnsemble

                var_i = var_i - pred_i**2
                if not T.all(var_i > 0):
                    var_i += hp.epsilonLatentVar

                # pred_i, var_i = self.models[0](last_N_seqsMu)  # (currentBatchSize, latentDim)
                
                last_N_seqsMu = T.cat((last_N_seqsMu[:, 1:], pred_i[:, None]), dim=1)

                if flag:
                    pred = T.cat((pred, pred_i[:, None]), dim=1)
                    var = T.cat((var, var_i[:, None]), dim=1)
                else:
                    pred = pred_i[:, None]
                    var = var_i[:, None]
                    flag = 1

            predMuLs.append(pred)
            predVarLs.append(var)
            dataLs.append(target)
        
        predData = T.cat(predMuLs, dim=0), T.cat(dataLs, dim=0), T.cat(predVarLs, dim=0), np.array([test_dataset.max, test_dataset.min])  # (numSampTest, timeStepModel, M, numNodes)

        self.savePredictions(predData, '_'.join(str(e) for e in hp.loadWeightsEpoch), False) 
        return predData[0][0].detach().cpu().numpy()