"""
class for Auto-Encoder training
"""
import h5py
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import os
from os.path import dirname, realpath, join


class AEPipeline():

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
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.info(pytorch_total_params)
        
        self.loss = T.nn.MSELoss(reduction = 'sum')

        self.info(self)
        self.info(self.model)

        if hasattr(self.hp, 'minValidAELossEpoch'):
            self.hp.loadAEWeightsEpoch  = self.hp.minValidAELossEpoch
 

    def saveModel(self, epoch, optimizer, scheduler, losses):
        PATH = join(self.path.weights, f'AEweights_epoch{epoch}.tar')
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
        plt.savefig(join(self.path.run, f'AE_Loss_plot.png'))
        # plt.close()
        plt.close()
        plt.close('all')


    def loadModel(self, epoch, optimizer=None, scheduler=None, losses=None):
        """Loads pre-trained network from file"""
        try:
            PATH = join(self.path.weights, f'AEweights_epoch{epoch}.tar')
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

            pred, _ = self.model(input)  # (currentBatchSize, latentDim)
            loss = self.loss(pred, target)   

            l2_lambda = 0.1
            l2_norm = sum(T.abs(p).sum()
                        for p in self.model.parameters())
        
            loss = loss + l2_lambda * l2_norm

                                              
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        return running_loss



    def train(self):
        hp = self.hp

        #self.info(f'before training: {hp.loadAEWeightsEpoch}')
        train_dataset = self.dataset(self.rawData, 'train', self.path, hp, device=self.args.device, info=self.info)
        train_loader = DataLoader(train_dataset, batch_size=hp.batchSizeTrainAE, shuffle=True)

        optimizer = T.optim.Adam(self.model.parameters(), lr=hp.lrAE, weight_decay=hp.weight_decay)
        lr_lambda = lambda epoch: 1 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        losses = {'train': [], 'valid': []}

        # ------------------- load model from checkpoint -----------------------
        if hp.epochStartTrainAE:
            optimizer, scheduler, losses = self.loadModel(hp.epochStartTrainAE, optimizer, scheduler, losses)

        for epoch in range(hp.epochStartTrainAE+1, hp.numItersAE):         

            epochLr = optimizer.param_groups[0]['lr']              
            loss = self.trainingEpoch(optimizer, train_loader, epoch)/train_dataset.dataTrainX.shape[0]
            losses['train'].append(loss.item())

            # -------------------------- adjusted lr ---------------------------
            scheduler.step()

            # -------------------------- validate -------------------------------
            self.model.eval()
            pred, _ = self.model(train_dataset.dataValidX.to(self.args.device))
            valid_loss = self.loss(train_dataset.dataValidY, pred.cpu())/train_dataset.dataValidX.shape[0]
            losses['valid'].append(valid_loss.item())

            # ------------------- Save model periodically ----------------------
            if (epoch % hp.checkpointIntervalAE == 0) and (epoch > 0):
                self.saveModel(epoch, optimizer, scheduler, losses)

            # ------------------------ print progress --------------------------
            if epoch % hp.logIntervalAE == 0: self.info(f'({epoch}) Training loss: {loss:.8f} Validation loss: {valid_loss:.8f}')

        # set minValidLoss
        dd = self.hp.checkpointIntervalAE
        minValidLoss = min(losses['valid'][::dd])        
        hp.minValidAELossEpoch = losses['valid'].index(minValidLoss)
        #hp.loadAEWeightsEpoch = hp.minValidAELossEpoch

    
    def savePredictions(self, predData, epoch, trainBool):
        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        predData_Path = join(self.path.run, f'predLatentDataTest_epoch{epoch}{info}.hdf5')

        predArray, targetArray = predData

        with h5py.File(predData_Path, 'w') as f:
            f.create_dataset('pred', data=predArray.detach().cpu().numpy())
            f.create_dataset('target', data=targetArray.detach().cpu().numpy())
        print(f'pred data saved at {predData_Path}')
    
    
    def test(self):
        hp = self.hp
        
        test_dataset = self.dataset(self.rawData, 'test', self.path, hp, device=self.args.device, info=self.info)
        test_loader = DataLoader(test_dataset, batch_size=hp.batchSizeTestAE, shuffle=False)

        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadAEWeightsEpoch
        print("AE load epoch: ", epoch)
        self.loadModel(epoch)
        self.model.eval()

        predLs = []; dataLs = []

        for batchIdx, data in enumerate(test_loader):

            input = data[0]  # (currentBatchSize, imDim)
            target = data[1]  # (currentBatchSize, imDim)

            pred, _ = self.model(input)  # (currentBatchSize, latentDim)

            
            loss = T.mean(T.abs(pred - target)/target, 1)
            # self.info(f'({batchIdx}) Testing loss: {loss*100}')

            predLs.append(pred)
            dataLs.append(target)
        
        predData = T.cat(predLs, dim=0), T.cat(dataLs, dim=0)  # (numSampTest, latentDim)

        self.savePredictions(predData, epoch, False)


    def saveLatentVecs(self, LatentVecs):
        path = join(self.path.run, f'LatentVecs.npy')
        np.save(path, LatentVecs)

    
    def generateLatentVecs(self,):
        hp = self.hp
        
        encode_dataset = self.dataset(self.rawData, 'encode', self.path, hp, device=self.args.device, info=self.info)
        encode_loader = DataLoader(encode_dataset, batch_size=hp.batchSizeEncode, shuffle=False)

        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadAEWeightsEpoch
        self.loadModel(epoch)
        self.model.eval()

        LatentVecs = []

        for batchIdx, data in enumerate(encode_loader):

            input = data  # (currentBatchSize, imDim)
            _, _LatentVecs = self.model(input)  # (currentBatchSize, latentDim)
            LatentVecs.append(_LatentVecs)

        LatentVecs = T.cat(LatentVecs, 0).detach().cpu().numpy()  # (numSampTrainAE, latentDim)
        self.saveLatentVecs(LatentVecs)
        return LatentVecs


    def saveOutputs(self, pred, target):
        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        predData_Path = join(self.path.run, f'predHDataTest_epoch{self.hp.loadWeightsEpoch}{info}.hdf5')

        with h5py.File(predData_Path, 'w') as f:
            f.create_dataset('pred', data=pred.detach().cpu().numpy())
            f.create_dataset('target', data=target.detach().cpu().numpy())
        print(f'pred data saved at {predData_Path}')


    def denormalize(self, data, mean, std):
        return data*std + mean
    

    def decodeLatentVecs(self,):

        hp = self.hp
        dataset = self.dataset(self.rawData, '', self.path, hp, device=self.args.device, info=self.info)


        # ----------------- Load predicted Latent Vectors ----------------------
        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        name = f'predDataTest_epoch{hp.loadWeightsEpoch}{info}.hdf5'
        predData = h5py.File(join(self.path.run, name), 'r')

        predLv = T.tensor( predData['pred'][:], dtype=T.float32).to(self.args.device) 
        targetLv = T.tensor( predData['target'][:], dtype=T.float32).to(self.args.device) 

        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadAEWeightsEpoch
        self.loadModel(epoch)
        self.model.eval()

        pred = self.model(predLv[0])
        target = dataset.rawData.data[hp.seq_len:hp.seq_len+hp.timeStepsUnroll]

        print(self.hp.meanAE, self.hp.stdAE)
        pred = self.denormalize(pred, self.hp.meanAE, self.hp.stdAE)

        self.saveOutputs(pred, target)