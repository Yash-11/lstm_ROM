"""
class for Auto-Encoder training
"""
import h5py
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from filterpy.kalman import (unscented_transform, MerweScaledSigmaPoints,
                             JulierSigmaPoints, SimplexSigmaPoints,
                             KalmanFilter)

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
                                              
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        return running_loss



    def train(self):
        hp = self.hp

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
        hp.loadAEWeightsEpoch = hp.minValidAELossEpoch

    
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

        predLv = T.tensor( predData['pred'][0], dtype=T.float32).to(self.args.device) 
        targetLv = T.tensor( predData['target'][0], dtype=T.float32).to(self.args.device) 

        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadAEWeightsEpoch
        self.loadModel(epoch)
        self.model.eval()
        
        pred = self.model(predLv)
        target = self.model(targetLv)

        pred = self.denormalize(pred, self.hp.meanAE, self.hp.stdAE)
        target = self.denormalize(target, self.hp.meanAE, self.hp.stdAE)

        self.saveOutputs(pred, target)

    def decodeLatentVecDistributions(self, ):
        hp = self.hp
        dataset = self.dataset(self.rawData, '', self.path, hp, device=self.args.device, info=self.info)

        # ----------------- Load predicted Latent Vectors ----------------------
        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        loadWeightsEpoch = '_'.join(str(e) for e in hp.loadWeightsEpoch)
        name = f'predDataTest_epoch{loadWeightsEpoch}{info}.hdf5'
        predData = h5py.File(join(self.path.run, name), 'r') 

        # predLv [1, n_sampTest, latentDim]
        # predLv = T.tensor( predData['pred'][:], dtype=T.float32).to(self.args.device) 
        # targetLv = T.tensor( predData['target'][:], dtype=T.float32).to(self.args.device)
        # varLv = T.tensor( predData['var'][:], dtype=T.float32).to(self.args.device) 

        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadAEWeightsEpoch
        self.loadModel(epoch)
        self.model.eval()

        print(predData['pred'].shape, predData['var'].shape, predData['rescaleMinMax'].shape, predData['target'].shape)
        n_sampTest, N, _ = predData['pred'][0].shape
        M = hp.imH 

        mean = predData['pred'][0].reshape((n_sampTest, -1))  # [n_sampTest, N, N]
        var0 = predData['var'][0].reshape((n_sampTest, -1))  # [n_sampTest, N, N]
        _min = predData['rescaleMinMax'][0]
        _max = predData['rescaleMinMax'][1]

        def rescale(x, max, min):
            a = -0.9; b = 0.9
            x = (x-a)*(max- min)/(b-a)+min
            return x

        hp.sampling == 'mean_only'
        # ----------------------------------------------------------------------
        #                           Monte Carlo

        if hp.sampling == 'monte_carlo':
            hp.n_MC_Samp = 500        
            points = T.zeros((n_sampTest, hp.n_MC_Samp, N))

            for i in range(n_sampTest):
                cov = np.diag(var0[i])  # [N, N]
                pts = np.random.multivariate_normal(mean[i], cov, size=hp.n_MC_Samp)  # [n_MC_Samp, N]
                points[i] = T.tensor(pts, dtype=T.float32)
                
            points = points.to(self.args.device) 

            
            ptsN = self.model(T.reshape(points, (n_sampTest*hp.n_MC_Samp, N)))  # [n_sampTest*n_MC_Samp, M]
            ptsN = self.denormalize(ptsN, self.hp.meanAE, self.hp.stdAE)
            ptsN = T.reshape(ptsN, (n_sampTest, hp.n_MC_Samp, M))
            ptsN = ptsN.detach().cpu().numpy()

            meanOut = ptsN.mean(axis=1)
            varOut = ptsN.var(axis=1)
            # covN = np.cov(ptsN.T)
        
        # ----------------------------------------------------------------------
        #                         Unscented Transform

        if hp.sampling == 'unscented_transform':
            meanUT = np.zeros((n_sampTest, M*M))
            varUT = np.zeros((n_sampTest, M*M))
            
            sp = JulierSigmaPoints(n=N*N, kappa=.2)
            Wm, Wc = sp.Wm, sp.Wc

            for i in range(n_sampTest):

                self.info(f"current time step: {i}")
        
                x = mean[i][None]
                P = np.diag(var0[i])  # [N, N]

                try:
                    Xi = sp.sigma_points(x, P)  # [n_UT_Samp, N]
                except:
                    breakpoint()
                Xi = T.tensor(Xi, dtype=T.float32)
                Yi = rescale(Xi, _min, _max)

                Yi = Yi.reshape((-1, N, N))
                n_UT_Samp = Yi.shape[0]
                YYi = T.zeros((n_UT_Samp, M*M))

                bs = 100
                for j in range(n_UT_Samp//bs):
                    yy = Yi[j*bs:bs*(j+1)].to(self.args.device)
                    YYi[j*bs:bs*(j+1)] = self.model(yy).reshape((bs, M*M)).detach().cpu()  # [n_UT_Samp, M, M]

                j = n_UT_Samp//bs
                yy = Yi[j*bs:n_UT_Samp].to(self.args.device)
                YYi[j*bs:n_UT_Samp] = self.model(yy).reshape((n_UT_Samp%bs, M*M)).detach().cpu()  # [n_UT_Samp, M, M]

                # YYi = self.model(Yi.reshape((-1, N, N)).to(self.args.device)).detach().cpu()

                # Yi = Yi.reshape((-1, M*M))  # [n_UT_Samp, M*M]

                Yi = self.denormalize(YYi, self.hp.meanAE, self.hp.stdAE)
                Yi = Yi.numpy()

                # xm [200,] ucov [200, 200]
                xm, ucov = unscented_transform(Yi, Wm, Wc, 0)

                meanUT[i] = xm
                varUT[i] = np.diag(ucov)

            meanOut = meanUT.reshape((n_sampTest, M, M))
            varOut = varUT.reshape((n_sampTest, M, M))

        if hp.sampling == 'mean_only':
            meanOut = self.model(rescale(predLv[0], _min, _max)).detach().cpu().numpy()
            meanOut = self.denormalize(meanOut, self.hp.meanAE, self.hp.stdAE)
            varOut = np.zeros_like(meanOut)
        
        # w = 8
        # plt.figure()
        # plt.plot(meanUT[w], label='UT')
        # plt.plot(meanMC[w], label='MC')
        # plt.legend()
        # plt.savefig('mean.png')
        # plt.close()

        # plt.figure()
        # plt.plot(varUT[w], label='UT')
        # plt.plot(varMC[w], label='MC')
        # plt.legend()
        # plt.savefig('var.png')
        # plt.close()

        
        target = dataset.rawData.data[hp.seq_len:hp.seq_len+hp.timeStepsUnroll]

        info = self.hp.predData_Info if hasattr(self.hp, 'predData_Info') else ''
        predData_Path = join(self.path.run, f'predHDataTest_epoch{loadWeightsEpoch}{info}.hdf5')

        with h5py.File(predData_Path, 'w') as f:
            f.create_dataset('pred', data=meanOut)
            f.create_dataset('target', data=target.numpy())
            f.create_dataset('var', data=varOut)
        print(f'pred data saved at {predData_Path}')
