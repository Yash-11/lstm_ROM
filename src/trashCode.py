


####################### Auto regrassive Test function for for lstm

def test(self):
        hp = self.hp
        
        test_dataset = self.dataset(self.rawData, 'test', self.path, hp, device=self.args.device, info=self.info)
        test_loader = DataLoader(test_dataset, batch_size=hp.batchSizeTest, shuffle=False)

        self.max = test_dataset.max
        self.min = test_dataset.min


        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadWeightsEpoch
        self.loadModel(epoch)
        self.model.eval()

        predLs = []; dataLs = []

        for batchIdx, data in enumerate(test_loader):

            input = data[0]  # (currentBatchSize, init_seqs, latentDim)
            target = data[1]  # (currentBatchSize, timeStepsUnroll, latentDim)

            init_seqs = input
            pred, hc_i = self.model(init_seqs)

            # pred_i_ls = []
            # pred_i = pred_0[:, -1][:, None]  # (currentBatchSize, 1, latentDim)
            # pred_i_ls.append(pred_i)

            # self.model.resetHidden = False

            # for i in range(hp.timeStepsUnroll-1):

            #     self.model.rl_h = hc_i[0]
            #     self.model.rl_c = hc_i[1]
            #     pred_i, hc_i = self.model(pred_i)  # (currentBatchSize, 1, latentDim)
            #     pred_i_ls.append(pred_i)

            # pred = T.cat(pred_i_ls, dim=1)

            # pdb.set_trace()
            # pred[:, 20]
            # target[:, 20]
            # loss = T.mean(T.abs((pred - target)/target), 2)
            # self.info(f'({batchIdx}) Testing loss: {loss*100}')

            
            pred = self.rescale(pred)
            target = self.rescale(target)
            # pdb.set_trace()
            predLs.append(pred)
            dataLs.append(target)
        
        predData = T.cat(predLs, dim=0), T.cat(dataLs, dim=0)  # (numSampTest, timeStepModel, M, numNodes)

        self.savePredictions(predData, epoch, False)


############################# Auto regrassive Test function for for lstm

def test(self):
        hp = self.hp
        
        test_dataset = self.dataset(self.rawData, 'test', self.path, hp, device=self.args.device, info=self.info)
        test_loader = DataLoader(test_dataset, batch_size=hp.batchSizeTest, shuffle=False)

        # ------------------------ Load saved weights --------------------------
        epoch = hp.loadWeightsEpoch
        self.loadModel(epoch)
        self.model.eval()

        self.max = test_dataset.max
        self.min = test_dataset.min

        predLs = []; dataLs = []

        for batchIdx, data in enumerate(test_loader):

            input = data[0]  # (currentBatchSize, init_seqs, latentDim)
            target = data[1]  # (currentBatchSize, timeStepsUnroll, latentDim)

            init_seqs = input
            pred_0, hc_i = self.model(init_seqs)

            pred_i_ls = []
            pred_i = pred_0[:, -1][:, None]  # (currentBatchSize, 1, latentDim)
            pred_i_ls.append(pred_i)

            self.model.resetHidden = False

            for i in range(hp.timeStepsUnroll-1):

                self.model.rl_h = hc_i[0]
                self.model.rl_c = hc_i[1]
                pred_i, hc_i = self.model(pred_i)  # (currentBatchSize, 1, latentDim)
                pred_i_ls.append(pred_i)

            pred = T.cat(pred_i_ls, dim=1)

            pdb.set_trace()
            loss = T.mean(T.abs((pred - target)/target), 2)
            self.info(f'({batchIdx}) Testing loss: {loss*100}')

            predLs.append(self.rescale(pred))
            dataLs.append(self.rescale(target))
        
        predData = T.cat(predLs, dim=0), T.cat(dataLs, dim=0)  # (numSampTest, timeStepModel, M, numNodes)

        self.savePredictions(predData, epoch, False) 


############################# Dataset


rawData = self.rawData.LatentVecs  # (timeSteps, latentDim)
rawData, self.max, self.min = self.rescale(rawData, a = -1, b = 1)


test_init_seq = 20
rawDataTrain = rawData[0 : numSampTrain*seq_len+1]
rawDataTest = rawData[numSampTrain*seq_len+1 : numSampTrain*seq_len+1 + timeStepsUnroll+test_init_seq+1]

self.dataTrainX = T.zeros((numSampTrain, seq_len, latentDim))
self.dataTrainY = T.zeros((numSampTrain, seq_len, latentDim))
self.dataTestX = T.zeros((numSampTest, test_init_seq, latentDim))
self.dataTestY = T.zeros((numSampTest, timeStepsUnroll, latentDim))
# self.dataTestX = T.zeros((numSampTest, seq_len, latentDim))
# self.dataTestY = T.zeros((numSampTest, seq_len, latentDim))

for i in range(numSampTrain):

    self.dataTrainX[i] = rawDataTrain[i*seq_len:i*seq_len+seq_len,:]
    self.dataTrainY[i] = rawDataTrain[i*seq_len+1:i*seq_len+seq_len+1,:]

for i in range(numSampTest):

    # self.dataTestX[i] = T.stack([rawDataTest[i+sl, :] for sl in range(test_init_seq)], dim=0)
    # self.dataTestY[i] = rawDataTest[i + test_init_seq: i + test_init_seq +timeStepsUnroll,:]
    # self.dataTestX[i] = rawDataTrain[i*seq_len:i*seq_len+seq_len,:]
    # self.dataTestY[i] = rawDataTrain[i*seq_len+1:i*seq_len+seq_len+1,:]

    # pdb.set_trace()
    i = 75
    self.dataTestX[0] = T.stack([rawDataTrain[i+sl, :] for sl in range(test_init_seq)], dim=0)
    self.dataTestY[0] = rawData[i + test_init_seq: i + test_init_seq +timeStepsUnroll,:]