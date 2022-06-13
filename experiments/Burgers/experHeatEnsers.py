
# def setHyperParams(hp):
#     # model 
#     hp.hiddenDim = 128
#     hp.latentDim = 81
#     hp.seq_len = 7

#     # training
#     hp.numIters = 5001
#     hp.lr = 0.0005
#     hp.batchSizeTrain = 45
#     hp.epochStartTrain = 0

#     # testing
#     hp.loadWeightsEpoch = 5000
#     hp.batchSizeTest = 1
#     hp.timeStepsUnroll = 25

#     # data
#     hp.numSampTrain = 45
#     hp.numSampTest = 1

#     # logging
#     hp.save = 1
#     hp.show = 0
#     hp.saveLogs = 1
#     hp.saveInterval = 20
#     hp.logInterval = 100
#     hp.checkpointInterval = 5000