import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision 
import matplotlib.pyplot as plt
import scipy.misc
from skimage import transform
import numpy as np
import os
import sys
import time
from PIL import Image
import gc

import alexnet
import dataloader
import learncurve
import postprocess



print("Welcome! Starting from a breakpoint.")

BREAKPOINT = '../result/18090702/tmall_brand_classifier.pkl'
# 0. hyperparameters
BATCHSIZE = 50  # mini batch size
EPOCH = 20      # number of epoch
LR = 0.0003     # learning rate
WD = 1e-6       # weight decay in Adam
ISGPU = True    # use GPU or not
NH = 128        # num of neuron in hidden layer
DP = 0.5        # drop out
SZ = 224        # input img size


print("Reading a model at" + BREAKPOINT)

# network architecture: load the breakpoint
alex = torch.load(BREAKPOINT)
if ISGPU:
    alex.cuda()

print("Calculating the prediction of loaded model") # full validation set 30% of tmall pic
valxFile, valy = dataloader.loadValidationDataFileName(percentage = 0.3)
postprocess.makePredictionCPU(alex, valxFile, valy, SZ, True)

# in makePredictionCPU, alex will be reset to cpu mode
if ISGPU:
    alex.cuda()




print("Continue Training")

# training set: preprocess and load. 70% of whole training data
start = time.time()
dataTensor, targetTensor = dataloader.loadTrainData(0.7, SZ)
end = time.time()
print("training data has been loaded in %d seconds"%(end-start))

# validation set: 3% of whole training data
# only use for monitor learning curve
start = time.time()
valx, valy = dataloader.loadValidationData(0.02, SZ)  # only for learning curve visualisation
end = time.time()
print("validation data has been loaded in %d seconds"%(end-start))

# mini batch
loader = dataloader.miniBatchDataLoader(dataTensor, targetTensor, BATCHSIZE, 3)
print("mini-batch dataloader is ready")

# optimizer and loss function
optimizer = torch.optim.Adam(alex.parameters(), lr=LR, weight_decay=WD)
if ISGPU:
    lossfun = nn.CrossEntropyLoss().cuda()
else:
    lossfun = nn.CrossEntropyLoss()


# training
print("begin to train")
lossarray = np.array([])
accuracy1array = np.array([])
# accuarcy5array = np.array([]) # currently useless. only 3 categories
start = time.time()
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(loader): # x FloatTensor, y Long Tensor
        if ISGPU:
            bx = Variable(x).cuda()
            by = Variable(y).cuda()
        else:
            bx = Variable(x)
            by = Variable(y)
        output = alex(bx)
        loss = lossfun(output, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # learning curve
        if step % 50 == 0:
            alex.eval()  # don't drop out and fix moving mean/var in BN
            alex.cpu()
            val_output = alex(valx)
            # top 1 accuracy
            pred_y1 = torch.max(val_output, 1)[1].data.squeeze()
            accuracy1 = torch.sum(pred_y1 == valy).numpy() / valy.size(0)
            # top 5 accuracy. useless at present, only 3 categories
            # pred_y5 = torch.topk(val_output, 5)[1]
            # accuracy5 = sum([val_y.cpu().numpy()[i] in pred_y5.data.cpu().numpy()[i] for i in range(val_y.size(0))]) / val_y.size(0)
            print('Epoch: ', epoch, ' | train loss: %.4f' % loss.data, ' | TOP 1 val accuracy: ', accuracy1)
            # print('Epoch: ', epoch, ' | train loss: %.4f' % loss.data, ' | TOP 1 val accuracy: ', accuracy1, ' | TOP 5 val accuracy: ', accuracy5)
            lossarray = np.append(lossarray, loss.data)
            accuracy1array = np.append(accuracy1array, accuracy1)
            # accuarcy5array = np.append(accuarcy5array, accuracy5)
            if ISGPU:
                alex.cuda()
            alex.train() # enable drop out and free moving mean/var in BN
            del val_output
    print("Epoch ", epoch, " has been trained.")
end = time.time()
print("finish training in ", (end - start))

# save model
np.savetxt("loss.txt", lossarray)
np.savetxt("accuracy1.txt", accuracy1array)  # accuracy of small validation data (3%)
# np.savetxt("accuracy5.txt", accuracy5array)
torch.save(alex, 'tmall_brand_classifier.pkl')

# learning curve
learncurve.learningCurve(lossarray, accuracy1array, window = 5)

# final accuray
# final top 1
alex.eval()
alex.cpu()

del dataTensor
del targetTensor
del loader
gc.collect()

valxFile, valy = dataloader.loadValidationDataFileName(percentage = 0.3)  # full validation data (30%)
postprocess.makePredictionCPU(alex, valxFile, valy, SZ, True)

# final top 5
# pred_y5 = torch.topk(val_output, 5)[1]
# accuracy5 = sum([val_y.numpy()[i] in pred_y5.data.numpy()[i] for i in range(val_y.size(0))]) / val_y.size(0)
# print('TOP 5 val accuracy: ', accuracy1)


