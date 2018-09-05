import os
import torch
import preprocess
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np



def loadTrainData(percentage = 1.0, size = 224):
    '''
    percentage: only load first percentage*100% as training set.
    size: images will be resize to size * size. 
    return: data, label
    data: torch.FloatTensor ( 3764*percentage, 3, size, size)
    label: torch.LongTensor (3764*percentage)
    '''
    # start = time.time()
    # constant
    labelPath = '../json/label.dat'
    imgPath = '../train/'
    
    # make label dict
    labelDict = dict()
    labelFile = open(labelPath, 'r')
    for line in labelFile:
        idx, brand = line.strip().split(" ")
        labelDict[brand] = int(idx)
    labelFile.close()
        
    # get the number of training pic
    trainsize = 0
    for brand in labelDict:
        brandPath = imgPath + brand
        allImg = os.listdir(brandPath)
        totalImg = len(allImg)
        num = int(totalImg * percentage)
        trainsize += num
    trainx = torch.empty(trainsize, 3, size, size)
    trainy = []
    
    # read image. brand by brand
    count = 0
    for brand in labelDict:
        # for each brand, get the num of imgs in training set
        brandPath = imgPath + brand
        allImg = os.listdir(brandPath)
        totalImg = len(allImg)
        num = int(totalImg * percentage)
        # select first num of imgs
        for idx, img in enumerate(allImg):
            if idx == 1000:
                print(img)
                # output should be:
                # d792d0eba6b447049a28686b9298915d.jpg
                # 6df9e31532af4810a02d5abb9548f1f1.jpg
            if idx < num:
                imgTmp = preprocess.readImage(brandPath + "/" + img)
                imgTmp2 = preprocess.preprocess(imgTmp, size)
                imgout = torch.from_numpy(imgTmp2).permute(0,3,1,2).type(torch.FloatTensor)
                trainy.append(labelDict[brand])
                trainx[count:(count+1)] = imgout
                count += 1
    # end1 = time.time()
    # print("finish computation in " + str(end1 -start) )
    # convert to Tensor 
    label = torch.LongTensor(trainy)
    print(count)
    print(label.shape)
    print(trainx.shape)
    # return
    return trainx, label
    


def loadValidationData(percentage = 0.3, size = 224):
    '''
    percentage: only load the last percentage*100% as validation data
    size: image will be resize to size*size
    return data, label
    data: Variable(torch.FloatTensor) ( 3764*percentage, 3, size, size)
    label: torch.LongTensor (3764*percentage)
    '''    
    # constant
    labelPath = '../json/label.dat'
    imgPath = '../train/'
    
    # make label dict
    labelDict = dict()
    labelFile = open(labelPath, 'r')
    for line in labelFile:
        idx, brand = line.strip().split(" ")
        labelDict[brand] = int(idx)
    labelFile.close()
        
    # get the number of training pic
    valsize = 0
    for brand in labelDict:
        brandPath = imgPath + brand
        allImg = os.listdir(brandPath)
        totalImg = len(allImg)
        num = int(totalImg * percentage)
        valsize += num
        
    valx = torch.empty(valsize, 3, size, size)
    valy = []
    
    # read image. brand by brand
    count = 0
    for brand in labelDict:
        # for each brand, get the num of imgs in training set
        brandPath = imgPath + brand
        allImg = os.listdir(brandPath)
        totalImg = len(allImg)
        num = int(totalImg * (1 - percentage))
        # select first num of imgs
        for idx, img in enumerate(allImg):
            if idx > num:
                imgTmp = preprocess.readImage(brandPath + "/" + img)
                imgTmp2 = preprocess.preprocess(imgTmp, size)
                imgout = torch.from_numpy(imgTmp2).permute(0,3,1,2).type(torch.FloatTensor)
                valy.append(labelDict[brand])
                valx[count:(count+1)] = imgout
                count += 1
    # end1 = time.time()
    # print("finish computation in " + str(end1 -start) )
    # convert to Tensor 
    label = torch.LongTensor(valy)
    data = Variable(valx)
    print(count)
    print(label.shape)
    print(data.shape)
    # return
    return data, label



def miniBatchDataLoader(dataTensor, targetTensor, batchSize, numWorkers):
    '''
    used in the mini-batch training
    dataTensor: FloatTensor (number of images e.g., 2636, width, width, channel)
    targetTensor: LongTensor (number of images e.g., 2636)
    batchSize: batch size in mini-batch training
    numWorkers: number of CPU threads
    output: a torch.util.data.DataLoader type.
    '''
    torch_dataset = Data.TensorDataset(dataTensor, targetTensor)
    loader = Data.DataLoader( dataset = torch_dataset,
                            batch_size = batchSize,
                            shuffle = True,
                            num_workers=numWorkers)
    return loader



