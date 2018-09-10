import torch
import numpy as np
import dataloader


def getAccuracy(output, val_y, top):
    '''
    output: Variable type. net(val_x). output of validation set.
    val_y: LongTensor type. the target of validation set. 
    top: if it equals to k, top k accuracy
    '''
    pred = torch.topk(output, top)[1]
    accuracy = sum([val_y.cpu().numpy()[i] in pred.data.cpu().numpy()[i] for i in range(val_y.size(0))]) / val_y.size(0)
    return accuracy


def makePredictionCPU(net, valxFile, valy, size, printMistake = True):
    '''
    net: torch.nn.Module
    valxFile: a list of file name of validation data. because valx is usually very huge in size, so only pass their file names.
    valy: label of validation set.
    size: resize image to size*size
    printMistake: a boolean. whether to print pics which are mis-categorized.
    NOTICE, the valxFile and valy must be match each other,
    that is: label of valxFile[i] is valy[i]
    return: top 1 accuracy
    '''
    num = 100 # read 100 validation img each time
    total = len(valxFile)
    count = 0
    
    pred = torch.empty(total).type(torch.LongTensor)
    
    net.eval()
    net.cpu()
    
    while count < total:
        fileList = valxFile[count:(count+num)]
        valx = dataloader.loadImagesList(fileList, size)
        val_output = net(valx)
        pred_y1 = torch.max(val_output, 1)[1].data.squeeze()
        pred[count:(count+num)] = pred_y1
        # accuracy of this 100 pics
        accuracy1 = torch.sum(pred_y1 == valy[count:(count+num)]).numpy() / min(num, total-count)
        print('TOP 1 val accuracy of this batch: ', accuracy1)
        if accuracy1 < 0.6:
            print("accuracy is too low, printing image list in this batch")
            # print(fileList)
        if printMistake:
            mistake = pred_y1 == valy[count:(count+num)]
            for idx, r in enumerate(mistake):
                if r == 0:
                    print(fileList[idx] + " got: " + str(pred_y1[idx]) + " but expected:" + str(valy[count+idx]) )
        del valx
        del val_output
        count += num
    
    # fileList = valxFile[count:]
    # valx = dataloader.loadImagesList(fileList, size)
    # val_output = net(valx)
    # pred_y1 = torch.max(val_output, 1)[1].data.squeeze()
    # pred[count:] = pred_y1
    # del valx
    # del val_output

    accuracy1 = torch.sum(pred == valy).numpy() / valy.size(0)
    print('TOP 1 val accuracy: ', accuracy1)
    
    return accuracy1
    

def makePredictionGPU(net, valxFile, valy):
    '''
    the same function as CPU, but on GPU
    net: torch.nn.Module
    valxFile: a list of file name of validation data. because valx is usually very huge in size, so only pass their file names.
    valy: label of validation set.
    NOTICE, the valxFile and valy must be match each other,
    that is: label of valxFile[i] is valy[i]
    return: top 1 accuracy
    '''
    pass




