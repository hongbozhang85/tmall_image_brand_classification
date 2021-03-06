### 18 Sep 04. 

build the whole architecture of the classifier, and test the code.

1. 18-sep-04-exp01. crash. testing code: code is working. but memory is not enough, process will be killed after some computation
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 30% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p cpu
2. 18-sep-04-exp02. crash. testing code: code is working on gpu. but cpu memory is not enough, minibatch loader will be killed
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 30% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
3. 18-sep-04-exp03. crash. 18-sep-04-exp02 may be caused by not restart kernel in jupyter-notebook. This time, restart kernel and run again. `DataLoader worker (pid 2696) is killed by signal: Killed`
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 30% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
4. 18-sep-04-exp04. crash. testing code: reduce mini batch size to 10. still `DataLoader worker (pid 2696) is killed by signal: Killed`
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=10 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 30% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
5. 18-sep-04-exp05. crash. get rid of jupyter-notebook, run in terminal. still the same bug.
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=10 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 30% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
6. 18-sep-04-exp06. crash. decrease BATCHSIZE further! 
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=5 EPOCH=5 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 30% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
7. 18-sep-04-exp07. crash. even BATCHSIZE=5 is not enough! reduce the size of trianing set. 
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=5 EPOCH=5 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 10% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
8. 18-sep-04-exp08. dataloader crash solved by adding `del val_output`!
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=5 EPOCH=5 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 10% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
	+ 130s, 69.6%. result in ***result/18090408/***
9. 18-sep-04-exp09. increase BATCHSIZE and EPOCH
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 30% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
	+ 807s, 83.9%. result in ***result/18090409/***


### 18 Sep 05

1. 18-sep-05-exp01. increase size of training set. out of cpu memory in calulating `alexnet(valx)`
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic
	+ validation set: last 30% of tmall pic
	+ on T470p gpu
2. 18-sep-05-exp02. reduce size of validation set to 10% to save cpu memory. get killed, don't know why
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic
	+ validation set: last 10% of tmall pic
	+ on T470p gpu
3. 18-sep-05-exp03. reduce size of validation set further to 3% to save cpu memory.
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ BATCHSIZE=20, gpu memory (2G in total) usage 60%. PCI bandwidth utilization < 60%
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
	+ 1890s, 88.4%. result in ***result/18090503/***
4. 18-sep-05-exp04. 90.8%. testing function makePrediction, 0 EPOCH.
	+ reading model of 18-sep-05-exp03, full validation set (30%). accuracy: 90.8%
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=0 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic
	+ validation set in learning curve: last 3% of tmall pic
	+ validation set: last 30% of tmall pic
	+ on T470p gpu
	```
	100 
	torch.Size([100, 3, 224, 224]) 
	TOP 1 val accuracy of this batch:  0.88 
	100 
	torch.Size([100, 3, 224, 224]) 
	TOP 1 val accuracy of this batch:  0.94 
	100 
	torch.Size([100, 3, 224, 224]) 
	TOP 1 val accuracy of this batch:  0.9
	100
	torch.Size([100, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.95
	100
	torch.Size([100, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.97
	100
	torch.Size([100, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.99
	100
	torch.Size([100, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.95
	100
	torch.Size([100, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.95
	100
	torch.Size([100, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.96
	100
	torch.Size([100, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.79
	accuracy is too low, printing image list in this batch
	100
	torch.Size([100, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.78
	accuracy is too low, printing image list in this batch
	28
	torch.Size([28, 3, 224, 224])
	TOP 1 val accuracy of this batch:  0.6785714285714286
	accuracy is too low, printing image list in this batch
	```
5. 18-sep-05-exp05. training from breakpoint for another 20 epoches. accuracy is still 90.8%, additional 20 epoches training doesn't increase the model performance. BATCHSIZE is too small, hence the learning curve fluctuates too much. As a result, it is too hard to tell whether it is converged or has overfitting. But BATCHSIZE is limited by ~~~30~~~ due to the gpu memory.
	+ reading model of 18-sep-05-exp03, full validation set (30%). accuracy: 90.8%
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic
	+ validation set in learning curve: last 3% of tmall pic
	+ validation set: last 30% of tmall pic
	+ on T470p gpu
	+ 2088s, 90.8%. result in ***result/18090505/***
6. 18-sep-05-exp06. increase BATCHSIZE to 30. 
	+ gpu memory usage 74%. since BATCHSIZE=20, gpu mem usage is 60%, so BATCHSIZE should reach 50.
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=30 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic
	+ validation set in learning curve: last 3% of tmall pic
	+ validation set: last 30% of tmall pic
	+ on T470p gpu
	+ 1264s, 89.7%. result in ***result/18090506/***

TODO: BATCHSIZE = 50    

TODO: breakpoint from 18-sep-05-exp06 for additional epoches

### 18 Sep 07


1. 18-sep-07-exp01. 4 categories in total, add a new one ***other***. only 72.3%, xiaomi's accuracy is lowest. possibly due to xiaomi's product is so diversed. Furthermore, the learning curve oscillates all the time and has large amplitude. we should decrease LR and increase BATCHSIZE.
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic + garbage pic
	+ validation set in learning curve: last 2% of tmall pic + garbage pic
	+ validation set: last 30% of tmall pic + garbage pic
	+ on T470p gpu
	+ 2846s, 72.3%. result in ***result/18090701/***
2. 18-sep-07-exp02. continue from 18-sep-07-exp01, increase BATCHSIZE, reduce LR. 84.7%. "other"'s accuracy is lowest.
	+ cpu memory is only enough to calculate 2% tmall + garbage pic as validation set in learning curve.
	+ breakpoint from 18-sep-07-exp01
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=50 EPOCH=20 LR=0.0003 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic + garbage pic
	+ validation set in learning curve: last 2% of tmall pic + garbage pic
	+ validation set: last 30% of tmall pic + garbage pic
	+ on T470p gpu
	+ 1855s, 84.7%. result in ***result/18090702/***
3. 18-sep-07-exp03. continue from 18-sep-07-exp02. 86.3% (60 epoches when take account exp02 and exp01)
	+ breakpoint from 18-sep-07-exp02
	+ gpu memory 93%.
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=50 EPOCH=20 LR=0.0003 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic + garbage pic
	+ validation set in learning curve: last 2% of tmall pic + garbage pic
	+ validation set: last 30% of tmall pic + garbage pic
	+ on T470p gpu
	+ 2244s, 86.3%. result in ***result/18090703/***

smaller LR, e.g., LR = 0.0001
