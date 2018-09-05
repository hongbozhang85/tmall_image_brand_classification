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

10. 18-sep-05-exp01. increase size of training set
	+ alexnet, with BN, LRN and drop out
	+ BATCHSIZE=20 EPOCH=20 LR=0.001 WD=1e-6 ISGPU=True NH=128 DP=0.5 SZ=224      
	+ no data augmentation and no noise embedding
	+ training set: first 70% of tmall pic
	+ validation set: last 3% of tmall pic
	+ on T470p gpu
	+ s, %. result in ***result//***
