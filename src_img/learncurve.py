import matplotlib.pyplot as plt
import numpy as np




def smoothArray(array, window):
    '''
    array, 1 dim np array
    window, smooth array with the size of window by naive arthemic average
    out[i] = \sum_{d = -window/2}^{+window/2} array[i+d] / window
    '''
    out = np.array([])
    length = array.size
    for i, elem in enumerate(array):
        if i < window:
            out = np.append(out, elem)
        elif i > array.size - window:
            out = np.append(out, elem)
        else:
            tmp = np.mean(array[i-window:i+window])
            out = np.append(out, tmp)
    return out
                



# def learningCurve(loss, acc1, acc5, window):
def learningCurve(loss, acc1, window = 5):
    '''
    loss: 1 dim np array. loss function
    acc1: 1 dim np array. top 1 accuracy 
    acc5: 1 dim np array. top 5 accuracy 
    window: window size for smoothing
    '''
    
    # raw learning curve
    plt.xlabel("every 50 mini-batches")
    plt.ylabel("loss and accuarcy")
    plt.plot(np.arange(loss.shape[0]), loss, label="loss of training set" )
    plt.plot(np.arange(loss.shape[0]), acc1, label = "TOP 1 accuarcy of validation set")
    # plt.plot(np.arange(loss.shape[0]), acc5, label = "TOP 5 accuarcy of validation set")
    plt.legend()
    # imgpath = os.path.join(path, "learncurve.eps")
    imgpath = "learncurve.eps"
    plt.savefig(imgpath, format='eps', dpi = 100, bbox_inches="tight")
    
    # smooth learning curve
    lossave = smoothArray(loss, window)
    accuarcy1ave = smoothArray(acc1, window)
    # accuarcy5ave = smoothArray(acc5, window)
    plt.xlabel("every 50 mini-batches")
    plt.ylabel("loss and accuarcy")
    plt.plot(np.arange(loss.shape[0]), lossave, label="smooth loss of training set" )
    plt.plot(np.arange(loss.shape[0]), accuarcy1ave, label = "smooth TOP 1 accuarcy of validation set %d"%np.argmax(accuarcy1ave))
    # plt.plot(np.arange(loss.shape[0]), accuarcy5ave, label = "smooth TOP 5 accuarcy of validation set %d"%np.argmax(accuarcy5ave))
    plt.legend()
    # imgpath = os.path.join(path, "learncurveaverage.eps")
    imgpath = "smooth_learncurve.eps"
    plt.savefig(imgpath, format='eps', dpi = 100, bbox_inches="tight")                                      

