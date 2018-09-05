import scipy.misc
from PIL import Image
import numpy as np



def readImage(imgPath):
    '''
    imgPath: directory of image
    output: np array
    '''
    image = Image.open(imgPath)
    image.load()
    image = np.asarray(image, dtype="uint8")
    return image


def resize(img, size):
    '''
    resize an image to size * size
    img: an np array. (w_in, w_in, channel)
    size: an int
    output: an np array. (size, size, channel)
    '''
    return scipy.misc.imresize(img, (size, size))
    # return transform.resize(img, (size, size))



def normalise(img):
    '''
    normalise an image from [0, 255] to [0, 1]
    linear squash, most naive normalisation.
    doesn't take variance into account.
    img: np array (w, w, channel), each pixel ranges from 0 to 255
    output: np array (w, w, channel), each pixel ranges from 0 to 1
    '''
    return img/255.0



def colorise(img):
    '''
    convert black/white (2-dim) img to colorful one (3-dim).
    '''
    d = len(img.shape)
    w, h = img.shape
    if d == 2:
        return np.repeat(img, 3).reshape(w, h, 3)
    else:
        return img



def dumpdim(img):
    '''
    if there is no data augmentation operation, one has to add a dump dimesion to img.
    img: np array [s, s, 3]
    return: np array [1, s, s, 3]
    '''
    w, h = img.shape[0:2]
    res = np.empty((1, w, h, 3))
    res[0] = img
    return res



def rmAlpha(img):
    '''
    remove alpha channel in img.
    img: np array [s, s, 3 or 4]
    return: np array [s, s, 3]
    '''
    if img.shape[2] == 4:
        return img[:,:,0:3]
    else:
        return img



def preprocess(img, size):
    '''
    current version includes:
    1. resize
    2. normalise
    3. colorise
    4. remove alpha
    5. dumpdim
    future: augmentation, noise embedding
    '''
    # resize, normalise
    imgTmp = normalise(resize(img, size))
    # colorise
    if len(imgTmp.shape) == 2:
        imgTmp = colorise(imgTmp)
    # remove alpha
    imgTmp2 = rmAlpha(imgTmp)
    # dump dim
    return dumpdim(imgTmp2)

