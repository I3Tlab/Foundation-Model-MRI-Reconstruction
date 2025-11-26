
import numpy as np
import os
from typing import Tuple
from skimage.metrics import structural_similarity as compute_ssim



def getFilePathList(path,filetype):
    fileList = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                fileList.append(file)
    fileList.sort() 
    return fileList


def GenGaussianMask(shape: Tuple[int,int], acc: int, ACSnum: int) -> np.ndarray:
    np.random.seed(0)
    mask = np.zeros(shape)
    ctr  = shape[1]//2
    mask[:,ctr-ACSnum//2:ctr+ACSnum//2] = 1
    numLines = shape[1]//acc
    count = ACSnum
    while count < numLines:
        idx = int(np.random.randn(1)*shape[1]//4 + shape[1]//2)
        if idx < 0 or idx >= shape[1] or mask[0,idx] == 1:
            continue
        else:
            mask[:,idx] = 1
            count += 1
    return mask


def div0( a, b ):
    """ This function handles division by zero """
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c

def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==3:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    for i in range(nimg):
        img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)

def calculate_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    '''This function calculates the peak signal-to-noise ratio (PSNR)'''

    # normalize the intensity of both input and label
    pred = normalize01(pred)
    gt   = normalize01(gt)
    mse=np.sum(np.square( np.abs(gt-pred)))/gt.size
    psnr=20*np.log10(gt.max()/(np.sqrt(mse)+1e-10 ))
    return psnr

def calculate_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    '''This function calculates the structural similarity index (SSIM)'''

    # normalize the intensity of both input and label
    pred = normalize01(pred)
    gt   = normalize01(gt)
    ssim = compute_ssim(pred,gt,data_range=1,gaussian_weights=False)
    return ssim











     

    






