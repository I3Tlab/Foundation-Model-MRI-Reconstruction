import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple
from skimage.metrics import structural_similarity as compute_ssim
from sklearn.decomposition import PCA



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

def sample_points_from_mask(mask, std_scale=1, proportion=0.4):
    """
    Randomly sample a certain proportion of points from each mask using PyTorch.
    
    :param masks: PyTorch tensor of shape (8, H, W) containing 8 binary masks
    :param proportion: Float, fraction of available points to sample per mask
    :return: List of sampled points for each mask as PyTorch tensors
    """

    H, W = mask.shape
    y    = np.linspace(-1,1,H)
    x    = np.linspace(-1,1,W)
    Y, X = np.meshgrid(y, x, indexing = 'ij')

    # Create 2D Gaussian weight map centered in the middle of the mask
    sigma = std_scale
    gaussian_weights = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Normalize weights only over valid points
    prob_weights = gaussian_weights * mask
    flat_mask    = mask.flatten()
    flat_weights = prob_weights.flatten()
    valid_indices = np.where(flat_mask == 1)[0]
    valid_weights = flat_weights[valid_indices]

    prob = valid_weights / valid_weights.sum()
 
    # Flatten indices of valid points
    num_to_sample = int(len(valid_indices) * proportion)

    # Sample from the valid indices using the Gaussian-weighted probability
    sampled_indices = np.random.choice(valid_indices, size=num_to_sample, replace=False, p=prob)

    # Create output mask
    mask_loss = np.zeros_like(mask.flatten())
    mask_loss[sampled_indices] = 1
    mask_loss = mask_loss.reshape(H, W)

    mask_dc = mask - mask_loss

    return mask_dc, mask_loss

def fit_pca(positive_features, negative_features, out_dim=128):
    pos = positive_features.reshape(-1, positive_features.shape[-1])
    neg = negative_features.reshape(-1, negative_features.shape[-1])
    all_feats = torch.cat([pos, neg], dim=0).cpu().numpy()

    pca = PCA(n_components=out_dim)
    pca.fit(all_feats)

    # 
    W = torch.tensor(pca.components_.T, dtype=torch.float32)  # (1024, out_dim)
    mu = torch.tensor(pca.mean_, dtype=torch.float32)         # (1024,)
    return W, mu

def ChangeSize(x, size):
    ''' change the size of the input to fit the ViT input size '''
    M, N     = x.shape[-2], x.shape[-1]
    tS1, tS2 = size[0], size[1]

    if M < tS1:
        pad_size = (tS1 - M)//2
        x = F.pad(x, (0, 0, pad_size, pad_size), "constant", 0)
    if M > tS1:
        cut_size = (M - tS1)//2
        x = x[:,cut_size:cut_size+tS1,:]
    if N < tS2:
        pad_size = (tS2 - N)//2
        x = F.pad(x, (pad_size, pad_size), "constant", 0)
    if N > tS2:
        cut_size = (N - tS2)//2
        x = x[:,:,cut_size:cut_size+tS2]

    return x

def build_coordinate_train_2D(L_RD, L_PE):
    x = np.linspace(-1, 1, L_RD)              #*********
    y = np.linspace(-1, 1, L_PE)           #*********
    x, y = np.meshgrid(x, y, indexing='ij')  # (L, L), (L, L), (L, L)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (L*L*L, 3)
    xy = xy.reshape(L_RD, L_PE, 2)
    return xy


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











     

    






