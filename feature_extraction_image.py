'''
This demo shows how to use the pre-trained Junas model to extract high-level embeddings from the auxiliary images. 
The resulting embeddings serve as the prior distribution to guide MRI reconstruction
'''

import os
import torch
import numpy as np
import h5py as h5
import sys
import shutil
from transformers import AutoModelForCausalLM
from Janus.janus.models import VLChatProcessor
import utils
from scipy.io import savemat
import random
import umap
import matplotlib.pyplot as plt

torch.manual_seed(3976)
torch.cuda.manual_seed(3976)
DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

outpath = './prior_embeddings/fastMRI_knee'
if not os.path.exists(outpath):
    os.makedirs(outpath)

try:
    current_file = os.path.abspath(sys.argv[0])
    shutil.copy2(current_file, outpath)
    print(f"file saved: {outpath}")
except Exception as e:
    print(f"save filed: {str(e)}")


'''load the pre-trained Junas model for embedding extraction'''
# this is the weight path of Junas model, which can be downloaded from here: https://huggingface.co/deepseek-ai/Janus-Pro-1B/tree/main
foundation_model_path = "/local_mount/space/cookie2/1/users/rf552/code/foundation_model_self_supervise/Janus_weights/Janus-Pro-1B"  # change to your folder
vl_chat_processor     = VLChatProcessor.from_pretrained(foundation_model_path)
vl_gpt                = AutoModelForCausalLM.from_pretrained(foundation_model_path, trust_remote_code=True)
vl_gpt                = vl_gpt.float().to(DEVICE).eval()

'''load fastMRI data'''
# The fastMRI data can be downloaded from here: https://fastmri.med.nyu.edu/
data_path = '/local_mount/space/cookie2/1/users/xh095/MRI_foundation_model/dataset/FastMRI_knee/k_space/multicoil_val'  # change to your folder


'''extract embeddings'''
sublist    = utils.getFilePathList(data_path,'.h5')
FS_feature_vis = []
DS_feature_vis = []
Nsample = 0


for ss in range(len(sublist)):
    print('loading {} sub: {}......'.format(ss,sublist[ss]))
    fpath         = data_path + '/' + sublist[ss]
    f             = h5.File(fpath,'r')

    if 'FS' not in f.attrs['acquisition']:   # Here we consider the auxiliary images with the same contrast as the reconstructed image
        continue
   
    data_cpl      = f['kspace'][:]
    Nsli,Nchl,Nrd,Npe = data_cpl.shape

    # randomly generate acceleration rates to enhance the variations of negative samples
    Accr = random.randint(3, 8)
    mask          = utils.GenGaussianMask((Nrd,Npe),Accr,24)


    tstDsKsp     = data_cpl*mask[np.newaxis,np.newaxis,:,:]
    DsImg_coil   = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(tstDsKsp,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
    DsImg        = np.sqrt(np.sum(np.abs(DsImg_coil)**2,axis=1))

    GtImg_coil   = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_cpl,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
    GtImg        = np.sqrt(np.sum(np.abs(GtImg_coil)**2,axis=1))


    # normalization and change the image size to fit the Janus model input
    # The image intensity should be in the range of [-1,1]
    max_val      = np.max(np.abs(DsImg),axis=(-1,-2),keepdims=True)
    DsImg        = DsImg / max_val
    max_val      = np.max(np.abs(GtImg),axis=(-1,-2),keepdims=True)
    GtImg        = GtImg / max_val

    DsImg        = (DsImg - 0.5) / 0.5    
    GtImg        = (GtImg - 0.5) / 0.5
    # The image size should be [384,384]
    cut_size = (Nrd-384)//2
    DsImg = DsImg[:,cut_size:cut_size+384,:]
    GtImg = GtImg[:,cut_size:cut_size+384,:]

    if Npe<384:
        pad_size = (384-Npe) // 2
        DsImg    = np.pad(DsImg,((0,0),(0,0),(pad_size,384-pad_size-Npe)),'constant')
        GtImg    = np.pad(GtImg,((0,0),(0,0),(pad_size,384-pad_size-Npe)),'constant')   
    if Npe>384:
        cut_size = (Npe-384) // 2    
        DsImg    = DsImg[:,:,cut_size:cut_size+384]
        GtImg    = GtImg[:,:,cut_size:cut_size+384]

    DsImg_inp      = torch.tensor(DsImg).float().to(DEVICE)
    GtImg_inp      = torch.tensor(GtImg).float().to(DEVICE)
    
    # the pre-processed images are input to the image encoder of the Janus model
    with torch.no_grad():
        _,_,_, DsImg_features    = vl_gpt.vision_model.vision_tower(DsImg_inp.unsqueeze(1).repeat(1,3,1,1))   
        _,_,_, GtImg_features    = vl_gpt.vision_model.vision_tower(GtImg_inp.unsqueeze(1).repeat(1,3,1,1))

    DsImg_features_cpu  = torch.stack((DsImg_features[0],DsImg_features[12],DsImg_features[23]),dim=0).cpu().detach().numpy()
    GtImg_features_cpu  = torch.stack((GtImg_features[0],GtImg_features[12],GtImg_features[23]),dim=0).cpu().detach().numpy()

    FS_feature_vis.append(GtImg_features_cpu)
    DS_feature_vis.append(DsImg_features_cpu)

    savemat(os.path.join(outpath,sublist[ss].split('.')[0] + '.mat'),{'fully_sample_feat':GtImg_features_cpu,'under_sample_feat':DsImg_features_cpu})
    del DsImg, GtImg, DsImg_features, GtImg_features

    Nsample += 1
    if Nsample > 20:  # We used 20 subjects for knee dataset and 50 subjects for brain dataset in the manuscript
        break


'''embedding visualization using UMAP'''
def VecNormalization(vector):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    vector_norm = vector / (norm + 1e-8)
    return vector_norm

FS_feature_vis = np.concatenate(FS_feature_vis,axis=1)
DS_feature_vis = np.concatenate(DS_feature_vis,axis=1)

title_list = ['Low level','Middle level','High level']

# Store UMAP results for each level
umap_results = []

for level in range(3):
    features_FS = FS_feature_vis[level]
    features_US = DS_feature_vis[level]
    features_FS = VecNormalization(features_FS).mean(1) 
    features_US = VecNormalization(features_US).mean(1)
    all_features = np.vstack([features_FS,features_US]) 
    labels = np.array([0]*len(features_FS) + [1]*len(features_US) )

    umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    features_2d  = umap_model.fit_transform(all_features)   

    FS_2d        = features_2d[labels==0]
    DS_2d        = features_2d[labels==1]
    
    umap_results.append((FS_2d, DS_2d))

# Create a figure with subplots for each level
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for level in range(3):
    FS_2d, DS_2d = umap_results[level]
    
    # Plot on the corresponding subplot
    axes[level].scatter(FS_2d[:, 0], FS_2d[:, 1], c='orange', alpha=0.6, s=10, label='Fully Sampled')
    axes[level].scatter(DS_2d[:, 0], DS_2d[:, 1], c='blue', alpha=0.6, s=10, label='Under Sampled')
    axes[level].set_title(title_list[level], fontsize=12)
    axes[level].set_xlabel('Dim 1', fontsize=10)
    axes[level].set_ylabel('Dim 2', fontsize=10)
    axes[level].legend(fontsize=8)
    axes[level].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(outpath, 'umap_visualization_all_levels.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"UMAP visualization saved to {os.path.join(outpath, 'umap_visualization_all_levels.png')}")













