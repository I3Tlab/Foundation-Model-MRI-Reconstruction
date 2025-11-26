import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AutoImageProcessor, SwinModel
from unet import Pure_UNet
import numpy as np
import torch.nn.functional as F
from scipy.io import savemat
import h5py as h5
from sigpy.mri.app import EspiritCalib
import utils
from datetime import datetime
import sys
import shutil
import argparse
from collections import defaultdict
from model import ChangeSize,Aclass_MC,myCG
from transformers import AutoModelForCausalLM
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from scipy.io import savemat,loadmat
import matplotlib.pyplot as plt
from minlora import add_lora  ## you may need to install lora for model adaptation: https://github.com/changjonathanc/minLoRA
import loss_function as lf


DEVICE     = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3976)
torch.cuda.manual_seed(3976)

'''set parameters'''
epochs = 1500
summary_epoch = 500
lr = 1e-3
accR  = 4

outpath = f'reconstruction_results/Unet_R{accR}' 
if not os.path.exists(outpath):
    os.makedirs(outpath)

model_path = os.path.join(outpath,'model')
log_path   = os.path.join(outpath,'log')
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path): 
    os.makedirs(log_path)

try:
    current_file = os.path.abspath(sys.argv[0])
    shutil.copy2(current_file, outpath)
    print(f"file saved: {outpath}")
except Exception as e:
    print(f"save filed: {str(e)}")

# initialize the SummaryWriter
writer = SummaryWriter(log_path)

'''load the demo data and process'''
data_cpl      = loadmat('demo_data.mat')['kspace'][:]
Nsli,Nchl,Nrd,Npe  = data_cpl.shape

mask               = utils.GenGaussianMask((Nrd,Npe),accR,24)
savemat(os.path.join(outpath,'mask.mat'),{'mask':mask})

csm           = np.zeros_like(data_cpl)
for sli in range(Nsli):
    print(sli)
    csm[sli]  = EspiritCalib(data_cpl[sli], calib_width=24).run()
savemat(os.path.join(outpath,'csm.mat'),{'csm':csm.transpose(2,3,1,0)})
tissue_mask = np.ones((Nsli,Nrd,Npe))
temp        = csm.sum(axis=1)
tissue_mask[temp==0] = 0
savemat(os.path.join(outpath,'tissue_mask.mat'),{'tissue_mask':tissue_mask.transpose(1,2,0)})

tstDsKsp     = data_cpl*mask[np.newaxis,np.newaxis,:,:]
DsImg_coil   = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(tstDsKsp,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
DsImg        = np.sum(DsImg_coil*np.conj(csm),axis=1)

GtImg_coil   = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_cpl,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
GtImg        = np.sum(np.abs(GtImg_coil)**2,axis=1)**0.5

max_val      = np.max(np.abs(DsImg))
DsImg        = DsImg / max_val
GtImg        = GtImg / np.max(np.abs(GtImg))
tstDsKsp     = tstDsKsp / max_val
savemat(os.path.join(outpath,'DsImg.mat'),{'DsImg':DsImg.squeeze(0)})
savemat(os.path.join(outpath,'GtImg.mat'),{'GtImg':GtImg.squeeze(0)})
savemat(os.path.join(outpath,'DsImg_coil.mat'),{'DsImg_coil':DsImg_coil.squeeze(0).transpose(1,2,0)})

#########
DsImg          = torch.tensor(DsImg).to(torch.complex64).to(DEVICE)
GtImg          = torch.tensor(GtImg).to(torch.complex64).to(DEVICE)
tstDsKsp       = torch.tensor(tstDsKsp).to(torch.complex64).to(DEVICE)
mask           = torch.tensor(mask).float().to(DEVICE)
tissue_mask    = torch.tensor(tissue_mask).float().to(DEVICE)
csm            = torch.tensor(csm).to(torch.complex64).to(DEVICE)


'''load foundation model and prior embeddings'''
# this is the weight path of Junas model, which can be downloaded from here: https://huggingface.co/deepseek-ai/Janus-Pro-1B/tree/main
# Should be changed to your folder
foundation_model_path = "/local_mount/space/cookie2/1/users/rf552/code/foundation_model_self_supervise/Janus_weights/Janus-Pro-1B"
vl_chat_processor     = VLChatProcessor.from_pretrained(foundation_model_path)
vl_gpt                = AutoModelForCausalLM.from_pretrained(foundation_model_path, trust_remote_code=True)
vl_gpt                = vl_gpt.float().to(DEVICE).eval()

# Load the prior image embeddings, which can be generated using "feature_extraction_image.py"
# Should be changed to your path of prior embeddings
feat_path = '/local_mount/space/cookie2/1/users/rf552/code/foundation_model_self_supervise/feature_collection/FastMRI_knee_FS_3levels'
sublist   = utils.getFilePathList(feat_path,'.mat')
features_FS = []
features_US = []

for ss in np.arange(0,20):  
    print('loading {} sub: {}......'.format(ss,sublist[ss]))

    fpath = feat_path + '/' + sublist[ss]
    f_FS  = loadmat(fpath)['fully_sample_feat']
    f_US  = loadmat(fpath)['under_sample_feat']
    for hhh in range(f_FS.shape[1]):
        features_FS.append(f_FS[:,hhh,:,:])
        features_US.append(f_US[:,hhh,:,:])

features_FS = torch.tensor(np.array(features_FS)).float().to(DEVICE)
features_US = torch.tensor(np.array(features_US)).float().to(DEVICE)

W1, mu1 = utils.fit_pca(features_FS[:,0,:,:], features_US[:,0,:,:], out_dim=64)
W1      = W1.to(DEVICE)
mu1     = mu1.to(DEVICE)

W2, mu2 = utils.fit_pca(features_FS[:,1,:,:], features_US[:,1,:,:], out_dim=128)
W2      = W2.to(DEVICE)
mu2     = mu2.to(DEVICE)

W3, mu3 = utils.fit_pca(features_FS[:,2,:,:], features_US[:,2,:,:], out_dim=256)
W3      = W3.to(DEVICE)
mu3     = mu3.to(DEVICE)


'''load U-Net'''
pretrained_path = "./unet/pre_trained_weights"    # This is the pre-trained weight on the 20 auxiliary images used for embedding extraction
stats           = loadmat(os.path.join(pretrained_path,'stats.mat'))
input_mean_val  = torch.tensor(stats['input_mean_val']).float().to(DEVICE)
input_std_val   = torch.tensor(stats['input_std_val']).float().to(DEVICE)
label_mean_val  = torch.tensor(stats['label_mean_val']).float().to(DEVICE)
label_std_val   = torch.tensor(stats['label_std_val']).float().to(DEVICE)

Unet_model = Pure_UNet(in_channels=2, out_channels=2, bilinear=True)#.to(DEVICE)

state_dict = torch.load(os.path.join(pretrained_path,'model','model.pkl'), map_location=DEVICE)
Unet_model.load_state_dict(state_dict, strict=True)

for param in Unet_model.parameters():
    param.requires_grad = False
    
add_lora(Unet_model)
Unet_model = Unet_model.to(DEVICE)

optimizer = torch.optim.Adam([{'params':Unet_model.parameters(), 'lr':1e-3}])
scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

# define loss function
criterion1 = lf.ContrastiveLoss_image()
criterion2 = lf.ContrastiveLoss_image()
criterion3 = lf.ContrastiveLoss_image()


'''starting reconstruction'''
print('Reconstruction start...')
iter_loop = tqdm(range(epochs))

DsImg_inp = ChangeSize(DsImg,[384,384])
DsImg_inp = torch.stack([DsImg_inp.real, DsImg_inp.imag],dim=1).float()
DsImg_inp = (DsImg_inp - input_mean_val) / input_std_val

for e in iter_loop:
    Unet_model.train()

    ############################################################
    out = Unet_model(DsImg_inp)
    out = out*label_std_val + label_mean_val
    out_cplx = torch.complex(out[:,0,:,:], out[:,1,:,:])
    out_cplx = ChangeSize(out_cplx,[Nrd,Npe])
 
    MAE_loss = lf.compute_cost_ksp([out_cplx], tstDsKsp, mask, csm) 

    # process to fit the range and size of image encoder's input
    recon_img = torch.abs(out_cplx)
    max_val   = torch.amax(torch.abs(recon_img),dim=(-1,-2),keepdim=True)
    recon_img = (recon_img / max_val - 0.5) / 0.5
    recon_img = ChangeSize(recon_img,[384,384])

    # map the reconstructed image into the same semantic space as the prior embedding
    _,_,_,vision_features = vl_gpt.vision_model.vision_tower(recon_img.unsqueeze(1).repeat(1,3,1,1))  
  
    # contrastive loss at three levels
    con_loss0  = criterion1(vision_features[0], features_FS[:,0,:,:], features_US[:,0,:,:], W1, mu1)
    con_loss1  = criterion2(vision_features[12], features_FS[:,1,:,:], features_US[:,1,:,:], W2, mu2)
    con_loss2  = criterion3(vision_features[23], features_FS[:,2,:,:], features_US[:,2,:,:], W3, mu3)
    con_loss = 0.01*con_loss0 + 0.5*con_loss1 + 1*con_loss2 
        
    loss = MAE_loss + 0.1*con_loss
    writer.add_scalar('Loss/Contrastive Loss', con_loss, e)
    writer.add_scalar('Loss/Total Loss', loss, e)
    writer.add_scalar('Loss/MAE Loss', MAE_loss, e)
     
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    iter_loop.set_postfix(loss='{:.4f}'.format(loss.item()))

    if e==0 or (e+1)%summary_epoch == 0:
        out1 = Unet_model(DsImg_inp)
        out1 = out1 * label_std_val + label_mean_val
        out1 = torch.complex(out1[:,0,:,:], out1[:,1,:,:])
        out1 = ChangeSize(out1,[Nrd,Npe])
        
        rhs = DsImg[0] + 0.5*out1[0]
        A = Aclass_MC(mask, csm[0], 0.5)
        pre_intensity_dc = myCG(A, rhs)

        savemat(os.path.join(outpath,f'pred_results_{e+1}.mat'),  {'out':  out1.permute(1,2,0).cpu().detach().numpy(),
                                                                          'recon_img_sos':pre_intensity_dc.cpu().detach().numpy(),
                                                                       
                                                                         })

        torch.save(Unet_model.state_dict(), os.path.join(model_path, f'Unet_model_{e+1}.pkl'))