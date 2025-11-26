import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from unet import Pure_UNet
import numpy as np
import torch.nn.functional as F
from scipy.io import savemat
import h5py as h5
from sigpy.mri.app import EspiritCalib
import utils
import sys
import shutil
from model import Unet_CG
from model import ChangeSize
from transformers import AutoModelForCausalLM
from Janus.janus.models import VLChatProcessor
from scipy.io import savemat,loadmat
from minlora import add_lora   # you may need to install lora for model adaptation: https://github.com/changjonathanc/minLoRA
import loss_function as lf


DEVICE     = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3976)
torch.cuda.manual_seed(3976)


'''set parameters'''
epochs = 1500
summary_epoch = 500
lr = 1e-3
niters = 4
accR  = 4


outpath = f'reconstruction_results/Unroll_R{accR}'  
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
mask_dc,mask_loss  = utils.sample_points_from_mask(mask, std_scale=0.5,proportion=0.4)
savemat(os.path.join(outpath,'mask.mat'),{'mask':mask,'mask_dc':mask_dc,'mask_loss':mask_loss})

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

tstDsKsp_dc   = tstDsKsp*mask_dc[np.newaxis,np.newaxis,:,:]
DsImg_dc_coil = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(tstDsKsp_dc,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
DsImg_dc     = np.sum(DsImg_dc_coil*np.conj(csm),axis=1)

GtImg_coil   = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_cpl,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
GtImg        = np.sqrt(np.sum(abs(GtImg_coil)**2,axis=1))


max_val      = np.max(np.abs(DsImg))
DsImg        = DsImg / max_val
GtImg        = GtImg / np.max(np.abs(GtImg))
DsImg_dc    = DsImg_dc / max_val
tstDsKsp     = tstDsKsp / max_val
tstDsKsp_dc  = tstDsKsp_dc / max_val
savemat(os.path.join(outpath,'DsImg.mat'),{'DsImg':DsImg.squeeze(0)})
savemat(os.path.join(outpath,'GtImg.mat'),{'GtImg':GtImg.squeeze(0)})
savemat(os.path.join(outpath,'DsImg_coil.mat'),{'DsImg_coil':DsImg_coil.squeeze(0).transpose(1,2,0)})

DsImg          = torch.tensor(DsImg).to(torch.complex64).to(DEVICE)
GtImg          = torch.tensor(GtImg).float().to(DEVICE)
DsImg_dc       = torch.tensor(DsImg_dc).to(torch.complex64).to(DEVICE)
tstDsKsp       = torch.tensor(tstDsKsp).to(torch.complex64).to(DEVICE)
mask           = torch.tensor(mask).float().to(DEVICE)
mask_dc        = torch.tensor(mask_dc).float().to(DEVICE)
mask_loss      = torch.tensor(mask_loss).float().to(DEVICE)
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

# first do PCA for these image embeddings
W1, mu1 = utils.fit_pca(features_FS[:,0,:,:], features_US[:,0,:,:], out_dim=64)
W1      = W1.to(DEVICE)
mu1     = mu1.to(DEVICE)

W2, mu2 = utils.fit_pca(features_FS[:,1,:,:], features_US[:,1,:,:], out_dim=128)
W2      = W2.to(DEVICE)
mu2     = mu2.to(DEVICE)

W3, mu3 = utils.fit_pca(features_FS[:,2,:,:], features_US[:,2,:,:], out_dim=512)
W3      = W3.to(DEVICE)
mu3     = mu3.to(DEVICE)


'''load Unrolled model'''
Unet_model = []
lam        = []
pretrained_path = "./unet/pre_trained_weights"    # This is the pre-trained weight on the 20 auxiliary images used for embedding extraction
stats           = loadmat(os.path.join(pretrained_path,'stats.mat'))
input_mean_val  = torch.tensor(stats['input_mean_val']).float().to(DEVICE)
input_std_val   = torch.tensor(stats['input_std_val']).float().to(DEVICE)
label_mean_val  = torch.tensor(stats['label_mean_val']).float().to(DEVICE)
label_std_val   = torch.tensor(stats['label_std_val']).float().to(DEVICE)

for i in range(niters):
    punet = Pure_UNet(in_channels=2, out_channels=2, bilinear=True)

    state_dict = torch.load(os.path.join(pretrained_path,'model','model.pkl'))
    punet.load_state_dict(state_dict, strict=True)

    for param in punet.parameters():
        param.requires_grad = False
    
    add_lora(punet)
    punet = punet.to(DEVICE)
    Unet_model.append(punet)

    lam.append(torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float32, device=DEVICE)))

CG_model = Unet_CG(unet=Unet_model, lam=lam, niter=niters,input_mean_val=input_mean_val, input_std_val=input_std_val, label_mean_val=label_mean_val, label_std_val=label_std_val)

#define optimizer
params = []

for punet in Unet_model:
    params += list(punet.parameters())

params += lam

optimizer = torch.optim.Adam([{'params':params, 'lr':1e-3}])
scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

# define loss function
criterion1 = lf.ContrastiveLoss_image()
criterion2 = lf.ContrastiveLoss_image()
criterion3 = lf.ContrastiveLoss_image()

'''starting reconstruction'''
print('Reconstruction start...')
iter_loop = tqdm(range(epochs))

for e in iter_loop:
    CG_model.train()

    ############################################################
    out, pred_x_list = CG_model(DsImg, mask_dc, csm, DsImg_dc)

    MAE_loss = lf.compute_cost_ksp(out, tstDsKsp, mask_loss, csm) + lf.compute_cost_ksp(pred_x_list, tstDsKsp, mask_loss, csm)  

    # process to fit the range and size of image encoder's input
    recon_img = torch.abs(pred_x_list[-1])   
    max_val   = torch.amax(recon_img,dim=(-1,-2),keepdim=True)
    recon_img = (recon_img / max_val - 0.5) / 0.5
    recon_img = ChangeSize(recon_img,[384,384])
    # map the reconstructed image into the same semantic space as the prior embedding
    _,_,_,vision_features = vl_gpt.vision_model.vision_tower(recon_img.unsqueeze(1).repeat(1,3,1,1))
    
    # contrastive loss at three levels
    con_loss0  = criterion1(vision_features[0], features_FS[:,0,:,:], features_US[:,0,:,:], W1, mu1)  
    con_loss1  = criterion2(vision_features[12], features_FS[:,1,:,:],features_US[:,1,:,:], W2, mu2) 
    con_loss2  = criterion3(vision_features[23], features_FS[:,2,:,:],features_US[:,2,:,:], W3, mu3) 
    con_loss =  0.01*con_loss0 + 0.5*con_loss1 + 1*con_loss2 
        
    loss = MAE_loss + con_loss 
    writer.add_scalar('Loss/Contrastive Loss_l0', con_loss0, e)
    writer.add_scalar('Loss/Contrastive Loss_l1', con_loss1, e)
    writer.add_scalar('Loss/Contrastive Loss_l2', con_loss2, e)
    writer.add_scalar('Loss/Total Loss', loss, e)
    writer.add_scalar('Loss/MAE Loss', MAE_loss, e)
    writer.add_scalar('Loss/MAE Loss', con_loss, e)
           
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    iter_loop.set_postfix(loss='{:.4f}'.format(loss.item()))

    
    if e==0 or (e+1)%summary_epoch == 0:
        with torch.no_grad():
            savemat(os.path.join(outpath,f'pred_results_{e+1}.mat'),  {'out':  torch.stack(out,-1).squeeze(0).cpu().detach().numpy(),
                                                                      'pred_x':torch.stack(pred_x_list,-1).squeeze(0).cpu().detach().numpy(),
                                                                   
                                                          
                                                                     })
            

            torch.save(CG_model.state_dict(), os.path.join(model_path, f'CG_model_{e+1}.pkl'))

