import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import AutoImageProcessor
from sigpy.mri.app import EspiritCalib
import numpy as np
import torch.nn.functional as F
from scipy.io import savemat
import h5py as h5
import utils
import sys
import shutil
from transformers import AutoModelForCausalLM
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from scipy.io import savemat,loadmat
from numpy import fft
from model import ChangeSize,Aclass_MC,myCG 
from model import siren_model, CNN_Adaptor
from minlora import add_lora  ## you may need to install lora for model adaptation: https://github.com/changjonathanc/minLoRA
import loss_function as lf

torch.manual_seed(3976)
torch.cuda.manual_seed(3976)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''set parameters'''
MaxIter = 3200
summary_epoch = 400
lr = 1e-4
accR  = 4

outpath = f'reconstruction_results/INR_R{accR}' 
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
data_cpl      = loadmat('demo_data.mat')['kspace'][0].transpose(1,2,0)
Nrd,Npe,Nchl  = data_cpl.shape

tstCsm  = EspiritCalib(data_cpl.transpose(2,0,1), calib_width=24).run().transpose(1,2,0)
savemat(os.path.join(outpath,'csm.mat'),{'csm':tstCsm})

mask          = utils.GenGaussianMask((Nrd,Npe),accR,24)
savemat(os.path.join(outpath,'mask.mat'),{'mask':mask})
tstDsKsp     = data_cpl*mask[:,:,np.newaxis]
DsImg_coil   = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(tstDsKsp,axes=(0,1)),axes=(0,1)),axes=(0,1))
DsImg        = np.sum(DsImg_coil*np.conj(tstCsm),axis=-1)

GtImg_coil   = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_cpl,axes=(0,1)),axes=(0,1)),axes=(0,1))
GtImg        = np.sqrt(np.sum(abs(GtImg_coil)**2,axis=-1))

max_val      = np.max(np.abs(DsImg))
DsImg        = DsImg / max_val
GtImg        = GtImg / np.max(np.abs(GtImg))
tstDsKsp     = tstDsKsp / max_val

savemat(os.path.join(outpath,'DsImg.mat'),{'DsImg':DsImg})
savemat(os.path.join(outpath,'GtImg.mat'),{'GtImg':GtImg})

#########
tstDsKsp_tensor = torch.tensor(tstDsKsp).to(torch.complex64).to(DEVICE)
tstCsm_tensor = torch.tensor(tstCsm).to(torch.complex64).to(DEVICE)
DsImg_tensor = torch.tensor(DsImg).to(torch.complex64).to(DEVICE)
GtImg = torch.tensor(GtImg).float().to(DEVICE).unsqueeze(0)
SamMask         =np.repeat(mask[:, :, np.newaxis], Nchl, axis=2).astype(int)
mask_tensor = torch.tensor(mask).float().to(DEVICE)


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


'''load SIREN model'''
# build coordinates
coor_cpu = utils.build_coordinate_train_2D(L_RD=Nrd, L_PE=Npe) 
coor = torch.from_numpy(coor_cpu).to(DEVICE).float()   


IMJENSE  = siren_model(num_layers=8, input_dim=2, hidden_dim=256, out_dim=2, w0=15).to(DEVICE)
adp      = CNN_Adaptor(in_channels=2,out_channels=2).to(DEVICE)


optimizer = torch.optim.Adam([ 
                              {'params':IMJENSE.parameters(), 'lr':lr},
                              {'params':adp.parameters(), 'lr':1e-3},
                              ])
scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

# set contrastive loss function
criterion1 = lf.ContrastiveLoss_image()
criterion2 = lf.ContrastiveLoss_image()
criterion3 = lf.ContrastiveLoss_image()

'''starting reconstruction'''
print('Reconstruction start...')
iter_loop = tqdm(range(MaxIter//2))
# we first pre-train the network using traditional total variation
for ite in iter_loop:
    IMJENSE.train()
    adp.train()

    pre_intensity_inp  = IMJENSE(coor)
    pre_intensity_adp  = adp(pre_intensity_inp.permute(2,0,1).unsqueeze(0)).squeeze(0)
    
    pre_intensity_0 = torch.complex(pre_intensity_inp[:,:,0],pre_intensity_inp[:,:,1]).unsqueeze(-1)
    pre_intensity   = torch.complex(pre_intensity_adp[0],pre_intensity_adp[1]).unsqueeze(-1)

    # fft along phase encoding
    fft_pre_intensity_0=torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(pre_intensity_0*tstCsm_tensor,dim=(0,1)),dim=(0,1)),dim=(0,1))  
    fft_pre_intensity=torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(pre_intensity*tstCsm_tensor,dim=(0,1)),dim=(0,1)),dim=(0,1))                 
        
    # compute loss  
    mae_loss_0 = lf.MAE_loss_function(torch.view_as_real(fft_pre_intensity_0[SamMask==1]).float(), torch.view_as_real(tstDsKsp_tensor[SamMask==1]).float())
    mae_loss = lf.MAE_loss_function(torch.view_as_real(fft_pre_intensity[SamMask==1]).float(), torch.view_as_real(tstDsKsp_tensor[SamMask==1]).float())
    
    TV_loss_0 = lf.compute_TV_loss(pre_intensity_0.real)+lf.compute_TV_loss(pre_intensity_0.imag)   
    TV_loss   = lf.compute_TV_loss(pre_intensity.real)+lf.compute_TV_loss(pre_intensity.imag)   
    loss = mae_loss+mae_loss_0 + 3*TV_loss_0 + 3*TV_loss

    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    # record and print loss
    iter_loop.set_postfix(loss=loss.item())
    scheduler.step()
    writer.add_scalar('Loss/Total Loss', loss, ite)
    writer.add_scalar('Loss/MAE Loss', mae_loss, ite)
    writer.add_scalar('Loss/TV Loss', TV_loss, ite)

    if (ite+1) % summary_epoch ==0:
        with torch.no_grad():
            IMJENSE.eval()
            adp.eval()
        
            # forward
            pre_intensity_inp  = IMJENSE(coor)   
            pre_intensity_adp  = adp(pre_intensity_inp.permute(2,0,1).unsqueeze(0)).squeeze(0)

            pre_intensity_0 = torch.complex(pre_intensity_inp[:,:,0],pre_intensity_inp[:,:,1]).unsqueeze(-1)
            pre_intensity = torch.complex(pre_intensity_adp[0],pre_intensity_adp[1])#.unsqueeze(-1)

            rhs = DsImg_tensor + 0.05*pre_intensity
            A = Aclass_MC(mask_tensor, tstCsm_tensor.permute(2,0,1), 0.05)
            pre_intensity_dc = myCG(A, rhs)

            pre_img_0   = pre_intensity_0.cpu().detach().numpy().reshape(Nrd,Npe)
            pre_img    = pre_intensity.cpu().detach().numpy().reshape(Nrd,Npe)
            pre_img_dc = pre_intensity_dc.cpu().detach().numpy().reshape(Nrd,Npe)

            savemat(os.path.join(outpath,f'pred_results_{ite+1}.mat'),  {'out':  pre_img,
                                                                         'out0': pre_img_0,
                                                                         'out_dc':pre_img_dc,
                                                                         })
            
            torch.save(IMJENSE.state_dict(), os.path.join(model_path, f'IMJENSE_{ite+1}.pkl'))
            torch.save(adp.state_dict(), os.path.join(model_path, f'adp_{ite+1}.pkl'))


# Then, refine it on the proposed contrastive objective
for param in IMJENSE.parameters():
    param.requires_grad = False

iter_loop = tqdm(range(MaxIter//2))
for ite in iter_loop:
    adp.train()

    pre_intensity_inp  = IMJENSE(coor).permute(2,0,1).unsqueeze(0)  
    pre_intensity_adp  = adp(pre_intensity_inp).squeeze(0)
    
    pre_intensity      = torch.complex(pre_intensity_adp[0],pre_intensity_adp[1]).unsqueeze(-1)
       
    # fft along phase encoding
    fft_pre_intensity=torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(pre_intensity*tstCsm_tensor,dim=(0,1)),dim=(0,1)),dim=(0,1))                 
        
    # compute loss     
    mae_loss = lf.MAE_loss_function(torch.view_as_real(fft_pre_intensity[SamMask==1]).float(), torch.view_as_real(tstDsKsp_tensor[SamMask==1]).float())
    

    recon_img = torch.abs(pre_intensity).squeeze(-1).unsqueeze(0)
    max_val   = torch.amax(recon_img,dim=(-1,-2),keepdim=True)
    recon_img = (recon_img / max_val - 0.5) / 0.5

    recon_img = ChangeSize(recon_img,[384,384])
       
    _,_,_,vision_features = vl_gpt.vision_model.vision_tower(recon_img.unsqueeze(1).repeat(1,3,1,1))

    con_loss =  0.01*criterion1(vision_features[0], features_FS[:,0,:,:], features_US[:,0,:,:], W1, mu1) + \
                    0.5*criterion2(vision_features[12], features_FS[:,1,:,:], features_US[:,1,:,:], W2, mu2) + \
                     1*criterion3(vision_features[23], features_FS[:,2,:,:], features_US[:,2,:,:], W3, mu3)

    loss     = mae_loss + 5*con_loss

    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    # record and print loss
    iter_loop.set_postfix(loss=loss.item())
    scheduler.step()
    writer.add_scalar('Loss/Total Loss', loss, ite)
    writer.add_scalar('Loss/MAE Loss', mae_loss, ite)
    writer.add_scalar('Loss/Contrastive Loss', con_loss, ite)

    if (ite+1) % summary_epoch ==0:
        with torch.no_grad():
            adp.eval()

        
            # forward
            pre_intensity_inp  = IMJENSE(coor).permute(2,0,1).unsqueeze(0)   #(1,256,Nrd, Npe)
            pre_intensity_adp  = adp(pre_intensity_inp).squeeze(0)
    
            pre_intensity=torch.complex(pre_intensity_adp[0],pre_intensity_adp[1])

            rhs = DsImg_tensor + 0.05*pre_intensity  
            A = Aclass_MC(mask_tensor, tstCsm_tensor.permute(2,0,1), 0.05)
            pre_intensity_dc = myCG(A, rhs)

            pre_img = pre_intensity.cpu().detach().numpy().reshape(Nrd,Npe)
            pre_img_dc = pre_intensity_dc.cpu().detach().numpy().reshape(Nrd,Npe)
           
            savemat(os.path.join(outpath,f'pred_results_{ite+1+1600}.mat'),  {'out':  pre_img,
                                                                         'out_dc':pre_img_dc,
                                                                        })
            
            torch.save(IMJENSE.state_dict(), os.path.join(model_path, f'IMJENSE_{ite+1+1600}.pkl'))
            torch.save(adp.state_dict(), os.path.join(model_path, f'adp_{ite+1+1600}.pkl'))
            

