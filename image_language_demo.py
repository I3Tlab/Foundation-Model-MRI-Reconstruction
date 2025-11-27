'''This file shows how to use the image-language embeddings to guide reconstruction process'''
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
from model import ChangeSize
from model import Unet_CG
from transformers import AutoModelForCausalLM
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from scipy.io import savemat,loadmat
import matplotlib.pyplot as plt
from minlora import add_lora
from einops import rearrange
from loss_function import compute_cost_ksp, ContrastiveLoss_language


torch.manual_seed(3976)
torch.cuda.manual_seed(3976)
DEVICE     = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

'''set parameters'''
epochs = 200
summary_epoch = 200
lr = 1e-3
niters = 8
accR  = 4

outpath = f'reconstruction_results/Image_language_R{accR}' 

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
#########
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
vl_gpt                = vl_gpt.to(torch.bfloat16).to(DEVICE).eval()
for param in vl_gpt.parameters():
    param.requires_grad = False
tokenizer             = vl_chat_processor.tokenizer

prompt = 'Determine whether this image is high-quality or low-quality. Respond with single word.'

# "./image_example/img.jpg" is just a placeholder example, included so we can
# leverage Janus's built-in pipeline for image preprocessing and formatting.
conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [f"./image_examples/img.jpg"],
        },
        {"role": "Assistant", "content": ""},
    ]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device) 
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
images_seq_mask = (prepare_inputs.images_seq_mask)
images_emb_mask  = rearrange(prepare_inputs.images_emb_mask, "b n t -> b (n t)")


'''Calculate features for undersampled data'''
recon_img = torch.abs(DsImg)   
max_val   = torch.amax(recon_img,dim=(-1,-2),keepdim=True)
recon_img = recon_img / max_val 
recon_img = (recon_img -0.5) / 0.5
recon_img = ChangeSize(recon_img,[384,384])
vision_features =  vl_gpt.aligner(vl_gpt.vision_model(recon_img.to(torch.bfloat16).unsqueeze(1).repeat(1,3,1,1)))
# get image language embeddings
new_inputs_embeds = inputs_embeds.detach().clone()
new_inputs_embeds[images_seq_mask] = vision_features[images_emb_mask]
outputs = vl_gpt.language_model(
        inputs_embeds=new_inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        output_hidden_states=True,
    )
answer_embeds   = outputs.hidden_states[-1]
savemat(os.path.join(outpath,'ds_feature.mat'),{'ds_feature':answer_embeds.float().cpu().detach().numpy()})

'''load pre-generated image-language embeddings'''
feat_path = f'./prior_embeddings_image_language/high_quality_vs_low_quality'
pos_features = h5.File(os.path.join(feat_path,'image_language_feat.h5'),'r')['positive_features'][:]
neg_features = h5.File(os.path.join(feat_path,'image_language_feat.h5'),'r')['negative_features'][:]

neg_features = torch.tensor(neg_features).float().to(DEVICE)
pos_features = torch.tensor(pos_features).float().to(DEVICE)


'''load Unrolled model'''
Unet_model = []
lam        = []
input_mean_val  = torch.tensor(0.0).float().to(DEVICE)
input_std_val   = torch.tensor(0.1).float().to(DEVICE)
label_mean_val  = torch.tensor(0.0).float().to(DEVICE)
label_std_val   = torch.tensor(0.1).float().to(DEVICE)
for i in range(niters):
    punet = Pure_UNet(in_channels=2, out_channels=2, bilinear=True).to(DEVICE)
    Unet_model.append(punet)
    lam.append(torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float32, device=DEVICE)))
CG_model = Unet_CG(unet=Unet_model, lam=lam, niter=niters,input_mean_val=input_mean_val, input_std_val=input_std_val, label_mean_val=label_mean_val, label_std_val=label_std_val)

#define optimizer
params = []
for punet in Unet_model:
    params += list(punet.parameters())
params += lam
optimizer = torch.optim.Adam([{'params':params, 'lr':1e-3}])
scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

'''starting reconstruction'''
print('Reconstruction start...')
iter_loop = tqdm(range(epochs))
loss_train = 0.0
answer_embeds_list = []
images_seq_mask = images_seq_mask.repeat(niters,1)
images_emb_mask = images_emb_mask.repeat(niters,1)
images_seq_mask2 = (images_seq_mask).unsqueeze(-1).float()
for e in iter_loop:
    CG_model.train()
    ############################################################
    out, pred_x_list = CG_model(DsImg, mask_dc, csm, DsImg_dc)
    MAE_loss = compute_cost_ksp(out, tstDsKsp, mask_loss, csm) + compute_cost_ksp(pred_x_list, tstDsKsp, mask_loss, csm)  
  
    # process to fit the range and size of image encoder's input
    recon_img = torch.abs(torch.cat(pred_x_list,dim=0))
    max_val   = torch.amax(torch.abs(recon_img),dim=(-1,-2),keepdim=True)
    recon_img = recon_img / max_val 
    recon_img = (recon_img -0.5) / 0.5
    recon_img = ChangeSize(recon_img,[384,384])
    vision_features =  vl_gpt.aligner(vl_gpt.vision_model(recon_img.to(torch.bfloat16).unsqueeze(1).repeat(1,3,1,1))) # image embedding
    new_inputs_embeds = inputs_embeds.detach().clone().repeat(niters,1,1)
    new_inputs_embeds[images_seq_mask] = vision_features[images_emb_mask]  # image-language embedding

    # the image-language embedding is processed by the large language model to generate the final embedding of the response
    outputs = vl_gpt.language_model(
        inputs_embeds=new_inputs_embeds,
        attention_mask=prepare_inputs.attention_mask.repeat(niters,1),
        output_hidden_states=True,
    )
    answer_embeds   = outputs.hidden_states[-1]
    con_loss =  ContrastiveLoss_language(answer_embeds[:,-1,:],pos_features.detach(), neg_features.detach())
 
    loss = MAE_loss + 5e2*con_loss 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_train += loss.item()
    iter_loop.set_postfix(loss='{:.4f}'.format(loss_train/(e+1)))
    writer.add_scalar('Loss/Total Loss', loss, e)
    writer.add_scalar('Loss/MAE Loss', MAE_loss, e)
    writer.add_scalar('Loss/Contrastive Loss_l0', con_loss, e)
    answer_embeds_cpu = answer_embeds[:,-1,:].float().cpu().detach().numpy()
    answer_embeds_list.append(answer_embeds_cpu)


    if e==0 or e==10 or e==20 or e==50 or e==100 or e==200 or (e+1)%summary_epoch == 0:
        with torch.no_grad():
            new_inputs_embeds = inputs_embeds.detach().clone().repeat(niters,1,1)
            new_inputs_embeds[images_seq_mask] = vision_features[images_emb_mask]

            # Here, we do the language generation to check the large language model's response
            outputs2 = vl_gpt.language_model.generate(
            inputs_embeds=new_inputs_embeds[-2:-1],
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            )
            answer2 = tokenizer.decode(outputs2[0].cpu().tolist(), skip_special_tokens=True)
            savemat(os.path.join(outpath,f'pred_results_{e+1}.mat'),  {'out':  torch.stack(out,-1).squeeze(0).cpu().detach().numpy(),
                                                                      'pred_x':torch.stack(pred_x_list,-1).squeeze(0).cpu().detach().numpy(),
                                                                     })
            with open(os.path.join(outpath,f"quality_evaluate_{e+1}.txt"), "w", encoding="utf-8") as f:
                f.write(answer2)
            print(answer2)
            torch.save(CG_model.state_dict(), os.path.join(model_path, f'CG_model_{e+1}.pkl'))

answer_embeds_list = np.array(answer_embeds_list)

save_file = os.path.join(outpath, 'answer_embeds_list.h5')
with h5.File(save_file, 'w') as f:
    f.create_dataset('answer_embeds_list', data=answer_embeds_list)