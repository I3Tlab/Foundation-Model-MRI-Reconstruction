'''
This is a implementation of the unrolled model, which unrolls a variable-splitting iterative reconstruction algorithm into a sequence of learnable stages. 
Each stage alternates between a U-Net and an explicit data-consistency step solved using conjugate gradient descent.
'''

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

    
class Aclass_MC:
    ''' data consistency module for multi-coil reconstruction '''
    def __init__(self, mask, csm,lam):
        # mask: mask of the k-space Nrd x Npe
        self.nrow, self.ncol = mask.shape[-2], mask.shape[-1]
        self.pixels          = self.nrow * self.ncol
        self.mask            = mask.unsqueeze(0)
        self.csm             = csm
        self.lam             = lam

    def myAtA(self, img):
        # img: B x Nrd xNpe
        # csm: B x Ncoil x Nrd x Npe
        # forward operator
        coilImages     = (self.csm)*(img.unsqueeze(0))  # B x Ncoil x Nrd x Npe
        kspace         = fft.fftshift(fft.fft2(fft.fftshift(coilImages,dim=(-1,-2)),dim=(-1,-2)),dim=(-1,-2))
        temp           = kspace*self.mask               # B x Ncoil x Nrd x Npe
        coilImg_out    = fft.fftshift(fft.ifft2(fft.fftshift(temp,dim=(-1,-2)),dim=(-1,-2)),dim=(-1,-2))
        coilComb       = torch.sum(coilImg_out*torch.conj(self.csm),dim=0)
        coilComb       = coilComb + self.lam*img

        return coilComb

def myCG(A, rhs, max_iter=20):
    ''' conjugate gradient method '''
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rTr = torch.sum(torch.conj(r)*r).real

    for _ in range(max_iter):
        if rTr < 1e-10:
            break

        Ap    = A.myAtA(p)
        alpha = rTr / torch.sum(torch.conj(p)*Ap).real
        x     = x + alpha * p
        r     = r - alpha * Ap
        rTrNew = torch.sum(torch.conj(r) * r).real

        beta = rTrNew / rTr
        p = r + beta * p
        rTr = rTrNew

    return x 

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
    
class Unet_CG(nn.Module):
    def __init__(self, unet, lam, niter, input_mean_val, input_std_val, label_mean_val, label_std_val):  #input_mean_val, input_std_val, label_mean_val, label_std_val
        super(Unet_CG, self).__init__()
        self.unet_model          = unet
        self.lam                 = lam           # lambda
        self.niter               = niter         # number of iterations
        self.input_mean_val      = input_mean_val
        self.input_std_val       = input_std_val
        self.label_mean_val      = label_mean_val
        self.label_std_val       = label_std_val

    def forward(self, atb, mask, csm, atb_dc):
        out = [atb]
        pred_x_list = []

        for k in range(self.niter):
            # 1. cut or pad to fit the Unet input size
            x = ChangeSize(out[k], [384, 384])
            x = torch.stack([x.real, x.imag],dim=1).float()

            # 2. Unet
            x = (x - self.input_mean_val ) / self.input_std_val
            pred_x   = self.unet_model[k](x,return_feature = False)    ############## [k]
            pred_x   = pred_x * self.label_std_val  + self.label_mean_val
            pred_x   = torch.complex(pred_x[:,0,:,:], pred_x[:,1,:,:])
         

            # Change size back to the original size of atb
            pred_x          = ChangeSize(pred_x, [atb.shape[-2], atb.shape[-1]])
            pred_x_list.append(pred_x)
            
            # 3. DC block
            rhs = atb_dc + self.lam[k]*pred_x     ##############[k]
            def process_slice(args):
                rhs_slice, mask_slice, csm_slice, lam = args
                A = Aclass_MC(mask_slice, csm_slice, lam)
                return myCG(A, rhs_slice)
            
            batch_size = rhs.shape[0]
            dc_output = torch.stack([process_slice(
                    (rhs[i], mask, csm[i], self.lam[k])) for i in range(batch_size)])   #################[k]
            
            out.append(dc_output)

        return out, pred_x_list
    
