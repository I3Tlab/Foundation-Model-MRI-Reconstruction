'''
This file constains the loss functions used in this study.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


MAE_loss_function = torch.nn.L1Loss()
def compute_cost_ksp(output, target, mask, csm):
    mask = mask.unsqueeze(0).unsqueeze(0)
    cost = 0.
    for img in output:
        output_ksp   = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift((img.unsqueeze(1))*csm,dim=(-1,-2)), dim=(-1,-2)), dim=(-1,-2))  
        cost         +=  MAE_loss_function(torch.view_as_real((output_ksp*mask)).float(), 
                                            torch.view_as_real(target*mask).float())
    return cost

def compute_TV_loss(x):
    N1, N2  = x.shape[0], x.shape[1]
    tv_loss = (torch.sum(torch.abs(x[1:, :, :] - x[:N1-1, :, :])) + torch.sum(torch.abs(x[:, 1:, :] - x[:,:N2-1,:])))/ ((N1-1)*(N2-1))

    return tv_loss


def pca_transform(x, W, mu):
    # x: (B, D)
    return (x - mu) @ W  # (B, out_dim)


class ContrastiveLoss_image(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
      
        self.temperature = temperature

    def forward(self, query, positives, negatives, W, mu):
        """
        query: (N1, 576, 1024)
        positives: (N, 576, 1024)
        negatives: (M, 576, 1024)
        """
        B,N,C = positives.shape
        B2 = negatives.shape[0]
        query     = query.reshape(-1, C)
        positives = positives.reshape(-1, C)
        negatives = negatives.reshape(-1, C)

        # PCA projection
        q_proj    = pca_transform(query, W, mu).reshape(1, N, -1)
        pos_proj  = pca_transform(positives, W, mu).reshape(B, N, -1)
        neg_proj  = pca_transform(negatives, W, mu).reshape(B2, N, -1)

        all_features = torch.cat([pos_proj, neg_proj], dim=0)

        # Normalize
        all_features = F.normalize(all_features, p=2, dim=-1)
        q_proj       = F.normalize(q_proj, p=2, dim=-1)

        # Similarity
        sim = torch.sum(q_proj * all_features, dim=-1).mean(-1) / self.temperature

        numerator   = torch.exp(sim[:pos_proj.shape[0]]).sum()
        denominator = torch.exp(sim).sum()

        loss = -torch.log(numerator / denominator + 1e-8)
        return loss
    
def ContrastiveLoss_language(
        query, pos_features, neg_features,
        tau_pos=0.1, tau_neg=0.7):
        """
        Improved version: 
        - log-sum-exp for positives (encourages pulling toward entire pos cluster)
        - larger temperature for negatives (weaker push-away)
        """

        # Normalize
        query = F.normalize(query, p=2, dim=-1).unsqueeze(0)      # (1,4,D)
        pos = F.normalize(pos_features, p=2, dim=-1).unsqueeze(1) # (P,1,D)
        neg = F.normalize(neg_features, p=2, dim=-1).unsqueeze(1) # (Q,1,D)

        # cosine similarities
        sim_pos = torch.sum(query * pos, dim=-1) / tau_pos  # (P,4)
        sim_neg = torch.sum(query * neg, dim=-1) / tau_neg  # (Q,4)

        # log-sum-exp over positives (stronger cluster-wise pull)
        lse_pos = torch.logsumexp(sim_pos, dim=0)  # (4,)

        # log-sum-exp over all (pos + neg)
        all_sim = torch.cat([sim_pos, sim_neg], dim=0)
        lse_all = torch.logsumexp(all_sim, dim=0)

        loss = (-(lse_pos - lse_all)).sum()

        return loss

