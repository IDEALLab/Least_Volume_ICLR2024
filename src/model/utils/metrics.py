import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def l2_loss(x, y):
    d = x - y
    return d.flatten(1).norm(dim=1).mean()

@torch.no_grad()
def explained_reconstruction(encoder, decoder, dataset, rec_loss, idx=None):
    z = encoder(dataset)
    eps = rec_loss(decoder(z), dataset) # inherent loss
    z_mean = z.mean(0)
    denom = rec_loss(
        decoder(z_mean.unsqueeze(0)).expand(len(dataset), -1), 
        dataset
        )
    
    if idx is None:
        loss = torch.empty_like(z_mean)
        for i in range(z.size(-1)):
            z_p = z.clone(); z_p[:, i] = z_mean[i]
            loss[i] = rec_loss(decoder(z_p), dataset) 
        exp_rec = (loss - eps) / (denom - eps)
        return *exp_rec.sort(), z.std(0) # e_r and idx
    else:
        z_p = z.clone(); z_p[:, idx] = z_mean[idx]
        loss = rec_loss(decoder(z_p), dataset)
        exp_rec = (loss - eps) / (denom - eps)
        return exp_rec

@torch.no_grad()
def mean_correlation(encoder, dataset, idx):
    pass

@torch.no_grad()
def importance_correlation(z_std, exp_rec):
    X = torch.stack([z_std, exp_rec]) 
    return torch.corrcoef(X)[1, 1]