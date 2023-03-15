import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import functorch as ft
import math
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy
from model.autoencoder import AutoEncoder
from model.cmpnts import MLP
from dataset.toy import TensorDataset, DataLoader
from itertools import product

class Experiment:
    def __init__(self, configs, Encoder, Decoder, Optimizer, AE, device='cpu') -> None:
        self.configs = configs
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Optimizer = Optimizer
        self.AE = AE
        self.device = device
    
    def run(self, dataloader, epochs, lams, save_dir):
        for lam in lams:
            history = []
            self.init_model(lam)
            self.model.fit(
                dataloader=dataloader,
                epochs=epochs, # maybe convergence criterion is better
                history=history
                )
            self.save_result(lam, history, epochs, save_dir)
    
    def save_result(self, lam, history, epochs, save_dir):
        path = os.path.join(save_dir, '{:.0e}'.format(lam)) # model/man/amb/i/lam
        os.makedirs(path, exist_ok=True)
        self.model.save(path, epochs)
        np.savetxt(os.path.join(path, self.model.name+'_history.csv'), np.asarray(history))
        
    def init_model(self, lam):
        self.model = self.AE(
            self.configs, self.Encoder, self.Decoder, 
            self.Optimizer, weights=[1., lam]
            )
        self.model.to(self.device)


##### Models #####

class _SparseAE(AutoEncoder):
    def loss(self, x, **kwargs):
        z = self.encode(x)
        x_hat = self.decode(z)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_spar(z)])
    
    def loss_spar(self, z):
        raise NotImplementedError
    
    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({'rec': loss[0].item(), 'spar': loss[1].item()}); pbar.refresh()

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer=None, callbacks=[], report_interval=1, history=[], **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('rec', loss[0].item(), epoch)
                tb_writer.add_scalar('spar', loss[1].item(), epoch)
                tb_writer.add_histogram('z_std', self.encoder(batch).std(0).detach().cpu().numpy(), epoch)
            else:
                history.append(loss.detach().cpu().numpy())
        for callback in callbacks:
            try: 
                callback(self, epoch=epoch, epochs=pbar.total, batch=batch, loss=loss, tb_writer=tb_writer, **kwargs)
            except:
                pass

class VolumeAE(_SparseAE):
    def loss_spar(self, z):
        return torch.exp(torch.log(z.std(0)+1e-2).mean())

class DynamicPruningAE(_SparseAE):
    def __init__(self, configs: dict, Encoder, Decoder, Optimizer, weights=[1., 1e-3]):
        super().__init__(configs, Encoder, Decoder, Optimizer, weights)
        m = configs['decoder']['in_features']
        self.register_buffer('_p', torch.as_tensor([False] * m, dtype=torch.bool)) # indices to be pruned
        self.register_buffer('_z', torch.zeros(m)) # pruning values
        self._rec_ = self._zstd_ = self._zmean_ = None # moving averages for pruning
        self._beta, self._z_t, self._r_t = configs['beta'], configs['z_t'], configs['r_t'] # momentum and thresholds for pruning

    def decode(self, z):
        z[:, self._p] = self._z[self._p]    # type: ignore
        return self.decoder(z)

    def encode(self, x):
        z = self.encoder(x)
        z[:, self._p] = self._z[self._p]    # type: ignore
        return z

    def loss(self, x, **kwargs):
        z = self.encode(x)
        loss_rec = self.loss_rec(x, self.decode(z))
        loss_spar = self.loss_spar(z)

        self._update_moving_mean(loss_rec, z)
        self._prune()
        return torch.stack([loss_rec, loss_spar])

    def loss_spar(self, z):
        return torch.exp(torch.log(z.std(0)[~self._p]).mean())   # type: ignore
    
    @torch.no_grad()
    def _update_moving_mean(self, loss_rec, z):
        if all(each is not None for each in [self._rec_, self._zstd_, self._zmean_]):
            self._rec_ = torch.lerp(self._rec_, loss_rec, 1-self._beta)
            self._zstd_ = torch.lerp(self._zstd_, z.std(0), 1-self._beta) 
            self._zmean_ = torch.lerp(self._zmean_, z.mean(0), 1-self._beta)
        else:
            self._rec_ = loss_rec
            self._zstd_ = z.std(0)
            self._zmean_ = z.mean(0)

    @torch.no_grad()
    def _prune(self):
        if self._rec_ < self._r_t:
            p_idx = self._zstd_ < self._z_t
            z_idx = (p_idx != self._p) & (self._p == False) # idx to be pruned: False -> True
            self._p |= p_idx # update pruning index
            self._z[z_idx] = self._zmean_[z_idx] # type: ignore


class L1AE(_SparseAE):
    def loss_spar(self, z):
        return z.std(0).norm(p=1)

class L2AE(_SparseAE):
    def loss_spar(self, z):
        return z.std(0).norm(p=1)

class LassoAE(_SparseAE):
    def loss_spar(self, z):
        return z.norm(dim=-1, p=1).mean()

ae_dict = {
    'vol': VolumeAE,
    'l1': L1AE,
    'l2': L2AE,
    'lasso': LassoAE
}


##### Functions ######

def generate_configs(data_dim, width, name):
    configs = dict()
    configs['encoder'] = {
        'in_features': data_dim,
        'out_features': min(128, data_dim),
        'layer_width': width
        }
    configs['decoder'] = {
        'in_features': min(128, data_dim),
        'out_features': data_dim,
        'layer_width': width
        }
    configs['optimizer'] = {'lr': 1e-3}
    configs['name'] = name
    return configs

def read_dataset(latent_dim, data_dim, i, device='cpu'):
    path = '../data/toy/{}-manifold/{}-ambient/'.format(latent_dim, data_dim)
    data_name = '{}-{}_{}.npy'.format(latent_dim, data_dim, i)
    tensor = torch.as_tensor(np.load(os.path.join(path, data_name)), dtype=torch.float, device=device) # type:ignore
    return TensorDataset(tensor)

def create_savedir(l, d, i):
    dir = os.path.join('../saves/toy/{}-man/{}-amb/#{}/'.format(l, d, i))
    os.makedirs(dir, exist_ok=True)
    return dir


#### main #####

def main(ae_name, epochs=10000, batch=100, device='cpu'):
    ll = [1, 2, 4, 8, 16, 32]
    dd = [2, 4, 8, 16, 32]
    ii = range(5)
    lams = 10 ** np.linspace(-6, 0, 13)
    ww = [[32]*4, [48]*4, [64]*4, [96]*4, [128]*4, [256]*4]

    for l, width in zip(ll, ww):
        for d, i in product(dd, ii):
            dataset = read_dataset(l, d*l, i, device=device)
            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
            configs = generate_configs(d*l, width, ae_name)
            save_dir = create_savedir(l, d*l, i)

            experiment = Experiment(configs, MLP, MLP, Adam, ae_dict[ae_name], device=device)
            experiment.run(dataloader=dataloader, epochs=epochs, lams=lams, save_dir=save_dir) # epochs to be modified


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ae_name', help='name of the autoencoder to be trained')
    parser.add_argument('-d', '--device', default='cpu', help='device to run the experiments')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('-b', '--batch', type=int, default=100, help='number of samples in a mini-batch')
    args = parser.parse_args()

    main(args.ae_name, args.epochs, args.batch, args.device)