import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch as ft
import math
from torch.optim import Optimizer
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from .utils import first_element
from random import randrange
from tqdm import tqdm

class _AutoEncoder(nn.Module):
    def __init__(self, configs: dict, Encoder, Decoder, Optimizer, weights=1):
        super().__init__()
        self.configs = configs
        self.encoder = Encoder(**configs['encoder'])
        self.decoder = Decoder(**configs['decoder'])
        self.optim = Optimizer(self.parameters(), **configs['optimizer'])
        self.register_buffer('w', torch.as_tensor(weights, dtype=torch.float))
        self.name = configs['vae']['name']
        self._init_epoch = 0
    
    def loss(self, batch, **kwargs):
        """tensor of loss terms"""
        raise NotImplementedError

    def decode(self, latent_code):
        return self.decoder(latent_code)
    
    def encode(self, x_batch):
        return self.encoder(x_batch)

    def fit(
        self, dataloader, epochs,
        save_dir=None, save_iter_list=[100,], tb_writer=None, callbacks=[], **kwargs
        ):
        with tqdm(
            range(self._init_epoch, epochs), initial=self._init_epoch, total=epochs,
            bar_format='{l_bar}{bar:30}{r_bar}', desc='Training {}'.format(self.name)
            ) as pbar:
            for epoch in pbar:
                self._epoch_hook(epoch, pbar, tb_writer, callbacks, **kwargs)
                for i, batch in enumerate(dataloader):
                    self._batch_hook(i, batch, pbar, tb_writer, callbacks, **kwargs)
                    self.optim.zero_grad()
                    loss = self.loss(batch, **kwargs)
                    (loss * self.w).sum().backward()
                    self.optim.step()
                    self._batch_report(i, batch, loss, pbar, tb_writer, callbacks, **kwargs)
                self._epoch_report(epoch, batch, loss, pbar, tb_writer, callbacks, **kwargs)
    
                if save_dir:
                    if save_iter_list and epoch in save_iter_list:
                        self.save(save_dir, epoch=epoch)
    
    def _batch_hook(self, i, batch, pbar, tb_writer, callbacks, **kwargs): pass
    
    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): pass

    def _epoch_hook(self, epoch, pbar, tb_writer, callbacks, **kwargs): pass

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer, callbacks, **kwargs): pass
    
    def save(self, save_dir, **kwargs):
        torch.save({
            'params': self.state_dict(),
            'optim': self.optim.state_dict(),
            'configs': self.configs,
            'misc': kwargs
            }, os.path.join(save_dir, self.name+str(kwargs['epoch'])+'.tar'))

    def load(self, checkpoint):
        ckp = torch.load(checkpoint)
        self.load_state_dict(ckp['params'])
        self.optim.load_state_dict(ckp['optim'])
        self.configs = ckp['configs']
        self._init_epoch = ckp['misc']['epoch'] + 1


class AutoEncoder(_AutoEncoder):
    def loss(self, x, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return self.loss_rec(x, x_hat)

    def loss_rec(self, x, x_hat):
        return F.mse_loss(x, x_hat)

    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({'rec': loss.item()}); pbar.refresh()

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer, callbacks=[], report_interval=1, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('rec', loss.item(), epoch)
            else:
                pass
        for callback in callbacks:
            try: 
                callback(self, epoch=epoch, epochs=pbar.total, batch=batch, loss=loss, tb_writer=tb_writer, **kwargs)
            except:
                pass


class IsometricAE(AutoEncoder):
    def loss(self, x, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        v = F.normalize(torch.randn_like(z), dim=1)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_iso(z, v, 1),  self.loss_piso(x, v, 1)])

    def loss_iso(self, z, v, n=None):
        f = lambda z: self.decoder(z)
        jvp = ft.jvp(f, (z[:n],), (v[:n],))[1]
        norm = jvp.flatten(1).norm(dim=1)
        return F.mse_loss(norm, torch.ones_like(norm))

    def loss_piso(self, x, v, n=None):
        g = lambda x: self.encoder(x)
        vjp, = ft.vjp(g, x[:n])[1](v[:n]) # i.e., vjp_fn(v)
        norm = vjp.flatten(1).norm(dim=1)
        return F.mse_loss(norm, torch.ones_like(norm))
    
    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({
            'rec': loss[0].item(), 
            'iso': loss[1].item(), 
            'piso': loss[2].item()
            }) 
        pbar.refresh()

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer, callbacks=[], report_interval=1, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('rec', loss[0].item(), epoch)
                tb_writer.add_scalar('iso', loss[1].item(), epoch)
                tb_writer.add_scalar('piso', loss[2].item(), epoch)
            else:
                pass
        for callback in callbacks:
            try: 
                callback(self, epoch=epoch, epochs=pbar.total, batch=batch, loss=loss, tb_writer=tb_writer, **kwargs)
            except:
                pass

class CondensedAE(AutoEncoder):
    def __init__(self, configs: dict, Encoder: object, Decoder: object, Optimizer: object, weights=[1., 0.001]):
        super().__init__(configs, Encoder, Decoder, Optimizer, weights)
    
    def loss(self, x, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_vol(z)])
    
    def loss_vol(self, z):
        return torch.exp(torch.log(z.std(0)+1).mean())
    
    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({'rec': loss[0].item(), 'vol': loss[1].item()}); pbar.refresh()

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer, callbacks=[], report_interval=1, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('rec', loss[0].item(), epoch)
                tb_writer.add_scalar('vol', loss[1].item(), epoch)
                tb_writer.add_histogram('z_std', self.encoder(batch).std(0).detach().cpu().numpy(), epoch)
            else:
                pass
        for callback in callbacks:
            try: 
                callback(self, epoch=epoch, epochs=pbar.total, batch=batch, loss=loss, tb_writer=tb_writer, **kwargs)
            except:
                pass

class VAE(_AutoEncoder):
    def __init__(self, configs: dict, Encoder, Decoder, Optimizer):
        super().__init__(configs, Encoder, Decoder, Optimizer)
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.optim = Optimizer(self.parameters(), **configs['optimizer'])
    
    def _kl_div_qp(self, x):
        z_mean, z_log_std = self.encoder(x)
        z_std = torch.exp(z_log_std)
        p_z = Independent(Normal(torch.zeros_like(z_mean), torch.ones_like(z_std)), 1)
        q_zx = Independent(Normal(z_mean, z_std), 1)
        return kl_divergence(q_zx, p_z) # [batch]
    
    def _log_pxz(self, x): # batch: samples of x
        z_mean, z_log_std = self.encoder(x) # [batch, latent_dim]
        z_std = torch.exp(z_log_std)
        q_zx = Independent(Normal(z_mean, z_std), 1)
        x_hat = first_element(self.decoder(q_zx.rsample())) # [batch, x_dim_0,...]
        p_xz = Independent(Normal(x_hat, torch.exp(self.log_sigma)), len(x_hat.shape[1:]))
        return p_xz.log_prob(x) # [batch]

    def elbo(self, x, z_num=1):
        expz_log_pxz = torch.stack([self._log_pxz(x) for _ in range(z_num)]).mean(0)
        return torch.mean(expz_log_pxz - self._kl_div_qp(x))
    
    def loss(self, batch, **kwargs):
        return -self.elbo(batch)
    
    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({'ELBO': -loss.item()}); pbar.refresh()

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer, callbacks=[], report_interval=1, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('ELBO', -loss, epoch)
            else:
                print('[Epoch {}/{}] ELBO: {:d}'.format(epoch, pbar.total, -loss))
        for callback in callbacks:
            try: 
                callback(self, epoch=epoch, epochs=pbar.total, batch=batch, loss=loss, tb_writer=tb_writer, **kwargs)
            except:
                pass

class BinaryVAE(VAE):
    def _log_pxz(self, x): # batch: samples of x
        z_mean, z_log_std = self.encoder(x) # [batch, latent_dim]
        z_std = torch.exp(z_log_std)
        q_zx = Independent(Normal(z_mean, z_std), 1)
        x_hat = first_element(self.decoder(q_zx.rsample())) # [batch, x_dim_0,...]
        return -F.binary_cross_entropy(x_hat, x, reduction='none').flatten(1).sum(-1) # / torch.exp(self.log_sigma).pow(2) / 2 # [batch]