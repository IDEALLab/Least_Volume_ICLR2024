import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch as ft
from tqdm import tqdm

class _AutoEncoder(nn.Module):
    def __init__(self, configs: dict, Encoder, Decoder, Optimizer, weights=1):
        super().__init__()
        self.configs = configs
        self.encoder = Encoder(**configs['encoder'])
        self.decoder = Decoder(**configs['decoder'])
        self.optim = Optimizer(self.parameters(), **configs['optimizer'])
        self.register_buffer('w', torch.as_tensor(weights, dtype=torch.float))
        self.name = configs['name']
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
    
    def save(self, save_dir, epoch, *args, **kwargs):
        file_name = self.name+'{}'.format(epoch)+'_'.join([str(arg) for arg in args])
        torch.save({
            'params': self.state_dict(),
            'optim': self.optim.state_dict(),
            'configs': self.configs,
            'epoch': epoch,
            'misc': kwargs
            }, os.path.join(save_dir, file_name+'.tar'))

    def load(self, checkpoint):
        ckp = torch.load(checkpoint)
        self.load_state_dict(ckp['params'])
        self.optim.load_state_dict(ckp['optim'])
        self.configs = ckp['configs']
        self._init_epoch = ckp['epoch'] + 1


class AutoEncoder(_AutoEncoder):
    def loss(self, x, **kwargs):
        z = self.encode(x)
        x_hat = self.decode(z)
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

class CondensedAE(AutoEncoder):
    def __init__(self, configs: dict, Encoder: object, Decoder: object, Optimizer: object, weights=[1., 0.001]):
        super().__init__(configs, Encoder, Decoder, Optimizer, weights)
    
    def loss(self, x, **kwargs):
        z = self.encode(x)
        x_hat = self.decode(z)
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