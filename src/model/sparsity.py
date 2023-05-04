import torch
import torch.nn.functional as F
from .autoencoder import AutoEncoder
from math import sqrt

class _SparseAE(AutoEncoder):
    def loss(self, x, **kwargs):
        z = self.encode(x)
        x_hat = self.decode(z)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_spar(z)])
    
    def loss_spar(self, z):
        raise NotImplementedError
    
    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({'rec': loss[0].item(), 'spar': loss[1].item()}); pbar.refresh()

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer=None, callbacks=[], history=[], report_interval=1, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('rec', loss[0].item(), epoch)
                tb_writer.add_scalar('spar', loss[1].item(), epoch)
                tb_writer.add_histogram('z_std', self.encode(batch).std(0).detach().cpu().numpy(), epoch)
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
            self._rec_ = loss_rec.clone()
            self._zstd_ = z.std(0)
            self._zmean_ = z.mean(0)

    @torch.no_grad()
    def _prune(self):
        n = self.configs['data_dim']
        if self._rec_ / n < self._r_t:
            p_idx = self._zstd_ / n < self._z_t
            z_idx = (p_idx != self._p) & (self._p == False) # idx to be pruned: False -> True
            self._p = self._p | p_idx # update pruning index
            self._z[z_idx] = self._zmean_[z_idx] # type: ignore

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer=None, callbacks=[], history=[], report_interval=1, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('rec', loss[0].item(), epoch)
                tb_writer.add_scalar('spar', loss[1].item(), epoch)
                tb_writer.add_histogram('z_std', self.encode(batch).std(0).detach().cpu().numpy(), epoch)
                tb_writer.add_scalar('num_p', self._p.sum(), epoch)
            else:
                history.append(loss.detach().cpu().numpy())
        for callback in callbacks:
            try: 
                callback(self, epoch=epoch, epochs=pbar.total, batch=batch, loss=loss, tb_writer=tb_writer, **kwargs)
            except:
                pass

class DynamicPruningAE_v2(DynamicPruningAE):
    """This version requires only r_t threshold"""
    @torch.no_grad()
    def _prune(self):
        n = self.configs['data_dim']
        sup = self._zstd_[~self._p].min() # type:ignore
        if torch.sqrt(self._rec_) * sqrt(n) + sup < self._r_t: # type:ignore
            p_idx = self._zstd_ == sup
            z_idx = (p_idx != self._p) & (self._p == False) # idx to be pruned: False -> True
            self._p = self._p | p_idx # update pruning index
            self._z[z_idx] = self._zmean_[z_idx] # type: ignore

class DynamicPruningAE_v3(DynamicPruningAE):
    """This version requires only r_t threshold"""
    @torch.no_grad()
    def _prune(self):
        if self._rec_ < self._r_t:
            n = self.configs['data_dim']
            sup = self._zstd_[~self._p].min() # type:ignore
            if sup / n < self._z_t: # type:ignore
                p_idx = self._zstd_ == sup
                z_idx = (p_idx != self._p) & (self._p == False) # idx to be pruned: False -> True
                self._p = self._p | p_idx # update pruning index
                self._z[z_idx] = self._zmean_[z_idx] # type: ignore

class DynamicPruningAE_v4(DynamicPruningAE):
    def loss_spar(self, z):
        std = z.std(0)[~self._p]
        c = torch.exp(-20 * std.min()).detach()
        return torch.exp(torch.log(std + c).mean())   # type: ignore
    
    @torch.no_grad()
    def _prune(self):
        if self._rec_ < self._r_t:
            n = self.configs['data_dim']
            sup = self._zstd_[~self._p].min() # type:ignore
            if sup / n < self._z_t: # type:ignore
                p_idx = self._zstd_ == sup
                z_idx = (p_idx != self._p) & (self._p == False) # idx to be pruned: False -> True
                self._p = self._p | p_idx # update pruning index
                self._z[z_idx] = self._zmean_[z_idx] # type: ignore

class DynamicPruningAE_BCE(DynamicPruningAE_v3):
    def loss_rec(self, x, x_hat):
        return F.binary_cross_entropy(x_hat, x)
    
class DynamicPruningAE_BCEO(DynamicPruningAE_v2):
    '''single delta control'''
    def loss_rec(self, x, x_hat):
        return F.binary_cross_entropy(x_hat, x), F.mse_loss(x_hat, x)
    
    def loss(self, x, **kwargs):
        z = self.encode(x)
        loss_rec, mse_loss = self.loss_rec(x, self.decode(z))
        loss_spar = self.loss_spar(z)

        self._update_moving_mean(mse_loss, z)
        self._prune()
        return torch.stack([loss_rec, loss_spar])

    @torch.no_grad()
    def _update_moving_mean(self, loss_rec, z):
        if all(each is not None for each in [self._rec_, self._zstd_, self._zmean_]):
            self._rec_ = torch.lerp(self._rec_, loss_rec, 1-self._beta)
            self._zstd_ = torch.lerp(self._zstd_, z.std(0), 1-self._beta) 
            self._zmean_ = torch.lerp(self._zmean_, z.mean(0), 1-self._beta)
        else:
            self._rec_ = loss_rec.clone()
            self._zstd_ = z.std(0)
            self._zmean_ = z.mean(0)

class DynamicPruningAE_BCEv2(DynamicPruningAE_v4):
    def loss_rec(self, x, x_hat):
        return F.binary_cross_entropy(x_hat, x)

class L1AE(_SparseAE):
    def loss_spar(self, z):
        return z.std(0).norm(p=1)

class L1AE_BCE(L1AE):
    def loss_rec(self, x, x_hat):
        return F.binary_cross_entropy(x_hat, x)

class L2AE(_SparseAE):
    def loss_spar(self, z):
        return z.std(0).norm(p=1)

class LassoAE(_SparseAE):
    def loss_spar(self, z):
        return z.norm(dim=-1, p=1).mean()

class LassoAE_BCE(LassoAE):
    def loss_rec(self, x, x_hat):
        return F.binary_cross_entropy(x_hat, x)