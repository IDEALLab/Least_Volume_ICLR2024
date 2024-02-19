import torch
import torch.nn.functional as F
from .autoencoder import AutoEncoder, BCEAutoencoder

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
    def __init__(self, configs: dict, Encoder, Decoder, Optimizer, weights=1):
        super().__init__(configs, Encoder, Decoder, Optimizer, weights)
        self.eps = configs['eps']

    def loss_spar(self, z):
        return torch.exp(torch.log(z.std(0) + self.eps).mean())

class VolumeAE_BCE(VolumeAE, BCEAutoencoder): pass

class L1AE(_SparseAE):
    def loss_spar(self, z):
        return z.std(0).norm(p=1) / z.size(1)

class L1AE_BCE(L1AE, BCEAutoencoder): pass

class L1AEv(_SparseAE):
    def loss_spar(self, z):
        return z.var(0).norm(p=1) / z.size(1)

class L1AE_BCEv(L1AEv, BCEAutoencoder): pass

class L2AE(_SparseAE):
    def loss_spar(self, z):
        return z.std(0).norm(p=2) / z.size(1)
    
class LassoAE(_SparseAE):
    def loss_spar(self, z):
        return z.abs().mean()

class LassoAE_BCE(LassoAE, BCEAutoencoder): pass

class STAE(_SparseAE):
    def loss_spar(self, z):
        return torch.log(1 + z.pow(2)).mean()
    
class STAE_BCE(STAE, BCEAutoencoder): pass