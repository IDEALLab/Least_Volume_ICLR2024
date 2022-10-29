import os
from pyexpat.errors import XML_ERROR_INVALID_TOKEN
from matplotlib.colors import same_color
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim import Optimizer
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from torch.autograd.functional import jacobian, jvp, vjp
from .sinkhorn import sinkhorn_divergence
from .utils import first_element
from random import randrange
from tqdm import tqdm

class _AutoEncoder(nn.Module):
    def __init__(self, configs: dict, Encoder: object, Decoder: object, Optimizer: object, weights=1):
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
        x_hat = first_element(self.decoder(z))
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
        x_hat = first_element(self.decoder(z))
        v = torch.randn_like(z)
        v = v / torch.norm(v, dim=-1, keepdim=True)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_iso_mc(z, v),  self.loss_piso_mc(x, v)])
    
    def loss_iso_mc(self, z, v):
        idx = randrange(0, len(v))
        z = z[idx]; v = v[idx]
        f = lambda z: first_element(self.decoder(z.unsqueeze(0))).flatten()
        p = jvp(f, z, v, create_graph=True)[-1]
        norm = torch.norm(p)
        return F.mse_loss(norm, torch.ones_like(norm))

    def loss_piso_mc(self, x, v):
        idx = randrange(0, len(v))
        x = x[idx]; v = v[idx]
        shape = x.shape; x = x.flatten()
        g = lambda x: self.encoder(x.view(shape).unsqueeze(0)).flatten()
        p = vjp(g, x, v, create_graph=True)[-1]
        norm = torch.norm(p)
        return F.mse_loss(norm, torch.ones_like(norm))

    def loss_iso_batch(self, z, v):
        f = lambda z: first_element(self.decoder(z)).flatten(1).sum(0)
        j = jacobian(f, z, create_graph=True).permute(1, 0, 2)
        norm = torch.norm(torch.bmm(j, v.unsqueeze(-1)), dim=(1, 2))
        return F.mse_loss(norm, torch.ones_like(norm))

    def loss_piso_batch(self, x, v):
        shape = x.shape
        x = x.flatten(1)
        g = lambda x: self.encoder(x.view(shape)).sum(0)
        j = jacobian(g, x, create_graph=True).permute(1, 2, 0)
        norm = torch.norm(torch.bmm(j, v.unsqueeze(-1)), dim=(1, 2))
        return F.mse_loss(norm, torch.ones_like(norm))
    
    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({'rec': loss[0].item(), 'iso': loss[1].item(), 'piso': loss[2].item()}); pbar.refresh()

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
        x_hat = first_element(self.decoder(z))
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

class AEAI(AutoEncoder):
    def __init__(self, configs: dict, Encoder: object, Decoder: object, Critic: object, Optimizer_AE: object, Optimizer_C: object, weights=1):
        super().__init__(configs, Encoder, Decoder, Optimizer_AE, weights)
        self.critic = Critic(**configs['critic'])
        self.optim_C = Optimizer_C(self.critic.parameters(), **configs['optimizer_C'])

    def loss(self, x, **kwargs):
        z = self.encoder(x)
        x_hat = first_element(self.decoder(z))
        z_perm = torch.cat([z[1:], z[:1]])
        
        a = torch.rand(len(z), 5).to(z.device) # [b, n]

        def f(a, output_z=False):
            a = a.unsqueeze(-1)
            z_int = torch.flatten(z.unsqueeze(1) * a + z_perm.unsqueeze(1) * (1-a), 0, 1) # [b*n, z] # detach to see the effect
            x_int = first_element(self.decoder(z_int)) # [b*n, x]
            return x_int, z_int if output_z else x_int.sum(0) # [x]
        
        x_int, z_int = f(a, output_z=True)
        return torch.stack([self.loss_rec(x, x_hat), self.loss_adv_ae(x_int), self.loss_cyc(x_int, z_int), self.loss_smooth(f, a)])

    def loss_c(self, x, **kwargs):
        z = self.encoder(x)
        z_perm = torch.cat([z[1:], z[:1]])
        a = torch.rand(len(z), 5).unsqueeze(-1).to(z.device) # [b, n]
        z_int = torch.flatten(z.unsqueeze(1) * a + z_perm.unsqueeze(1) * (1-a), 0, 1) # [b*n, z]
        x_int = first_element(self.decoder(z_int))
        return self.loss_adv_c(x, x_int)


    def loss_rec(self, x, x_hat):
        return F.mse_loss(x, x_hat)
    
    def loss_adv_c(self, x, x_int):
        logit_t = self.critic(x)
        logit_f = self.critic(x_int)
        return F.binary_cross_entropy_with_logits(logit_t, torch.ones_like(logit_t)) \
            + F.binary_cross_entropy_with_logits(logit_f, torch.zeros_like(logit_f))

    def loss_adv_ae(self, x_int):
        logit_f = self.critic(x_int)
        return F.binary_cross_entropy_with_logits(logit_f, torch.ones_like(logit_f))

    def loss_cyc(self, x_int, z_int):
        z_hat = self.encoder(x_int)
        return F.mse_loss(z_hat, z_int.detach()) # z_int) 
    
    def loss_smooth(self, f, a):
        # jcb = jacobian(f, a, create_graph=True)
        # loss = F.mse_loss(jcb, torch.zeros_like(jcb))
        # return loss
        return torch.tensor(0., device=a.device)
    
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
                    if i % 1 == 0:
                        self.optim_C.zero_grad()
                        loss_c = self.loss_c(batch, **kwargs)
                        loss_c.backward()
                        self.optim_C.step()

                    self.optim.zero_grad()
                    loss = self.loss(batch, i=i, **kwargs)
                    (loss * self.w).sum().backward()
                    self.optim.step()
                    self._batch_report(i, batch, loss, pbar, tb_writer, callbacks, **kwargs)
                self._epoch_report(epoch, batch, loss, pbar, tb_writer, callbacks, **kwargs)
    
                if save_dir:
                    if save_iter_list and epoch in save_iter_list:
                        self.save(save_dir, epoch=epoch)

    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({'rec': loss[0].item(), 'adv': loss[1].item(), 'cyc': loss[2].item(), 'sm': loss[3].item()}); pbar.refresh()

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer, callbacks=[], report_interval=1, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('rec', loss[0].item(), epoch)
                tb_writer.add_scalar('adv', loss[1].item(), epoch)
                tb_writer.add_scalar('cyc', loss[2].item(), epoch)
                tb_writer.add_scalar('sm', loss[3].item(), epoch)
            else:
                pass
        for callback in callbacks:
            try: 
                callback(self, epoch=epoch, epochs=pbar.total, batch=batch, loss=loss, tb_writer=tb_writer, **kwargs)
            except:
                pass

class AEAIAlter(AEAI):
    gamma = 1
    def _inject_noise(self, x):
        noise = torch.randn_like(x) * self.gamma
        return (noise + x).clip(0, 1)

    def _interp(self, z, n, seeds=2, offset=0.):
        z_ = torch.stack([z[torch.randperm(n) % len(z)] for _ in range(seeds)], dim=1) # [n, 1, z]
        w = F.normalize(torch.rand(n, seeds).to(z.device), p=1., dim=-1) # [n, s]
        o_ = offset * (torch.rand(n, seeds-1).to(z.device) - 0.5)
        o = torch.cat([o_, -o_.sum(-1, keepdim=True)], dim=-1)
        return torch.sum((w + o).unsqueeze(-1) * z_, dim=1) # [n, s, z] -> [n, z]

    def loss(self, x, i, **kwargs):
        z = self.encoder(self._inject_noise(x))
        x_hat = self.decoder(z)

        z_int = self._interp(z, n=1*len(z), seeds=2, offset=0.5)
        w = self.decoder._w_forward(z_int)
        w_mix = self.mix_code(w)
        x_mix = self.decoder._style_forward(w_mix)
        x_int = self.decoder(z_int)

        loss_rec = self.loss_rec(x, x_hat)
        loss_adv_ae = self.loss_adv_ae(x_mix)
        loss_cyc = self.loss_cyc(x_int, z_int)
        loss_smooth = self.loss_smooth(z_int, x_int) if i % 8 == 0 else torch.zeros_like(loss_cyc)
        return torch.stack([loss_rec, loss_adv_ae, loss_cyc, loss_smooth])

    def loss_c(self, x, **kwargs):
        with torch.no_grad():
            z = self.encoder(self._inject_noise(x))
            z_int = self._interp(z, n=1*len(z), seeds=2, offset=0.5)
            w = self.decoder._w_forward(z_int)
            w_mix = self.mix_code(w)
            x_mix = self.decoder._style_forward(w_mix)
        return self.loss_adv_c(x, x_mix)

    def mix_code(self, w, prob=0.9): # w.shape = [batch, w_dim] -> [block, batch, w_dim]
        w = self.decoder._expand_w(w)
        if torch.empty([]).bernoulli(prob):
            w_ = w.roll(1, dims=1) # w[:, torch.randperm(w.size(1))]
            i = torch.randint(0, w.size(0), [])
            w = torch.cat([w[:i], w_[i:]], dim=0)
        return w
    
    def loss_smooth(self, z_int, x_int):
        noise = torch.randn_like(x_int) / math.sqrt(x_int.size(2)*x_int.size(3))
        grad, = torch.autograd.grad(torch.sum(x_int*noise), z_int, create_graph=True) #[1, batch, w_dim]
        return grad.squeeze(0).norm(dim=-1).var()

    # def loss_smooth(self, z_int):
    #     z_int = z_int[:16]
    #     f = lambda z: first_element(self.decoder(z.unsqueeze(0)))
    #     norms = z_int.new_empty(len(z_int))
    #     for i, z in enumerate(z_int.detach()):
    #         v = F.normalize(torch.randn_like(z), dim=None)
    #         p = jvp(f, z, v, create_graph=True)[1]
    #         norms[i] = p.norm()
    #     return norms.mean()

class BinaryAEAI(AEAI):
    def loss_rec(self, x, x_hat):
        return F.binary_cross_entropy(x_hat, x)

class BinaryAEAIAlter(AEAIAlter):
    def loss_rec(self, x, x_hat):
        return F.binary_cross_entropy(x_hat, x)

class VAE(_AutoEncoder):
    def __init__(self, configs: dict, Encoder: object, Decoder: object, Optimizer: object):
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

class SinkhornVAE(_AutoEncoder):
    def __init__(
        self, configs: dict, Encoder: object, Decoder: object, Optimizer: object, 
        cost_func=lambda x1, x2: torch.cdist(x1.flatten(1), x2.flatten(1), p=1)
        ):
        super().__init__(configs, Encoder, Decoder, Optimizer)
        self.lamb = configs['vae']['lamb']
        self.cost = cost_func

    def _sink_loss(self, x, z):
        z_hat = first_element(self.encode(x))
        x_hat = first_element(self.decode(z))
        a = torch.ones(len(x), 1, device=x.device) / len(x)
        b = torch.ones(len(z), 1, device=z.device) / len(z)
        return sinkhorn_divergence(
            a, (x, z_hat.contiguous()), b, (x_hat.contiguous(), z), 
            eps=self.lamb, assume_convergence=True, cost_func=self.cost, nits=1000
            )

    def loss(self, batch, noise_gen, **kwargs):
        return self._sink_loss(batch, noise_gen(len(batch)))
    
    def _batch_report(self, i, batch, loss, pbar, tb_writer, callbacks, **kwargs): 
        pbar.set_postfix({'Sinkhorn': loss.item()}); pbar.refresh()

    def _epoch_report(self, epoch, batch, loss, pbar, tb_writer, callbacks=[], report_interval=1, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('Sinkhorn', loss, epoch)
            else:
                print('[Epoch {}/{}] Sinkhorn: {:d}'.format(epoch, pbar.total, loss))
        for callback in callbacks:
            try: 
                callback(self, epoch=epoch, epochs=pbar.total, batch=batch, loss=loss, **kwargs)
            except:
                pass