import torch
from torch import nn

class _Flow():
    def f(self, x):
        raise NotImplementedError
    
    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample([batchSize])
        x = self.g(z)
        return x
    

class _RealNVP(nn.Module):
    """With the forward mode only."""
    def __init__(self, nets, nett, mask):
        super().__init__()
        self.register_buffer('mask', mask)
        self.t = nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = nn.ModuleList([nets() for _ in range(len(mask))])
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x
    
    def forward(self, z):
        return self.g(z)
    
    
class RealNVP(_RealNVP, _Flow):
    def __init__(self, nets, nett, mask, prior):
        super().__init__(nets, nett, mask)
        self.prior = prior

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = z * self.mask[i]
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J