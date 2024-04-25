import torch
from torch import nn


class RealNVP(nn.Module):
    def __init__(self, net_s, net_t, masks, prior):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.masks = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([net_t() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([net_s() for _ in range(len(masks))])
        
    def reverse(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.masks[i]
            s = self.s[i](x_) * (1 - self.masks[i])
            t = self.t[i](x_) * (1 - self.masks[i])
            x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)
        return x

    def forward(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.masks[i] * z
            s = self.s[i](z_) * (1 - self.masks[i])
            t = self.t[i](z_) * (1 - self.masks[i])
            z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.forward(x)
        return self.prior.log_prob(z) + logp # sum z logprob dim=1?
        
    def sample(self, n): 
        z = self.prior.sample((n, 1))
        logp = self.prior.log_prob(z)
        x = self.reverse(z)
        return x