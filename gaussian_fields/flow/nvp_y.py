import torch
from torch import nn


class RealNVP(nn.Module):
    def __init__(self, net_s, net_t, masks, prior, preprocess_fn=None):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.masks = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList(
            [net_t() for _ in range(len(masks))]
        )
        self.s = torch.nn.ModuleList(
            [net_s() for _ in range(len(masks))]
        )
        self.preprocess_fn = preprocess_fn
        
    def reverse(self, z, y):
        if self.preprocess_fn is not None:
            _, y = self.preprocess_fn(torch.zeros_like(y), y)
        x = z
        for i in range(len(self.t)):
            x_ = x * self.masks[i]
            # NOTE: this loops through activations as well? no point concatenating there? No, loops through each layer s, t networks
            s = self.s[i](torch.cat([x_, y], 1)) * (1 - self.masks[i])
            t = self.t[i](torch.cat([x_, y], 1)) * (1 - self.masks[i])
            x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)
        return x

    def forward(self, x, y):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.masks[i] * z
            s = self.s[i](torch.cat([z_, y], 1)) * (1 - self.masks[i])
            t = self.t[i](torch.cat([z_, y], 1)) * (1 - self.masks[i])
            z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self, x, y):
        if self.preprocess_fn is not None:
            x, y = self.preprocess_fn(x, y)
        z, logp = self.forward(x, y)
        return self.prior.log_prob(z) + logp
        
    def sample(self, n, y): 
        z = self.prior.sample((n,))
        logp = self.prior.log_prob(z)
        x = self.reverse(z, y)
        return x