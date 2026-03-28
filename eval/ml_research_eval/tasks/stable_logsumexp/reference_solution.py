import torch

def stable_logsumexp(x, dim=-1):
    max_x, _ = x.max(dim=dim, keepdim=True)
    shifted = x - max_x
    return (max_x + shifted.exp().sum(dim=dim, keepdim=True).log()).squeeze(dim)
