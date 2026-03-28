import torch

def find_large_grad_norms(named_parameters, threshold):
    result = {}
    for name, param in named_parameters:
        if param.grad is None:
            continue
        norm = torch.norm(param.grad.detach(), p=2).item()
        if norm > threshold:
            result[name] = norm
    return result
