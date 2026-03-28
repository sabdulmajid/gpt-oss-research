import torch

def grad_scaler_unscale_and_clip(optimizer, scaler, parameters, max_norm):
    scaler.unscale_(optimizer)
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
