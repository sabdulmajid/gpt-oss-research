import torch

def assert_finite_loss(loss):
    if not torch.isfinite(loss):
        raise ValueError("loss is not finite")
    return float(loss.item())
