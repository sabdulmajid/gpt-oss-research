import torch

def select_autocast_dtype(device_type):
    if device_type in {"cuda", "cpu"}:
        return torch.bfloat16
    return torch.float32
