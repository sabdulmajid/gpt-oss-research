import torch

def expand_attention_mask(attention_mask, dtype):
    mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - mask) * torch.finfo(dtype).min
