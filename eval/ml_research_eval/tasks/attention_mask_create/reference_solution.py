import torch

def create_causal_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)