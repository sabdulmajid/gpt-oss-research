import torch

def combine_causal_and_padding_mask(attention_mask):
    batch, seq_len = attention_mask.shape
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    padding = attention_mask[:, None, None, :] == 0
    return causal.view(1, 1, seq_len, seq_len) | padding
