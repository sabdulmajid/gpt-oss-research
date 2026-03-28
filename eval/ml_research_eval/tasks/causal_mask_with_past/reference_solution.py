import torch

def causal_mask_with_past(seq_len, past_len=0):
    total = past_len + seq_len
    current_positions = torch.arange(seq_len).unsqueeze(1)
    key_positions = torch.arange(total).unsqueeze(0)
    return key_positions > (current_positions + past_len)
