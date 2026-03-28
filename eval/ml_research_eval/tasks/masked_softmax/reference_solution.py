import torch

def masked_softmax(logits, mask, dim=-1):
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(masked_logits, dim=dim)
    return probs.masked_fill(~mask, 0.0)
