import torch

def shift_labels_for_causal_lm(labels, ignore_index=-100):
    shifted = torch.full_like(labels, ignore_index)
    shifted[..., :-1] = labels[..., 1:]
    return shifted
