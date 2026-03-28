import torch

def gather_elements(x, indices):
    B, K = indices.shape
    C = x.shape[2]
    # Expand indices to match C dimension: [B, K, C]
    expanded_indices = indices.unsqueeze(-1).expand(B, K, C)
    return torch.gather(x, 1, expanded_indices)