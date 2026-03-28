import torch

def batch_matrix_trace(matrices):
    return torch.diagonal(matrices, dim1=-2, dim2=-1).sum(-1)