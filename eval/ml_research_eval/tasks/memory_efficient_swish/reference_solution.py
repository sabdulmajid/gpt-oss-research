import torch

def swish_forward(x):
    return x * torch.sigmoid(x)