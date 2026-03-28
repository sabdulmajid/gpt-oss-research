import torch

def shift_sequence(x, shift):
    return torch.roll(x, shifts=shift, dims=1)