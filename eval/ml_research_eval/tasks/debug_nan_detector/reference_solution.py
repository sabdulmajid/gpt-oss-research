import torch

def detect_nan_hook(module, grad_input, grad_output):
    for gi in grad_input:
        if gi is not None and torch.isnan(gi).any():
            raise ValueError("NaN in grad_input")
    for go in grad_output:
        if go is not None and torch.isnan(go).any():
            raise ValueError("NaN in grad_output")