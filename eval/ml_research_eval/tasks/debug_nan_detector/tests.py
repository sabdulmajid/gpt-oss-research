import torch
import torch.nn as nn
from solution import detect_nan_hook

def test_nan_hook():
    module = nn.Linear(2, 2)
    grad_input = (torch.tensor([1.0, float('nan')]),)
    grad_output = (torch.tensor([1.0, 1.0]),)
    try:
        detect_nan_hook(module, grad_input, grad_output)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass