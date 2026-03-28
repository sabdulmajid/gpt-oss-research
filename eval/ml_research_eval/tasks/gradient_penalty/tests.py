import torch
import torch.nn as nn
from solution import compute_gradient_penalty

def test_gp():
    critic = nn.Linear(2, 1)
    real = torch.randn(4, 2)
    fake = torch.randn(4, 2)
    gp = compute_gradient_penalty(critic, real, fake)
    assert gp.dim() == 0 and gp.requires_grad