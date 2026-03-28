import torch
from solution import clip_grad_norm_manual

def test_clip():
    p1 = torch.nn.Parameter(torch.zeros(2))
    p1.grad = torch.tensor([3.0, 4.0]) # norm 5
    clip_grad_norm_manual([p1], 2.5)
    assert torch.allclose(p1.grad, torch.tensor([1.5, 2.0]))