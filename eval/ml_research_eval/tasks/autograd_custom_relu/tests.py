import torch
from solution import apply_custom_relu

def test_custom_relu():
    x = torch.tensor([-1.0, 1.0, 2.0], requires_grad=True)
    y = apply_custom_relu(x)
    assert torch.allclose(y, torch.tensor([0.0, 1.0, 2.0]))
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([0.0, 1.0, 1.0]))