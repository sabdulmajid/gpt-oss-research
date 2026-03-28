import torch
from solution import swish_forward

def test_swish():
    x = torch.tensor([0.0, 1.0])
    out = swish_forward(x)
    assert torch.allclose(out[0], torch.tensor(0.0))
    assert out[1] > 0.5