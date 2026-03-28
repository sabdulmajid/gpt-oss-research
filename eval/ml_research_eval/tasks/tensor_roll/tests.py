import torch
from solution import shift_sequence

def test_roll():
    x = torch.arange(6).view(1, 3, 2)
    out = shift_sequence(x, 1)
    assert torch.allclose(out[0, 0], x[0, 2])