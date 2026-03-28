import torch
from solution import gather_elements

def test_gather():
    x = torch.arange(24).view(2, 4, 3)
    indices = torch.tensor([[0, 2], [1, 3]])
    out = gather_elements(x, indices)
    assert out.shape == (2, 2, 3)
    assert torch.allclose(out[0, 1], x[0, 2])