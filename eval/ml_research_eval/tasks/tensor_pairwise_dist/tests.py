import torch
from solution import pairwise_l2_dist

def test_dist():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    y = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    out = pairwise_l2_dist(x, y)
    expected = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    assert torch.allclose(out, expected, atol=1e-5)