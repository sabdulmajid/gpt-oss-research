import torch
from solution import smooth_labels

def test_smooth():
    targets = torch.tensor([0, 2])
    out = smooth_labels(targets, 3, 0.3)
    expected = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    assert torch.allclose(out, expected)