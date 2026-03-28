import torch
from solution import batched_outer_product

def test_batched_outer_product():
    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[3.0, 4.0, 5.0]])
    out = batched_outer_product(x, y)
    expected = torch.tensor([[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]])
    assert torch.allclose(out, expected)
