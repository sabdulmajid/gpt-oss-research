import torch
from solution import find_large_grad_norms

def test_find_large_grad_norms():
    p1 = torch.nn.Parameter(torch.zeros(2))
    p1.grad = torch.tensor([3.0, 4.0])
    p2 = torch.nn.Parameter(torch.zeros(2))
    p2.grad = torch.tensor([1.0, 1.0])
    out = find_large_grad_norms([("a", p1), ("b", p2)], threshold=3.0)
    assert set(out) == {"a"}
    assert abs(out["a"] - 5.0) < 1e-6
