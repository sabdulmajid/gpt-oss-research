import torch
from solution import accumulate_rows

def test_accumulate_rows():
    values = torch.tensor([[1.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
    indices = torch.tensor([0, 1, 0])
    out = accumulate_rows(values, indices, num_rows=3)
    expected = torch.tensor([[4.0, 5.0], [2.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(out, expected)
