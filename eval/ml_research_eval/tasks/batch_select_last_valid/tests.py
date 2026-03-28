import torch
from solution import select_last_valid

def test_select_last_valid():
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 1.0], [5.0, 1.0], [6.0, 1.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = select_last_valid(hidden, mask)
    expected = torch.tensor([[2.0, 0.0], [6.0, 1.0]])
    assert torch.allclose(out, expected)
