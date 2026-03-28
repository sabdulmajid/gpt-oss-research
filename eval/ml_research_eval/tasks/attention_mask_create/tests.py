import torch
from solution import create_causal_mask

def test_mask():
    mask = create_causal_mask(3)
    expected = torch.tensor([[False, True, True],
                             [False, False, True],
                             [False, False, False]])
    assert torch.equal(mask, expected)