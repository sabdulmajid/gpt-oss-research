import torch
from solution import build_position_ids_from_mask

def test_build_position_ids_from_mask():
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    out = build_position_ids_from_mask(mask)
    expected = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 0]])
    assert torch.equal(out, expected)
