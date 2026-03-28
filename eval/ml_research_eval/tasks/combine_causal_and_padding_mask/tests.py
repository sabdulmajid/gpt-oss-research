import torch
from solution import combine_causal_and_padding_mask

def test_combine_causal_and_padding_mask():
    mask = torch.tensor([[1, 1, 0]])
    out = combine_causal_and_padding_mask(mask)
    assert out.shape == (1, 1, 3, 3)
    assert out[0, 0, 0, 1].item() is True
    assert out[0, 0, 2, 2].item() is True
    assert out[0, 0, 1, 0].item() is False
