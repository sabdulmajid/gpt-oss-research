import torch
from solution import expand_attention_mask

def test_expand_attention_mask():
    mask = torch.tensor([[1, 0]])
    out = expand_attention_mask(mask, torch.float32)
    assert out.shape == (1, 1, 1, 2)
    assert out[0, 0, 0, 0].item() == 0.0
    assert out[0, 0, 0, 1].item() < -1e30
