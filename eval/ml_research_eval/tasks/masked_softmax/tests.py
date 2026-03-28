import torch
from solution import masked_softmax

def test_masked_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, False]])
    out = masked_softmax(logits, mask)
    assert out.shape == logits.shape
    assert torch.allclose(out[:, 2], torch.zeros(1))
    assert torch.allclose(out[:, :2].sum(dim=1), torch.ones(1))
