import torch
from solution import shift_labels_for_causal_lm

def test_shift_labels_for_causal_lm():
    labels = torch.tensor([[1, 2, 3]])
    out = shift_labels_for_causal_lm(labels)
    assert torch.equal(out, torch.tensor([[2, 3, -100]]))
