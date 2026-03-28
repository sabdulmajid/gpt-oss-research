import torch
from solution import focal_loss

def test_focal():
    logits = torch.tensor([0.0, 2.0, -2.0])
    targets = torch.tensor([1.0, 1.0, 0.0])
    loss = focal_loss(logits, targets)
    assert loss.dim() == 0