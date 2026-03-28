import torch
from solution import topk_accuracy

def test_topk_accuracy():
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.4, 0.3, 0.5]])
    targets = torch.tensor([1, 0])
    assert topk_accuracy(logits, targets, k=1) == 0.5
    assert topk_accuracy(logits, targets, k=2) == 1.0
