import torch.nn as nn
from solution import detect_frozen_parameters

def test_detect_frozen_parameters():
    model = nn.Linear(2, 2)
    model.bias.requires_grad = False
    assert detect_frozen_parameters(model) == ["bias"]
