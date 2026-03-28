import torch.nn as nn
from solution import count_trainable_parameters

def test_count():
    model = nn.Linear(10, 5)
    model.bias.requires_grad = False
    assert count_trainable_parameters(model) == 50