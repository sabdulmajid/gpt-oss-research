import torch.nn as nn
from solution import split_weight_decay_params

def test_split():
    model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    decay, no_decay = split_weight_decay_params(model)
    assert len(decay) == 1 # Linear weight
    assert len(no_decay) == 3 # Linear bias, LN weight, LN bias