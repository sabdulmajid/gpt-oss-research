import torch.nn as nn
import torch
from solution import update_ema

def test_ema():
    m = nn.Linear(1, 1, bias=False)
    ema = nn.Linear(1, 1, bias=False)
    m.weight.data.fill_(1.0)
    ema.weight.data.fill_(0.0)
    update_ema(m, ema, 0.9)
    assert torch.allclose(ema.weight.data, torch.tensor([[0.1]]))