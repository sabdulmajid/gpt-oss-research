import torch.nn as nn
from solution import freeze_module_params

def test_freeze_module_params():
    module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    frozen = freeze_module_params(module)
    assert frozen == 4
    assert all(not param.requires_grad for param in module.parameters())
