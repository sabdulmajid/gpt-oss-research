import torch
import torch.nn as nn
from solution import wrap_fsdp

def test_fsdp_init():
    model = nn.Linear(2, 2)
    wrapped = wrap_fsdp(model)
    assert isinstance(wrapped, nn.Linear)