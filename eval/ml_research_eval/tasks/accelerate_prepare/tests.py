from accelerate import Accelerator
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from solution import prepare_components

def test_prepare():
    accelerator = Accelerator()
    model = nn.Linear(2, 2)
    opt = optim.Adam(model.parameters())
    dl = DataLoader(TensorDataset(torch.randn(10, 2), torch.randn(10, 2)))
    m, o, d = prepare_components(accelerator, model, opt, dl)
    assert m is not None