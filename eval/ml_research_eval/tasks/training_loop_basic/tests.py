import torch
import torch.nn as nn
import torch.optim as optim
from solution import train_one_step

def test_train_one_step():
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    data = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0]])
    loss = train_one_step(model, optimizer, data, target)
    assert isinstance(loss, float)