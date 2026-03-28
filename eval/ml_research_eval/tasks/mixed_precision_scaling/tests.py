import torch
import torch.nn as nn
import torch.optim as optim
from solution import amp_forward_backward

def test_amp():
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler()
    data = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0]])
    loss = amp_forward_backward(model, optimizer, scaler, data, target)
    assert isinstance(loss, float)