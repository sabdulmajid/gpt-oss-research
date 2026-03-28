import torch
import torch.nn.functional as F

def train_one_step(model, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()