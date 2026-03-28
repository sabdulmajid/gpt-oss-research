import torch
import torch.nn.functional as F

def amp_forward_backward(model, optimizer, scaler, data, target):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer.zero_grad()
    with torch.autocast(device_type=device):
        output = model(data)
        loss = F.mse_loss(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()