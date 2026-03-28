import torch

def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for param_m, param_ema in zip(model.parameters(), ema_model.parameters()):
            param_ema.data.mul_(decay).add_(param_m.data, alpha=1 - decay)