import torch

def smooth_labels(targets, num_classes, smoothing):
    B = targets.size(0)
    smoothed = torch.full((B, num_classes), smoothing / num_classes, device=targets.device)
    smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing + smoothing / num_classes)
    return smoothed