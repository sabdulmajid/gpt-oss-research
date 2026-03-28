import math

import torch


def clip_grad_norm_and_report(parameters, max_norm):
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return 0.0
    total_norm_sq = 0.0
    for grad in grads:
        grad_norm = torch.linalg.vector_norm(grad.detach(), ord=2)
        total_norm_sq += float(grad_norm.item() ** 2)
    total_norm = math.sqrt(total_norm_sq)
    clip_coef = min(1.0, float(max_norm) / (total_norm + 1e-12))
    for grad in grads:
        grad.mul_(clip_coef)
    return float(total_norm)

