import math

def get_cosine_lr(step, total_steps, max_lr, min_lr):
    if step >= total_steps:
        return min_lr
    decay_ratio = step / total_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)