def linear_warmup_lr(step, warmup_steps, max_lr):
    if warmup_steps <= 0 or step >= warmup_steps:
        return max_lr
    return max_lr * step / warmup_steps
