Implement `linear_warmup_lr(step, warmup_steps, max_lr)`.

Requirements:
- Return `max_lr * step / warmup_steps` during warmup.
- Return `max_lr` once `step >= warmup_steps`.
- If `warmup_steps <= 0`, return `max_lr`.
