Implement `clip_grad_norm_and_report(parameters, max_norm)` in PyTorch.

Requirements:

- compute the global L2 norm of all parameter gradients before clipping
- clip gradients in place to `max_norm`
- return the pre-clipping norm as a Python float
- ignore parameters whose `.grad` is `None`

