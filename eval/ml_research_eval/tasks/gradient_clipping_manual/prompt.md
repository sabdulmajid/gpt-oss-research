Implement `clip_grad_norm_manual(parameters, max_norm)`.

Requirements:
- Compute the total L2 norm of the gradients of all parameters.
- If the norm exceeds `max_norm`, scale all gradients down by `max_norm / total_norm`.
- Do not use `torch.nn.utils.clip_grad_norm_`.