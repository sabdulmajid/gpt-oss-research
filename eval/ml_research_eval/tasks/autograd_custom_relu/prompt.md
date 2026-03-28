Implement a custom PyTorch autograd function for ReLU called `CustomReLU` and an entrypoint `apply_custom_relu(x)` that applies it.

Requirements:
- Forward pass should compute `max(0, x)`.
- Backward pass should return `1` if `x > 0` else `0` multiplied by grad_output.
- Do not use `torch.relu`.