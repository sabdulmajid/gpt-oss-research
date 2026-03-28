Implement `find_large_grad_norms(named_parameters, threshold)`.

Requirements:
- `named_parameters` is an iterable of `(name, parameter)` pairs.
- Return a dictionary mapping parameter names to gradient L2 norms for gradients whose norm exceeds `threshold`.
- Ignore parameters with `grad is None`.
