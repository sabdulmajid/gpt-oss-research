Implement `detect_frozen_parameters(model)`.

Requirements:
- Return a list of parameter names whose `requires_grad` is `False`.
- Preserve the order from `model.named_parameters()`.
