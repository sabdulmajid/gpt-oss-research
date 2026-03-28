Implement `gather_elements(x, indices)` in PyTorch.

Requirements:
- `x` has shape `[B, N, C]`.
- `indices` has shape `[B, K]`.
- Gather the elements from `x` along the `N` dimension using `indices`.
- Return a tensor of shape `[B, K, C]`.