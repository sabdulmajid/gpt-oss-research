Implement `pairwise_l2_dist(x, y)`.

Requirements:
- `x`: `[N, D]`, `y`: `[M, D]`.
- Return a tensor of shape `[N, M]` containing the squared L2 distances between each pair of vectors in `x` and `y`.
- Do not use `torch.cdist`.