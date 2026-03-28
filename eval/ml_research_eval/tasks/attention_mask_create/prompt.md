Implement `create_causal_mask(seq_len)`.

Requirements:
- Return a boolean tensor of shape `[seq_len, seq_len]`.
- Lower triangular elements (including diagonal) should be `False` (not masked).
- Upper triangular elements should be `True` (masked).