Implement `masked_softmax(logits, mask, dim=-1)`.

Requirements:
- `mask` is a boolean tensor with `True` for valid positions and `False` for masked positions.
- Return softmax probabilities over the valid positions only.
- Masked positions must have probability `0`.
