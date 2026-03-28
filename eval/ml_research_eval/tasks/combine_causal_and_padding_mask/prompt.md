Implement `combine_causal_and_padding_mask(attention_mask)`.

Requirements:
- `attention_mask` has shape `[batch, seq_len]`.
- Return a boolean mask of shape `[batch, 1, seq_len, seq_len]`.
- A position should be masked if it is in the future or if the key token is padding.
