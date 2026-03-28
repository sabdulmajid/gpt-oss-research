Implement `expand_attention_mask(attention_mask, dtype)`.

Requirements:
- Input mask has shape `[batch, seq_len]`.
- Return a tensor of shape `[batch, 1, 1, seq_len]`.
- Valid tokens should map to `0`.
- Masked tokens should map to a very negative value representable in `dtype`.
