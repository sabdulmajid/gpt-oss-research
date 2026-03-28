Implement `select_last_valid(hidden_states, attention_mask)`.

Requirements:
- `hidden_states` has shape `[batch, seq_len, hidden_dim]`.
- `attention_mask` has shape `[batch, seq_len]` with `1` for valid tokens and `0` for padding.
- Return a tensor of shape `[batch, hidden_dim]` containing the last valid token state for each sequence.
