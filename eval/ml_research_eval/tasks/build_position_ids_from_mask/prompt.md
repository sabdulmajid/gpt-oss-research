Implement `build_position_ids_from_mask(attention_mask)`.

Requirements:
- `attention_mask` is `[batch, seq_len]` with `1` for tokens and `0` for padding.
- Position ids should start at `0` for the first non-padding token in each row.
- Padding positions should be `0`.
