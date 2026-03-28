Implement `pad_batch(examples, pad_token_id)` for a causal LM data collator.

Each example contains `input_ids` and `labels`.

Requirements:

- pad `input_ids` with `pad_token_id`
- pad `labels` with `-100`
- build an `attention_mask` of `1` for real tokens and `0` for padding
- return PyTorch tensors

