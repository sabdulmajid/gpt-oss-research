Implement `masked_mean_pool(hidden_states, attention_mask)` in PyTorch.

Requirements:

- `hidden_states` has shape `[batch, seq, hidden]`
- `attention_mask` has shape `[batch, seq]`
- average only positions where the mask equals `1`
- preserve the hidden-state dtype
- avoid division by zero when an example is fully masked

