Implement `split_weight_decay_params(model)`.

Requirements:
- Return a tuple of two lists: `(decay_params, no_decay_params)`.
- `no_decay_params` should include all bias parameters and all 1D parameters such as LayerNorm scales.
- `decay_params` should include all other trainable parameters.
