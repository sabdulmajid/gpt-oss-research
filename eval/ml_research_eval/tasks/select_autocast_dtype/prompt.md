Implement `select_autocast_dtype(device_type)`.

Requirements:
- Return `torch.bfloat16` for `cuda` and `cpu`.
- Return `torch.float32` for any other device type.
