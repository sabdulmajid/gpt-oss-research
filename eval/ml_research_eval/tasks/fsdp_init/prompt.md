Implement `wrap_fsdp(model)` to wrap a PyTorch model in FullyShardedDataParallel.

Requirements:
- Only wrap the model if `torch.distributed.is_initialized()`.
- Return the wrapped model if initialized, otherwise return the original model.