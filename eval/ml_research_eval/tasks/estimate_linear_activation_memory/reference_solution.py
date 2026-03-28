def estimate_linear_activation_memory(batch_size, seq_len, hidden_size, dtype_bytes=2):
    return int(batch_size * seq_len * hidden_size * dtype_bytes)
