def tokens_per_second(num_tokens, elapsed_sec):
    if elapsed_sec <= 0:
        return 0.0
    return float(num_tokens) / float(elapsed_sec)
