def maybe_enable_gradient_checkpointing(model):
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if enable is None:
        return False
    enable()
    return True
