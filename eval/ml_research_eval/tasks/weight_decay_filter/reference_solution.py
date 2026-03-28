def split_weight_decay_params(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay
