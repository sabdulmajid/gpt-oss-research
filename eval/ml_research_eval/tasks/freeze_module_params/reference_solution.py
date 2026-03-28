def freeze_module_params(module):
    frozen = 0
    for param in module.parameters():
        if param.requires_grad:
            param.requires_grad = False
            frozen += 1
    return frozen
