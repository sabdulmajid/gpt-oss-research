def detect_frozen_parameters(model):
    return [name for name, param in model.named_parameters() if not param.requires_grad]
