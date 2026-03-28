def accelerator_gather_for_metrics(accelerator, tensor):
    gather = getattr(accelerator, "gather_for_metrics", None)
    if gather is None:
        return tensor
    return gather(tensor)
