def prepare_components(accelerator, model, optimizer, dataloader):
    return accelerator.prepare(model, optimizer, dataloader)