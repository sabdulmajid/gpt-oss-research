def pack_checkpoint_state(model, optimizer, step):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
