def accelerator_backward_loss(accelerator, loss):
    accelerator.backward(loss)
    return float(loss.detach().item())
