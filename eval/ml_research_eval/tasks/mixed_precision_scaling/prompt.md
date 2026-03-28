Implement `amp_forward_backward(model, optimizer, scaler, data, target)` using PyTorch AMP.

Requirements:
- Use `torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')`.
- Compute MSE loss.
- Scale the loss and call backward.
- Step the optimizer using the scaler.
- Update the scaler.