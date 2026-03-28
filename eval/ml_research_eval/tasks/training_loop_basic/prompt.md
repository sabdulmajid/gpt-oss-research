Implement `train_one_step(model, optimizer, data, target)` in PyTorch.

Requirements:
- Compute the MSE loss between `model(data)` and `target`.
- Perform a backward pass.
- Step the optimizer.
- Zero the gradients.
- Return the loss item as a float.