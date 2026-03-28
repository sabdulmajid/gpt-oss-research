Implement `grad_scaler_unscale_and_clip(optimizer, scaler, parameters, max_norm)`.

Requirements:
- Call `scaler.unscale_(optimizer)`.
- Clip gradients with `torch.nn.utils.clip_grad_norm_`.
- Return the pre-clipped total norm.
