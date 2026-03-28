Implement `update_ema(model, ema_model, decay)`.

Requirements:
- Update the parameters of `ema_model` using the parameters of `model`.
- Formula: `ema_param = decay * ema_param + (1 - decay) * model_param`.
- Use `torch.no_grad()`.