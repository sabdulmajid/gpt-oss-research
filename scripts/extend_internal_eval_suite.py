from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = ROOT / "eval" / "ml_research_eval" / "tasks"


EXISTING_BUCKETS = {
    "accelerate_prepare": "transformers_accelerate_fsdp_mixed_precision",
    "attention_mask_create": "tensor_api",
    "autograd_clip_grad": "debugging_performance",
    "autograd_custom_relu": "debugging_performance",
    "cosine_annealing_lr": "training_pipeline",
    "data_collator_pad_labels": "training_pipeline",
    "debug_nan_detector": "debugging_performance",
    "ema_update": "training_pipeline",
    "focal_loss": "training_pipeline",
    "fsdp_init": "transformers_accelerate_fsdp_mixed_precision",
    "gradient_clipping_manual": "training_pipeline",
    "gradient_penalty": "debugging_performance",
    "label_smoothing": "tensor_api",
    "layer_norm_manual": "tensor_api",
    "memory_efficient_swish": "debugging_performance",
    "mixed_precision_scaling": "transformers_accelerate_fsdp_mixed_precision",
    "model_parameter_count": "debugging_performance",
    "tensor_contraction_1": "tensor_api",
    "tensor_gather": "tensor_api",
    "tensor_pairwise_dist": "tensor_api",
    "tensor_roll": "tensor_api",
    "training_detach_metrics": "training_pipeline",
    "training_loop_basic": "training_pipeline",
    "transformer_masked_mean_pool": "tensor_api",
    "weight_decay_filter": "training_pipeline",
}


PATCHED_EXISTING_TASKS = {
    "weight_decay_filter": {
        "prompt": dedent(
            """
            Implement `split_weight_decay_params(model)`.

            Requirements:
            - Return a tuple of two lists: `(decay_params, no_decay_params)`.
            - `no_decay_params` should include all bias parameters and all 1D parameters such as LayerNorm scales.
            - `decay_params` should include all other trainable parameters.
            """
        ).strip()
        + "\n",
        "reference_solution": dedent(
            """
            def split_weight_decay_params(model):
                decay = []
                no_decay = []
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name.endswith("bias") or param.ndim == 1:
                        no_decay.append(param)
                    else:
                        decay.append(param)
                return decay, no_decay
            """
        ).lstrip(),
    }
}


def _task(
    *,
    task_id: str,
    difficulty: str,
    topic: str,
    entrypoint: str,
    benchmark_bucket: str,
    prompt: str,
    reference_solution: str,
    tests: str,
) -> dict[str, object]:
    return {
        "metadata": {
            "task_id": task_id,
            "difficulty": difficulty,
            "topic": topic,
            "entrypoint": entrypoint,
            "benchmark_bucket": benchmark_bucket,
            "timeout_sec": 30,
        },
        "prompt": dedent(prompt).strip() + "\n",
        "reference_solution": dedent(reference_solution).lstrip(),
        "tests": dedent(tests).lstrip(),
    }


NEW_TASKS = {
    "batch_select_last_valid": _task(
        task_id="batch_select_last_valid",
        difficulty="medium",
        topic="pytorch_tensor_ops",
        entrypoint="select_last_valid",
        benchmark_bucket="tensor_api",
        prompt="""
        Implement `select_last_valid(hidden_states, attention_mask)`.

        Requirements:
        - `hidden_states` has shape `[batch, seq_len, hidden_dim]`.
        - `attention_mask` has shape `[batch, seq_len]` with `1` for valid tokens and `0` for padding.
        - Return a tensor of shape `[batch, hidden_dim]` containing the last valid token state for each sequence.
        """,
        reference_solution="""
        import torch

        def select_last_valid(hidden_states, attention_mask):
            lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            gather_index = lengths.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
            return hidden_states.gather(1, gather_index).squeeze(1)
        """,
        tests="""
        import torch
        from solution import select_last_valid

        def test_select_last_valid():
            hidden = torch.tensor(
                [
                    [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                    [[4.0, 1.0], [5.0, 1.0], [6.0, 1.0]],
                ]
            )
            mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
            out = select_last_valid(hidden, mask)
            expected = torch.tensor([[2.0, 0.0], [6.0, 1.0]])
            assert torch.allclose(out, expected)
        """,
    ),
    "masked_softmax": _task(
        task_id="masked_softmax",
        difficulty="medium",
        topic="pytorch_tensor_ops",
        entrypoint="masked_softmax",
        benchmark_bucket="tensor_api",
        prompt="""
        Implement `masked_softmax(logits, mask, dim=-1)`.

        Requirements:
        - `mask` is a boolean tensor with `True` for valid positions and `False` for masked positions.
        - Return softmax probabilities over the valid positions only.
        - Masked positions must have probability `0`.
        """,
        reference_solution="""
        import torch

        def masked_softmax(logits, mask, dim=-1):
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            probs = torch.softmax(masked_logits, dim=dim)
            return probs.masked_fill(~mask, 0.0)
        """,
        tests="""
        import torch
        from solution import masked_softmax

        def test_masked_softmax():
            logits = torch.tensor([[1.0, 2.0, 3.0]])
            mask = torch.tensor([[True, True, False]])
            out = masked_softmax(logits, mask)
            assert out.shape == logits.shape
            assert torch.allclose(out[:, 2], torch.zeros(1))
            assert torch.allclose(out[:, :2].sum(dim=1), torch.ones(1))
        """,
    ),
    "accumulate_rows": _task(
        task_id="accumulate_rows",
        difficulty="medium",
        topic="pytorch_tensor_ops",
        entrypoint="accumulate_rows",
        benchmark_bucket="tensor_api",
        prompt="""
        Implement `accumulate_rows(values, row_indices, num_rows)`.

        Requirements:
        - `values` has shape `[num_items, dim]`.
        - `row_indices` has shape `[num_items]`.
        - Sum each row of `values` into an output tensor of shape `[num_rows, dim]`.
        - Use PyTorch tensor operations instead of Python loops.
        """,
        reference_solution="""
        import torch

        def accumulate_rows(values, row_indices, num_rows):
            out = torch.zeros(num_rows, values.size(-1), dtype=values.dtype, device=values.device)
            out.index_add_(0, row_indices.long(), values)
            return out
        """,
        tests="""
        import torch
        from solution import accumulate_rows

        def test_accumulate_rows():
            values = torch.tensor([[1.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
            indices = torch.tensor([0, 1, 0])
            out = accumulate_rows(values, indices, num_rows=3)
            expected = torch.tensor([[4.0, 5.0], [2.0, 0.0], [0.0, 0.0]])
            assert torch.allclose(out, expected)
        """,
    ),
    "batched_outer_product": _task(
        task_id="batched_outer_product",
        difficulty="easy",
        topic="pytorch_tensor_ops",
        entrypoint="batched_outer_product",
        benchmark_bucket="tensor_api",
        prompt="""
        Implement `batched_outer_product(x, y)`.

        Requirements:
        - `x` has shape `[batch, n]`.
        - `y` has shape `[batch, m]`.
        - Return the batched outer product with shape `[batch, n, m]`.
        """,
        reference_solution="""
        def batched_outer_product(x, y):
            return x.unsqueeze(-1) * y.unsqueeze(-2)
        """,
        tests="""
        import torch
        from solution import batched_outer_product

        def test_batched_outer_product():
            x = torch.tensor([[1.0, 2.0]])
            y = torch.tensor([[3.0, 4.0, 5.0]])
            out = batched_outer_product(x, y)
            expected = torch.tensor([[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]])
            assert torch.allclose(out, expected)
        """,
    ),
    "stable_logsumexp": _task(
        task_id="stable_logsumexp",
        difficulty="medium",
        topic="pytorch_tensor_ops",
        entrypoint="stable_logsumexp",
        benchmark_bucket="tensor_api",
        prompt="""
        Implement `stable_logsumexp(x, dim=-1)`.

        Requirements:
        - Compute `log(sum(exp(x)))` in a numerically stable way.
        - Do not call `torch.logsumexp`.
        """,
        reference_solution="""
        import torch

        def stable_logsumexp(x, dim=-1):
            max_x, _ = x.max(dim=dim, keepdim=True)
            shifted = x - max_x
            return (max_x + shifted.exp().sum(dim=dim, keepdim=True).log()).squeeze(dim)
        """,
        tests="""
        import torch
        from solution import stable_logsumexp

        def test_stable_logsumexp():
            x = torch.tensor([[1000.0, 1001.0]])
            out = stable_logsumexp(x, dim=1)
            expected = torch.logsumexp(x, dim=1)
            assert torch.allclose(out, expected, atol=1e-5)
        """,
    ),
    "linear_warmup_lr": _task(
        task_id="linear_warmup_lr",
        difficulty="easy",
        topic="model_training_loops",
        entrypoint="linear_warmup_lr",
        benchmark_bucket="training_pipeline",
        prompt="""
        Implement `linear_warmup_lr(step, warmup_steps, max_lr)`.

        Requirements:
        - Return `max_lr * step / warmup_steps` during warmup.
        - Return `max_lr` once `step >= warmup_steps`.
        - If `warmup_steps <= 0`, return `max_lr`.
        """,
        reference_solution="""
        def linear_warmup_lr(step, warmup_steps, max_lr):
            if warmup_steps <= 0 or step >= warmup_steps:
                return max_lr
            return max_lr * step / warmup_steps
        """,
        tests="""
        from solution import linear_warmup_lr

        def test_linear_warmup_lr():
            assert linear_warmup_lr(0, 10, 1.0) == 0.0
            assert linear_warmup_lr(5, 10, 1.0) == 0.5
            assert linear_warmup_lr(10, 10, 1.0) == 1.0
            assert linear_warmup_lr(5, 0, 1.0) == 1.0
        """,
    ),
    "topk_accuracy": _task(
        task_id="topk_accuracy",
        difficulty="easy",
        topic="model_training_loops",
        entrypoint="topk_accuracy",
        benchmark_bucket="training_pipeline",
        prompt="""
        Implement `topk_accuracy(logits, targets, k=1)`.

        Requirements:
        - `logits` has shape `[batch, num_classes]`.
        - `targets` has shape `[batch]`.
        - Return the top-k accuracy as a float in `[0, 1]`.
        """,
        reference_solution="""
        def topk_accuracy(logits, targets, k=1):
            topk = logits.topk(k, dim=1).indices
            correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
            return correct.float().mean().item()
        """,
        tests="""
        import torch
        from solution import topk_accuracy

        def test_topk_accuracy():
            logits = torch.tensor([[0.1, 0.9, 0.0], [0.2, 0.3, 0.5]])
            targets = torch.tensor([1, 0])
            assert topk_accuracy(logits, targets, k=1) == 0.5
            assert topk_accuracy(logits, targets, k=2) == 1.0
        """,
    ),
    "collate_pad_to_multiple": _task(
        task_id="collate_pad_to_multiple",
        difficulty="medium",
        topic="data_pipeline_and_collation",
        entrypoint="collate_pad_to_multiple",
        benchmark_bucket="training_pipeline",
        prompt="""
        Implement `collate_pad_to_multiple(batch, pad_value=0, multiple_of=8)`.

        Requirements:
        - `batch` is a list of 1D tensors with different lengths.
        - Pad them into a 2D tensor `[batch, padded_length]`.
        - Round `padded_length` up to the nearest multiple of `multiple_of`.
        """,
        reference_solution="""
        import torch

        def collate_pad_to_multiple(batch, pad_value=0, multiple_of=8):
            max_len = max(item.numel() for item in batch)
            padded_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of
            out = torch.full((len(batch), padded_len), pad_value, dtype=batch[0].dtype)
            for index, item in enumerate(batch):
                out[index, : item.numel()] = item
            return out
        """,
        tests="""
        import torch
        from solution import collate_pad_to_multiple

        def test_collate_pad_to_multiple():
            batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
            out = collate_pad_to_multiple(batch, pad_value=-1, multiple_of=4)
            assert out.shape == (2, 4)
            assert out[1, 2:].tolist() == [-1, -1]
        """,
    ),
    "pack_checkpoint_state": _task(
        task_id="pack_checkpoint_state",
        difficulty="easy",
        topic="model_training_loops",
        entrypoint="pack_checkpoint_state",
        benchmark_bucket="training_pipeline",
        prompt="""
        Implement `pack_checkpoint_state(model, optimizer, step)`.

        Requirements:
        - Return a dictionary containing the model state dict, optimizer state dict, and current step.
        - Use keys `model`, `optimizer`, and `step`.
        """,
        reference_solution="""
        def pack_checkpoint_state(model, optimizer, step):
            return {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
        """,
        tests="""
        import torch.nn as nn
        import torch.optim as optim
        from solution import pack_checkpoint_state

        def test_pack_checkpoint_state():
            model = nn.Linear(2, 1)
            optimizer = optim.SGD(model.parameters(), lr=0.1)
            state = pack_checkpoint_state(model, optimizer, step=7)
            assert set(state) == {"model", "optimizer", "step"}
            assert state["step"] == 7
        """,
    ),
    "freeze_module_params": _task(
        task_id="freeze_module_params",
        difficulty="easy",
        topic="model_training_loops",
        entrypoint="freeze_module_params",
        benchmark_bucket="training_pipeline",
        prompt="""
        Implement `freeze_module_params(module)`.

        Requirements:
        - Set `requires_grad = False` for every parameter in `module`.
        - Return the number of parameters that were frozen.
        """,
        reference_solution="""
        def freeze_module_params(module):
            frozen = 0
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += 1
            return frozen
        """,
        tests="""
        import torch.nn as nn
        from solution import freeze_module_params

        def test_freeze_module_params():
            module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
            frozen = freeze_module_params(module)
            assert frozen == 4
            assert all(not param.requires_grad for param in module.parameters())
        """,
    ),
    "shift_labels_for_causal_lm": _task(
        task_id="shift_labels_for_causal_lm",
        difficulty="easy",
        topic="transformers",
        entrypoint="shift_labels_for_causal_lm",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `shift_labels_for_causal_lm(labels, ignore_index=-100)`.

        Requirements:
        - Return a tensor with the same shape as `labels`.
        - Each position should contain the next token label.
        - The last position should be filled with `ignore_index`.
        """,
        reference_solution="""
        import torch

        def shift_labels_for_causal_lm(labels, ignore_index=-100):
            shifted = torch.full_like(labels, ignore_index)
            shifted[..., :-1] = labels[..., 1:]
            return shifted
        """,
        tests="""
        import torch
        from solution import shift_labels_for_causal_lm

        def test_shift_labels_for_causal_lm():
            labels = torch.tensor([[1, 2, 3]])
            out = shift_labels_for_causal_lm(labels)
            assert torch.equal(out, torch.tensor([[2, 3, -100]]))
        """,
    ),
    "build_position_ids_from_mask": _task(
        task_id="build_position_ids_from_mask",
        difficulty="medium",
        topic="transformers",
        entrypoint="build_position_ids_from_mask",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `build_position_ids_from_mask(attention_mask)`.

        Requirements:
        - `attention_mask` is `[batch, seq_len]` with `1` for tokens and `0` for padding.
        - Position ids should start at `0` for the first non-padding token in each row.
        - Padding positions should be `0`.
        """,
        reference_solution="""
        def build_position_ids_from_mask(attention_mask):
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            return position_ids
        """,
        tests="""
        import torch
        from solution import build_position_ids_from_mask

        def test_build_position_ids_from_mask():
            mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
            out = build_position_ids_from_mask(mask)
            expected = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 0]])
            assert torch.equal(out, expected)
        """,
    ),
    "expand_attention_mask": _task(
        task_id="expand_attention_mask",
        difficulty="medium",
        topic="transformers",
        entrypoint="expand_attention_mask",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `expand_attention_mask(attention_mask, dtype)`.

        Requirements:
        - Input mask has shape `[batch, seq_len]`.
        - Return a tensor of shape `[batch, 1, 1, seq_len]`.
        - Valid tokens should map to `0`.
        - Masked tokens should map to a very negative value representable in `dtype`.
        """,
        reference_solution="""
        import torch

        def expand_attention_mask(attention_mask, dtype):
            mask = attention_mask[:, None, None, :].to(dtype)
            return (1.0 - mask) * torch.finfo(dtype).min
        """,
        tests="""
        import torch
        from solution import expand_attention_mask

        def test_expand_attention_mask():
            mask = torch.tensor([[1, 0]])
            out = expand_attention_mask(mask, torch.float32)
            assert out.shape == (1, 1, 1, 2)
            assert out[0, 0, 0, 0].item() == 0.0
            assert out[0, 0, 0, 1].item() < -1e30
        """,
    ),
    "causal_mask_with_past": _task(
        task_id="causal_mask_with_past",
        difficulty="medium",
        topic="transformers",
        entrypoint="causal_mask_with_past",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `causal_mask_with_past(seq_len, past_len=0)`.

        Requirements:
        - Return a boolean mask of shape `[seq_len, past_len + seq_len]`.
        - `True` means the position is masked.
        - Tokens may attend to all past-cache positions and all current positions up to themselves.
        """,
        reference_solution="""
        import torch

        def causal_mask_with_past(seq_len, past_len=0):
            total = past_len + seq_len
            current_positions = torch.arange(seq_len).unsqueeze(1)
            key_positions = torch.arange(total).unsqueeze(0)
            return key_positions > (current_positions + past_len)
        """,
        tests="""
        import torch
        from solution import causal_mask_with_past

        def test_causal_mask_with_past():
            mask = causal_mask_with_past(seq_len=3, past_len=2)
            expected = torch.tensor(
                [
                    [False, False, False, True, True],
                    [False, False, False, False, True],
                    [False, False, False, False, False],
                ]
            )
            assert torch.equal(mask, expected)
        """,
    ),
    "accelerator_backward_loss": _task(
        task_id="accelerator_backward_loss",
        difficulty="easy",
        topic="accelerate",
        entrypoint="accelerator_backward_loss",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `accelerator_backward_loss(accelerator, loss)`.

        Requirements:
        - Call `accelerator.backward(loss)`.
        - Return the scalar loss value as a Python float.
        """,
        reference_solution="""
        def accelerator_backward_loss(accelerator, loss):
            accelerator.backward(loss)
            return float(loss.detach().item())
        """,
        tests="""
        import torch
        from solution import accelerator_backward_loss

        class FakeAccelerator:
            def __init__(self):
                self.called_with = None

            def backward(self, loss):
                self.called_with = loss

        def test_accelerator_backward_loss():
            accelerator = FakeAccelerator()
            loss = torch.tensor(3.5, requires_grad=True)
            value = accelerator_backward_loss(accelerator, loss)
            assert accelerator.called_with is loss
            assert value == 3.5
        """,
    ),
    "accelerator_gather_for_metrics": _task(
        task_id="accelerator_gather_for_metrics",
        difficulty="easy",
        topic="accelerate",
        entrypoint="accelerator_gather_for_metrics",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `accelerator_gather_for_metrics(accelerator, tensor)`.

        Requirements:
        - If `accelerator` defines `gather_for_metrics`, call it.
        - Otherwise return `tensor` unchanged.
        """,
        reference_solution="""
        def accelerator_gather_for_metrics(accelerator, tensor):
            gather = getattr(accelerator, "gather_for_metrics", None)
            if gather is None:
                return tensor
            return gather(tensor)
        """,
        tests="""
        import torch
        from solution import accelerator_gather_for_metrics

        class FakeAccelerator:
            def gather_for_metrics(self, tensor):
                return tensor + 1

        def test_accelerator_gather_for_metrics():
            tensor = torch.tensor([1, 2])
            assert torch.equal(accelerator_gather_for_metrics(FakeAccelerator(), tensor), torch.tensor([2, 3]))
            assert torch.equal(accelerator_gather_for_metrics(object(), tensor), tensor)
        """,
    ),
    "grad_scaler_unscale_and_clip": _task(
        task_id="grad_scaler_unscale_and_clip",
        difficulty="medium",
        topic="mixed_precision",
        entrypoint="grad_scaler_unscale_and_clip",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `grad_scaler_unscale_and_clip(optimizer, scaler, parameters, max_norm)`.

        Requirements:
        - Call `scaler.unscale_(optimizer)`.
        - Clip gradients with `torch.nn.utils.clip_grad_norm_`.
        - Return the pre-clipped total norm.
        """,
        reference_solution="""
        import torch

        def grad_scaler_unscale_and_clip(optimizer, scaler, parameters, max_norm):
            scaler.unscale_(optimizer)
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        """,
        tests="""
        import torch
        from solution import grad_scaler_unscale_and_clip

        class FakeScaler:
            def __init__(self):
                self.called = False

            def unscale_(self, optimizer):
                self.called = True

        def test_grad_scaler_unscale_and_clip():
            parameter = torch.nn.Parameter(torch.zeros(2))
            parameter.grad = torch.tensor([3.0, 4.0])
            scaler = FakeScaler()
            norm = grad_scaler_unscale_and_clip(None, scaler, [parameter], 2.5)
            assert scaler.called is True
            assert torch.isclose(norm, torch.tensor(5.0))
            assert torch.allclose(parameter.grad, torch.tensor([1.5, 2.0]), atol=1e-4)
        """,
    ),
    "select_autocast_dtype": _task(
        task_id="select_autocast_dtype",
        difficulty="easy",
        topic="mixed_precision",
        entrypoint="select_autocast_dtype",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `select_autocast_dtype(device_type)`.

        Requirements:
        - Return `torch.bfloat16` for `cuda` and `cpu`.
        - Return `torch.float32` for any other device type.
        """,
        reference_solution="""
        import torch

        def select_autocast_dtype(device_type):
            if device_type in {"cuda", "cpu"}:
                return torch.bfloat16
            return torch.float32
        """,
        tests="""
        import torch
        from solution import select_autocast_dtype

        def test_select_autocast_dtype():
            assert select_autocast_dtype("cuda") == torch.bfloat16
            assert select_autocast_dtype("cpu") == torch.bfloat16
            assert select_autocast_dtype("xpu") == torch.float32
        """,
    ),
    "maybe_enable_gradient_checkpointing": _task(
        task_id="maybe_enable_gradient_checkpointing",
        difficulty="easy",
        topic="transformers",
        entrypoint="maybe_enable_gradient_checkpointing",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `maybe_enable_gradient_checkpointing(model)`.

        Requirements:
        - If `model` defines `gradient_checkpointing_enable`, call it and return `True`.
        - Otherwise return `False`.
        """,
        reference_solution="""
        def maybe_enable_gradient_checkpointing(model):
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if enable is None:
                return False
            enable()
            return True
        """,
        tests="""
        from solution import maybe_enable_gradient_checkpointing

        class FakeModel:
            def __init__(self):
                self.enabled = False

            def gradient_checkpointing_enable(self):
                self.enabled = True

        def test_maybe_enable_gradient_checkpointing():
            model = FakeModel()
            assert maybe_enable_gradient_checkpointing(model) is True
            assert model.enabled is True
            assert maybe_enable_gradient_checkpointing(object()) is False
        """,
    ),
    "combine_causal_and_padding_mask": _task(
        task_id="combine_causal_and_padding_mask",
        difficulty="medium",
        topic="transformers",
        entrypoint="combine_causal_and_padding_mask",
        benchmark_bucket="transformers_accelerate_fsdp_mixed_precision",
        prompt="""
        Implement `combine_causal_and_padding_mask(attention_mask)`.

        Requirements:
        - `attention_mask` has shape `[batch, seq_len]`.
        - Return a boolean mask of shape `[batch, 1, seq_len, seq_len]`.
        - A position should be masked if it is in the future or if the key token is padding.
        """,
        reference_solution="""
        import torch

        def combine_causal_and_padding_mask(attention_mask):
            batch, seq_len = attention_mask.shape
            causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            padding = attention_mask[:, None, None, :] == 0
            return causal.view(1, 1, seq_len, seq_len) | padding
        """,
        tests="""
        import torch
        from solution import combine_causal_and_padding_mask

        def test_combine_causal_and_padding_mask():
            mask = torch.tensor([[1, 1, 0]])
            out = combine_causal_and_padding_mask(mask)
            assert out.shape == (1, 1, 3, 3)
            assert out[0, 0, 0, 1].item() is True
            assert out[0, 0, 2, 2].item() is True
            assert out[0, 0, 1, 0].item() is False
        """,
    ),
    "detect_frozen_parameters": _task(
        task_id="detect_frozen_parameters",
        difficulty="easy",
        topic="debugging",
        entrypoint="detect_frozen_parameters",
        benchmark_bucket="debugging_performance",
        prompt="""
        Implement `detect_frozen_parameters(model)`.

        Requirements:
        - Return a list of parameter names whose `requires_grad` is `False`.
        - Preserve the order from `model.named_parameters()`.
        """,
        reference_solution="""
        def detect_frozen_parameters(model):
            return [name for name, param in model.named_parameters() if not param.requires_grad]
        """,
        tests="""
        import torch.nn as nn
        from solution import detect_frozen_parameters

        def test_detect_frozen_parameters():
            model = nn.Linear(2, 2)
            model.bias.requires_grad = False
            assert detect_frozen_parameters(model) == ["bias"]
        """,
    ),
    "find_large_grad_norms": _task(
        task_id="find_large_grad_norms",
        difficulty="medium",
        topic="debugging",
        entrypoint="find_large_grad_norms",
        benchmark_bucket="debugging_performance",
        prompt="""
        Implement `find_large_grad_norms(named_parameters, threshold)`.

        Requirements:
        - `named_parameters` is an iterable of `(name, parameter)` pairs.
        - Return a dictionary mapping parameter names to gradient L2 norms for gradients whose norm exceeds `threshold`.
        - Ignore parameters with `grad is None`.
        """,
        reference_solution="""
        import torch

        def find_large_grad_norms(named_parameters, threshold):
            result = {}
            for name, param in named_parameters:
                if param.grad is None:
                    continue
                norm = torch.norm(param.grad.detach(), p=2).item()
                if norm > threshold:
                    result[name] = norm
            return result
        """,
        tests="""
        import torch
        from solution import find_large_grad_norms

        def test_find_large_grad_norms():
            p1 = torch.nn.Parameter(torch.zeros(2))
            p1.grad = torch.tensor([3.0, 4.0])
            p2 = torch.nn.Parameter(torch.zeros(2))
            p2.grad = torch.tensor([1.0, 1.0])
            out = find_large_grad_norms([("a", p1), ("b", p2)], threshold=3.0)
            assert set(out) == {"a"}
            assert abs(out["a"] - 5.0) < 1e-6
        """,
    ),
    "find_duplicate_parameter_names": _task(
        task_id="find_duplicate_parameter_names",
        difficulty="medium",
        topic="debugging",
        entrypoint="find_duplicate_parameter_names",
        benchmark_bucket="debugging_performance",
        prompt="""
        Implement `find_duplicate_parameter_names(named_parameters)`.

        Requirements:
        - `named_parameters` is an iterable of `(name, parameter)` pairs.
        - If the same parameter object appears under multiple names, return the duplicate names after the first occurrence.
        - Return duplicates in encounter order.
        """,
        reference_solution="""
        def find_duplicate_parameter_names(named_parameters):
            seen = {}
            duplicates = []
            for name, param in named_parameters:
                identifier = id(param)
                if identifier in seen:
                    duplicates.append(name)
                else:
                    seen[identifier] = name
            return duplicates
        """,
        tests="""
        import torch
        from solution import find_duplicate_parameter_names

        def test_find_duplicate_parameter_names():
            shared = torch.nn.Parameter(torch.zeros(1))
            params = [("encoder.weight", shared), ("decoder.weight", shared)]
            assert find_duplicate_parameter_names(params) == ["decoder.weight"]
        """,
    ),
    "tokens_per_second": _task(
        task_id="tokens_per_second",
        difficulty="easy",
        topic="performance",
        entrypoint="tokens_per_second",
        benchmark_bucket="debugging_performance",
        prompt="""
        Implement `tokens_per_second(num_tokens, elapsed_sec)`.

        Requirements:
        - Return throughput as `num_tokens / elapsed_sec`.
        - If `elapsed_sec <= 0`, return `0.0` instead of raising.
        """,
        reference_solution="""
        def tokens_per_second(num_tokens, elapsed_sec):
            if elapsed_sec <= 0:
                return 0.0
            return float(num_tokens) / float(elapsed_sec)
        """,
        tests="""
        from solution import tokens_per_second

        def test_tokens_per_second():
            assert tokens_per_second(200, 2.0) == 100.0
            assert tokens_per_second(200, 0.0) == 0.0
        """,
    ),
    "estimate_linear_activation_memory": _task(
        task_id="estimate_linear_activation_memory",
        difficulty="easy",
        topic="performance",
        entrypoint="estimate_linear_activation_memory",
        benchmark_bucket="debugging_performance",
        prompt="""
        Implement `estimate_linear_activation_memory(batch_size, seq_len, hidden_size, dtype_bytes=2)`.

        Requirements:
        - Estimate activation memory in bytes for a dense activation tensor of shape `[batch_size, seq_len, hidden_size]`.
        - Return the estimate as an integer.
        """,
        reference_solution="""
        def estimate_linear_activation_memory(batch_size, seq_len, hidden_size, dtype_bytes=2):
            return int(batch_size * seq_len * hidden_size * dtype_bytes)
        """,
        tests="""
        from solution import estimate_linear_activation_memory

        def test_estimate_linear_activation_memory():
            assert estimate_linear_activation_memory(2, 4, 8, dtype_bytes=2) == 128
        """,
    ),
    "assert_finite_loss": _task(
        task_id="assert_finite_loss",
        difficulty="easy",
        topic="debugging",
        entrypoint="assert_finite_loss",
        benchmark_bucket="debugging_performance",
        prompt="""
        Implement `assert_finite_loss(loss)`.

        Requirements:
        - If `loss` is not finite, raise `ValueError`.
        - Otherwise return the scalar loss value as a Python float.
        """,
        reference_solution="""
        import torch

        def assert_finite_loss(loss):
            if not torch.isfinite(loss):
                raise ValueError("loss is not finite")
            return float(loss.item())
        """,
        tests="""
        import torch
        import pytest
        from solution import assert_finite_loss

        def test_assert_finite_loss():
            assert assert_finite_loss(torch.tensor(1.25)) == 1.25
            with pytest.raises(ValueError):
                assert_finite_loss(torch.tensor(float("nan")))
        """,
    ),
}


def _write_task(task_id: str, spec: dict[str, object]) -> None:
    task_dir = TASKS_ROOT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(spec["metadata"])
    (task_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")
    (task_dir / "prompt.md").write_text(str(spec["prompt"]), encoding="utf-8")
    (task_dir / "reference_solution.py").write_text(str(spec["reference_solution"]), encoding="utf-8")
    (task_dir / "tests.py").write_text(str(spec["tests"]), encoding="utf-8")


def patch_existing_tasks() -> None:
    for task_id, bucket in EXISTING_BUCKETS.items():
        task_dir = TASKS_ROOT / task_id
        metadata_path = task_dir / "metadata.yaml"
        if not metadata_path.exists():
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        metadata["benchmark_bucket"] = bucket
        metadata.setdefault("timeout_sec", 30)
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")

        patch = PATCHED_EXISTING_TASKS.get(task_id)
        if not patch:
            continue
        if "prompt" in patch:
            (task_dir / "prompt.md").write_text(str(patch["prompt"]), encoding="utf-8")
        if "reference_solution" in patch:
            (task_dir / "reference_solution.py").write_text(str(patch["reference_solution"]), encoding="utf-8")
        if "tests" in patch:
            (task_dir / "tests.py").write_text(str(patch["tests"]), encoding="utf-8")


def main() -> None:
    TASKS_ROOT.mkdir(parents=True, exist_ok=True)
    patch_existing_tasks()
    for task_id, spec in NEW_TASKS.items():
        _write_task(task_id, spec)
    total_tasks = len([path for path in TASKS_ROOT.iterdir() if path.is_dir()])
    print(f"internal eval task count: {total_tasks}")


if __name__ == "__main__":
    main()
