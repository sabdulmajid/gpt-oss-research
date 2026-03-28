import torch

from solution import clip_grad_norm_and_report


def test_clip_grad_norm_and_report_clips_in_place():
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
    params[0].grad = torch.tensor([3.0, 4.0])
    params[1].grad = torch.tensor([12.0])

    before = clip_grad_norm_and_report(params, max_norm=5.0)
    total_after = torch.sqrt(sum((param.grad ** 2).sum() for param in params))

    assert abs(before - 13.0) < 1e-5
    assert abs(float(total_after) - 5.0) < 1e-4


def test_clip_grad_norm_and_report_handles_missing_grads():
    params = [torch.nn.Parameter(torch.zeros(1))]
    assert clip_grad_norm_and_report(params, max_norm=1.0) == 0.0

