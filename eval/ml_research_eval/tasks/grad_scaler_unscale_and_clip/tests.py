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
