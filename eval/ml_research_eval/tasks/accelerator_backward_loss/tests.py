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
