import torch
from solution import accelerator_gather_for_metrics

class FakeAccelerator:
    def gather_for_metrics(self, tensor):
        return tensor + 1

def test_accelerator_gather_for_metrics():
    tensor = torch.tensor([1, 2])
    assert torch.equal(accelerator_gather_for_metrics(FakeAccelerator(), tensor), torch.tensor([2, 3]))
    assert torch.equal(accelerator_gather_for_metrics(object(), tensor), tensor)
