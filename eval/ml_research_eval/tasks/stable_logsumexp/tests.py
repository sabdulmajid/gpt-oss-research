import torch
from solution import stable_logsumexp

def test_stable_logsumexp():
    x = torch.tensor([[1000.0, 1001.0]])
    out = stable_logsumexp(x, dim=1)
    expected = torch.logsumexp(x, dim=1)
    assert torch.allclose(out, expected, atol=1e-5)
