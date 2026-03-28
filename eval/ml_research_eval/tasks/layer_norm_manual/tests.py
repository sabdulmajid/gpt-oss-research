import torch
from solution import manual_layer_norm
import torch.nn.functional as F

def test_layer_norm():
    x = torch.randn(2, 3, 4)
    gamma = torch.ones(4)
    beta = torch.zeros(4)
    out = manual_layer_norm(x, gamma, beta)
    expected = F.layer_norm(x, (4,), gamma, beta, 1e-5)
    assert torch.allclose(out, expected, atol=1e-4)