import torch
import pytest
from solution import assert_finite_loss

def test_assert_finite_loss():
    assert assert_finite_loss(torch.tensor(1.25)) == 1.25
    with pytest.raises(ValueError):
        assert_finite_loss(torch.tensor(float("nan")))
