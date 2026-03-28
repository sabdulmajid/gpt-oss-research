import torch

from solution import masked_mean_pool


def test_masked_mean_pool_averages_only_valid_tokens():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            [[2.0, 6.0], [4.0, 8.0], [6.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    pooled = masked_mean_pool(hidden, mask)
    expected = torch.tensor([[2.0, 3.0], [5.0, 9.0]])
    assert torch.allclose(pooled, expected)


def test_masked_mean_pool_handles_all_zero_mask():
    hidden = torch.ones((1, 2, 3), dtype=torch.float32)
    mask = torch.zeros((1, 2), dtype=torch.long)
    pooled = masked_mean_pool(hidden, mask)
    assert pooled.dtype == hidden.dtype
    assert torch.allclose(pooled, torch.zeros((1, 3), dtype=torch.float32))

