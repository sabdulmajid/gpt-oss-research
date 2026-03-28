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
