import torch

from solution import pad_batch


def test_pad_batch_shapes_and_padding():
    batch = pad_batch(
        [
            {"input_ids": [10, 11, 12], "labels": [11, 12, 13]},
            {"input_ids": [20], "labels": [21]},
        ],
        pad_token_id=0,
    )
    assert set(batch.keys()) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert torch.equal(batch["input_ids"][1], torch.tensor([20, 0, 0]))
    assert torch.equal(batch["labels"][1], torch.tensor([21, -100, -100]))
    assert torch.equal(batch["attention_mask"][1], torch.tensor([1, 0, 0]))

