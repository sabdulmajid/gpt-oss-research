import torch
from solution import collate_pad_to_multiple

def test_collate_pad_to_multiple():
    batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    out = collate_pad_to_multiple(batch, pad_value=-1, multiple_of=4)
    assert out.shape == (2, 4)
    assert out[1, 2:].tolist() == [-1, -1]
