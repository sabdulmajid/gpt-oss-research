import torch
from solution import select_autocast_dtype

def test_select_autocast_dtype():
    assert select_autocast_dtype("cuda") == torch.bfloat16
    assert select_autocast_dtype("cpu") == torch.bfloat16
    assert select_autocast_dtype("xpu") == torch.float32
