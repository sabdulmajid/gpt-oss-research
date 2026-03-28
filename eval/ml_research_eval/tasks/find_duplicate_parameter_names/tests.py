import torch
from solution import find_duplicate_parameter_names

def test_find_duplicate_parameter_names():
    shared = torch.nn.Parameter(torch.zeros(1))
    params = [("encoder.weight", shared), ("decoder.weight", shared)]
    assert find_duplicate_parameter_names(params) == ["decoder.weight"]
