import torch
from solution import batch_matrix_trace

def test_batch_matrix_trace():
    m = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    res = batch_matrix_trace(m)
    assert torch.allclose(res, torch.tensor([5., 13.]))