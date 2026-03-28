import torch

def accumulate_rows(values, row_indices, num_rows):
    out = torch.zeros(num_rows, values.size(-1), dtype=values.dtype, device=values.device)
    out.index_add_(0, row_indices.long(), values)
    return out
