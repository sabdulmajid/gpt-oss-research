def batched_outer_product(x, y):
    return x.unsqueeze(-1) * y.unsqueeze(-2)
