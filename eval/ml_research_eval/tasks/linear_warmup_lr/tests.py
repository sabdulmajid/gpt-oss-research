from solution import linear_warmup_lr

def test_linear_warmup_lr():
    assert linear_warmup_lr(0, 10, 1.0) == 0.0
    assert linear_warmup_lr(5, 10, 1.0) == 0.5
    assert linear_warmup_lr(10, 10, 1.0) == 1.0
    assert linear_warmup_lr(5, 0, 1.0) == 1.0
