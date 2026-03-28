from solution import get_cosine_lr
import math

def test_cosine_lr():
    assert math.isclose(get_cosine_lr(0, 100, 1.0, 0.0), 1.0)
    assert math.isclose(get_cosine_lr(100, 100, 1.0, 0.0), 0.0)
    assert math.isclose(get_cosine_lr(50, 100, 1.0, 0.0), 0.5)