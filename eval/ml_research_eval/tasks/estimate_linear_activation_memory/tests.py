from solution import estimate_linear_activation_memory

def test_estimate_linear_activation_memory():
    assert estimate_linear_activation_memory(2, 4, 8, dtype_bytes=2) == 128
