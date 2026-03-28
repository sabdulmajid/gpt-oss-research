from solution import tokens_per_second

def test_tokens_per_second():
    assert tokens_per_second(200, 2.0) == 100.0
    assert tokens_per_second(200, 0.0) == 0.0
