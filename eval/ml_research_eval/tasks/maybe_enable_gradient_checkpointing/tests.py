from solution import maybe_enable_gradient_checkpointing

class FakeModel:
    def __init__(self):
        self.enabled = False

    def gradient_checkpointing_enable(self):
        self.enabled = True

def test_maybe_enable_gradient_checkpointing():
    model = FakeModel()
    assert maybe_enable_gradient_checkpointing(model) is True
    assert model.enabled is True
    assert maybe_enable_gradient_checkpointing(object()) is False
