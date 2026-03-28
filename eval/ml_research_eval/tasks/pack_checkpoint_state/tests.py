import torch.nn as nn
import torch.optim as optim
from solution import pack_checkpoint_state

def test_pack_checkpoint_state():
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    state = pack_checkpoint_state(model, optimizer, step=7)
    assert set(state) == {"model", "optimizer", "step"}
    assert state["step"] == 7
