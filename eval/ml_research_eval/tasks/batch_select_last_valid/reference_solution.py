import torch

def select_last_valid(hidden_states, attention_mask):
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    gather_index = lengths.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
    return hidden_states.gather(1, gather_index).squeeze(1)
