import torch

def collate_pad_to_multiple(batch, pad_value=0, multiple_of=8):
    max_len = max(item.numel() for item in batch)
    padded_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of
    out = torch.full((len(batch), padded_len), pad_value, dtype=batch[0].dtype)
    for index, item in enumerate(batch):
        out[index, : item.numel()] = item
    return out
