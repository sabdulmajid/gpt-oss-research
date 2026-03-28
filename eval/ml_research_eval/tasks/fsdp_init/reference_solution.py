import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def wrap_fsdp(model):
    if torch.distributed.is_initialized():
        return FSDP(model)
    return model