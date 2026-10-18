"""DataLoader 工厂与动态 padding collate_fn"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

IGNORE_INDEX = -100

def _pad_sequences(seqs, pad_value):
    max_len = max(s.shape[0] for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0]] = s
    return out

class DataCollator:
    def __init__(self, pad_token_id=0, label_pad_token_id=IGNORE_INDEX):
        self.pad_token_id = int(pad_token_id)
        self.label_pad = int(label_pad_token_id)
    def __call__(self, features):
        keys = features[0].keys()
        batch = {}
        for key in keys:
            values = [f[key] for f in features]
            if key.endswith("attention_mask"):
                padded = _pad_sequences(values, pad_value=0)
            elif key.startswith("labels") or key.endswith("_labels"):
                padded = _pad_sequences(values, pad_value=self.label_pad)
            else:
                padded = _pad_sequences(values, pad_value=self.pad_token_id)
            batch[key] = padded
        return batch

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False,
                      collate_fn=None, distributed=False, pad_token_id=0, drop_last=False):
    sampler = None
    if distributed and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                      rank=dist.get_rank(), shuffle=shuffle)
        shuffle_flag = False
    else:
        shuffle_flag = shuffle
    if collate_fn is None:
        collate_fn = DataCollator(pad_token_id=pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag if sampler is None else None,
                      sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                      collate_fn=collate_fn, drop_last=drop_last)
