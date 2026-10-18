"""预训练数据集"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, Iterator, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100

def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

class PretrainDataset(Dataset):
    def __init__(self, file_path, tokenizer=None, max_seq_len=2048, stride=None, memmap=True, eos_token_id=None):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.stride = int(stride) if stride is not None else int(max_seq_len)
        self.eos_token_id = eos_token_id
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".npy", ".bin"):
            self._tokens = self._load_token_file(file_path, ext, memmap=memmap)
            self._is_token_file = True
        else:
            assert tokenizer is not None, "jsonl 模式必须提供 tokenizer"
            self._tokens = self._encode_jsonl(file_path, eos_token_id)
            self._is_token_file = False
        if self.eos_token_id is None and tokenizer is not None:
            self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        n = int(self._tokens.shape[0])
        self._starts = []
        start = 0
        while start + self.max_seq_len + 1 <= n:
            self._starts.append(start)
            start += self.stride
        if n - start >= 2 and (not self._starts or start != self._starts[-1]):
            last = max(0, n - self.max_seq_len - 1)
            if last not in self._starts:
                self._starts.append(last)
    @staticmethod
    def _load_token_file(path, ext, memmap):
        if ext == ".npy":
            return np.load(path, mmap_mode="r" if memmap else None)
        if memmap:
            return np.memmap(path, dtype=np.int32, mode="r")
        return np.fromfile(path, dtype=np.int32)
    def _encode_jsonl(self, path, eos_token_id):
        eos = eos_token_id if eos_token_id is not None else getattr(self.tokenizer, "eos_token_id", 0)
        chunks = []
        for record in _read_jsonl(path):
            text = record.get("text", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            arr = np.asarray(ids, dtype=np.int32)
            if arr.size == 0:
                continue
            chunks.append(np.append(arr, np.int32(eos)))
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(chunks).astype(np.int32)
    def __len__(self):
        return len(self._starts)
    def __getitem__(self, idx):
        start = self._starts[idx]
        end = start + self.max_seq_len + 1
        buf = np.asarray(self._tokens[start:end], dtype=np.int64)
        input_ids = torch.from_numpy(buf[:-1].copy())
        labels = torch.from_numpy(buf[1:].copy())
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    @property
    def num_tokens(self):
        return int(self._tokens.shape[0])
