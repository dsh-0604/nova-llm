"""偏好数据集（DPO / RLHF）"""
from __future__ import annotations
import json
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

class PreferenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=2048, max_prompt_len=1024):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.max_prompt_len = int(max_prompt_len)
        self.rows = _read_jsonl(file_path)
    def _encode(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [int(i) for i in ids]
    def _build_pair(self, prompt_ids, response):
        resp_ids = self._encode(f"{IM_START}assistant\n{response}{IM_END}\n")
        full = prompt_ids + resp_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(resp_ids)
        if len(full) > self.max_seq_len:
            overflow = len(full) - self.max_seq_len
            full = full[overflow:]
            labels = labels[overflow:]
        input_ids_t = torch.tensor(full, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return {"input_ids": input_ids_t, "labels": labels_t, "attention_mask": torch.ones_like(input_ids_t)}
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]
        prompt_text = f"{IM_START}user\n{prompt}{IM_END}\n"
        prompt_ids = self._encode(prompt_text)
        if len(prompt_ids) > self.max_prompt_len:
            prompt_ids = prompt_ids[-self.max_prompt_len:]
        chosen_pair = self._build_pair(prompt_ids, chosen)
        rejected_pair = self._build_pair(prompt_ids, rejected)
        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "chosen_input_ids": chosen_pair["input_ids"],
            "chosen_labels": chosen_pair["labels"],
            "chosen_attention_mask": chosen_pair["attention_mask"],
            "rejected_input_ids": rejected_pair["input_ids"],
            "rejected_labels": rejected_pair["labels"],
            "rejected_attention_mask": rejected_pair["attention_mask"],
        }
