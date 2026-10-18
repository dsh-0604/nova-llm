"""监督微调（SFT）数据集"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM_PROMPT = "You are a helpful assistant."

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

class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=2048, system_prompt=SYSTEM_PROMPT):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.system_prompt = system_prompt
        self.rows = _read_jsonl(file_path)
        self.im_start_id = self._special_id(IM_START)
        self.im_end_id = self._special_id(IM_END)
    def _special_id(self, token):
        tok = self.tokenizer
        tid = getattr(tok, "convert_tokens_to_ids", None)
        if tid is not None:
            try:
                idx = tid(token)
                if isinstance(idx, int) and idx != tok.unk_token_id:
                    return idx
            except Exception:
                pass
        try:
            ids = tok.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        except Exception:
            pass
        return None
    def _encode(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [int(i) for i in ids]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        conv = self.rows[idx]["conversations"]
        messages = []
        if conv and conv[0].get("role") == "system":
            messages.append(conv[0])
            rest = conv[1:]
        else:
            messages.append({"role": "system", "content": self.system_prompt})
            rest = conv
        input_ids = []
        labels = []
        def append(role, content, trainable):
            header = f"{IM_START}{role}\n"
            footer = f"{IM_END}\n"
            h_ids = self._encode(header)
            c_ids = self._encode(content)
            f_ids = self._encode(footer)
            seq = h_ids + c_ids + f_ids
            seg_labels = ([IGNORE_INDEX] * len(h_ids) + ([t for t in c_ids] if trainable else [IGNORE_INDEX] * len(c_ids)) + [IGNORE_INDEX] * len(f_ids))
            input_ids.extend(seq)
            labels.extend(seg_labels)
        for turn in rest:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "assistant":
                append("assistant", content, trainable=True)
            elif role == "user":
                append("user", content, trainable=False)
            elif role == "system":
                append("system", content, trainable=False)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]
            labels = labels[-self.max_seq_len:]
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_t)
        return {"input_ids": input_ids_t, "labels": labels_t, "attention_mask": attention_mask}
