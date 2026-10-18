"""数据预处理工具：预编码、文本清洗、文档打包"""
from __future__ import annotations
import json
import re
from typing import Any, Iterator, List, Optional
import numpy as np

def clean_text(text):
    if not text:
        return ""
    text = "".join(ch for ch in text if ch in ("\t", "\n") or ord(ch) >= 32)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_document_packing(docs, max_seq_len, eos_id):
    packs = []
    current = []
    for doc in docs:
        ids = doc if isinstance(doc, list) else [doc]
        if len(current) + len(ids) + 1 > max_seq_len:
            if current:
                packs.append(current)
            current = list(ids)
        else:
            current.extend(ids)
        current.append(eos_id)
    if current:
        packs.append(current)
    return [np.asarray(p, dtype=np.int32) for p in packs]

def jsonl_to_tokens(input_file, output_file, tokenizer, max_seq_len=None, eos_token_id=None):
    eos = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", 0)
    tokens = []
    total = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = clean_text(record.get("text", ""))
            if not text:
                continue
            ids = tokenizer.encode(text)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            arr = np.asarray(ids, dtype=np.int32)
            if max_seq_len is not None:
                arr = arr[:max_seq_len]
            tokens.append(arr)
            tokens.append(np.asarray([eos], dtype=np.int32))
            total += arr.size + 1
    if not tokens:
        out = np.zeros(0, dtype=np.int32)
    else:
        out = np.concatenate(tokens).astype(np.int32)
    if output_file.endswith(".npy"):
        np.save(output_file, out)
    else:
        out.astype("<i4").tofile(output_file)
    return total
