from __future__ import annotations
from typing import List, Dict

import torch
from torch.utils.data import Dataset


class CLMDataset(Dataset):
    def __init__(self, tokenizer, texts: List[str], max_len: int):
        self.encodings = []
        for t in texts:
            enc = tokenizer(t, max_length=max_len, truncation=True,
                            return_tensors="pt", add_special_tokens=True)
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            self.encodings.append({"input_ids": ids, "attention_mask": mask, "labels": ids.clone()})

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_pad(batch, pad_id):
    out = {}
    for k in batch[0]:
        pv = -100 if k == "labels" else (pad_id if k == "input_ids" else 0)
        out[k] = torch.nn.utils.rnn.pad_sequence(
            [b[k] for b in batch], batch_first=True, padding_value=pv)
    return out


def box_answer_texts(texts):
    r"""Render plain QA rows into \boxed{} format (--boxed_answers mode).

    'Question: q\nAnswer: a'  ->  'Question: q\nAnswer: \boxed{a}'

    Replaces pre-rendered boxed dataset variants: the
    transform happens at load time so ONE on-disk dataset serves both modes.
    Contract matches evals/boxed_parse.extract_boxed (first box, content up
    to the first '}'): suitable for brace-free answers (symbol codes, short
    factoids) — answers containing '}' need a balanced parser first.

    - First '\nAnswer:' occurrence wins; rows without the marker pass through
      unchanged (e.g. seed-token-prefixed rows still match — the marker is
      internal).
    - Idempotent: rows whose answer already starts with \boxed are skipped,
      so double application (or pre-boxed data) cannot double-wrap.
    """
    out = []
    for tx in texts:
        head, sep, tail = tx.partition("\nAnswer:")
        if not sep:
            out.append(tx)
            continue
        ans = tail.strip()
        if ans.startswith("\\boxed"):
            out.append(tx)
            continue
        out.append(f"{head}{sep} \\boxed{{{ans}}}")
    return out
