"""SetFit/mnli loading, tokenization, and DataLoaders.

SetFit/mnli columns:
    text1 (premise), text2 (hypothesis), label (int), idx, label_text

Canonical label order used by this project:
    0 = entailment, 1 = neutral, 2 = contradiction
(matches SetFit/mnli; the teacher uses the reverse order and is remapped in
models/teacher.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


CANONICAL_LABEL_ORDER = ["entailment", "neutral", "contradiction"]


@dataclass
class MNLIBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def load_split(cfg: dict[str, Any], split_key: str) -> Dataset:
    """Load one MNLI split via the dataset_name+split mapping in cfg.

    `split_key` is one of the keys under cfg['splits'] (e.g. 'train', 'val', 'test').
    """
    split_name = cfg["splits"][split_key]
    return load_dataset(cfg["dataset_name"], split=split_name)


def tokenize_split(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: dict[str, Any],
) -> Dataset:
    text1, text2, label = cfg["text1_field"], cfg["text2_field"], cfg["label_field"]
    max_len = cfg["max_seq_len"]

    def _enc(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        enc = tokenizer(
            batch[text1],
            batch[text2],
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = batch[label]
        return enc

    keep = ["input_ids", "attention_mask", "labels"]
    remove = [c for c in ds.column_names if c not in keep]
    return ds.map(_enc, batched=True, remove_columns=remove)


def make_collator(tokenizer: PreTrainedTokenizerBase):
    pad_id = tokenizer.pad_token_id

    def _collate(rows: list[dict[str, Any]]) -> MNLIBatch:
        max_len = max(len(r["input_ids"]) for r in rows)
        input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), max_len), dtype=torch.long)
        labels = torch.empty(len(rows), dtype=torch.long)
        for i, r in enumerate(rows):
            n = len(r["input_ids"])
            input_ids[i, :n] = torch.tensor(r["input_ids"], dtype=torch.long)
            attn[i, :n] = torch.tensor(r["attention_mask"], dtype=torch.long)
            labels[i] = r["labels"]
        return MNLIBatch(input_ids=input_ids, attention_mask=attn, labels=labels)

    return _collate


def make_dataloader(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collator(tokenizer),
        pin_memory=False,  # MPS doesn't benefit from pinned memory
    )
