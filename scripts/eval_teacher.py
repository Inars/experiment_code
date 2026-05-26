"""Sanity-check teacher accuracy on MNLI validation.

Usage:
    uv run python scripts/eval_teacher.py [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from distill_nli.data.mnli import (
    CANONICAL_LABEL_ORDER,
    load_split,
    make_dataloader,
    tokenize_split,
)
from distill_nli.models.teacher import load_teacher
from distill_nli.utils.config import load_yaml
from distill_nli.utils.logging import get_logger
from distill_nli.utils.metrics import accuracy, macro_f1, per_class_accuracy
from distill_nli.utils.seed import seed_everything

from gromo.utils.utils import set_device


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cfg", default=str(REPO / "configs/data.yaml"))
    parser.add_argument("--teacher-cfg", default=str(REPO / "configs/teacher.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Eval first N examples only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log = get_logger()
    seed_everything(args.seed)

    data_cfg = load_yaml(args.data_cfg)
    teacher_cfg = load_yaml(args.teacher_cfg)

    device = torch.device(teacher_cfg["device"] if torch.backends.mps.is_available() else "cpu")
    set_device(str(device))
    log.info(f"device: {device}")

    log.info(f"loading teacher: {teacher_cfg['model_name']}")
    teacher, tokenizer = load_teacher(teacher_cfg, device=device)

    log.info(f"loading {data_cfg['dataset_name']} [{data_cfg['splits']['val']}]")
    ds = load_split(data_cfg, "val")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    log.info(f"val examples: {len(ds)}")

    ds_tok = tokenize_split(ds, tokenizer, data_cfg)
    loader = make_dataloader(
        ds_tok,
        tokenizer,
        batch_size=data_cfg["eval_batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="eval"):
        ids = batch.input_ids.to(device)
        msk = batch.attention_mask.to(device)
        logits = teacher(ids, msk)  # canonical-order logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch.labels.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = accuracy(preds, labels)
    mf1 = macro_f1(preds, labels)
    per_cls = per_class_accuracy(preds, labels, CANONICAL_LABEL_ORDER)

    log.info(f"accuracy : {acc:.4f}")
    log.info(f"macro-F1 : {mf1:.4f}")
    for name, v in per_cls.items():
        log.info(f"  acc[{name:>13}] = {v:.4f}")


if __name__ == "__main__":
    main()
