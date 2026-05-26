"""Control run: RoBERTa-base fine-tuned on SetFit/mnli with hard labels only.

No distillation, no growing — establishes the lower-bound number every later
experiment should beat.

Usage:
    uv run python scripts/train_baseline.py [--train-limit N] [--epochs E] \
        [--run-name NAME] [--no-save]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from distill_nli.data.mnli import (
    CANONICAL_LABEL_ORDER,
    load_split,
    make_dataloader,
    tokenize_split,
)
from distill_nli.models.student import load_student
from distill_nli.utils.config import load_yaml
from distill_nli.utils.logging import get_logger
from distill_nli.utils.metrics import accuracy, macro_f1, per_class_accuracy
from distill_nli.utils.seed import seed_everything

from gromo.utils.utils import set_device


REPO = Path(__file__).resolve().parents[1]


@dataclass
class EvalResult:
    step: int
    accuracy: float
    macro_f1: float


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[EvalResult, np.ndarray, np.ndarray]:
    model.eval()
    preds_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch.input_ids.to(device)
            msk = batch.attention_mask.to(device)
            logits = model(input_ids=ids, attention_mask=msk).logits
            preds_all.append(logits.argmax(dim=-1).cpu().numpy())
            labels_all.append(batch.labels.numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return EvalResult(step=-1, accuracy=accuracy(preds, labels), macro_f1=macro_f1(preds, labels)), preds, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cfg", default=str(REPO / "configs/data.yaml"))
    parser.add_argument("--student-cfg", default=str(REPO / "configs/student.yaml"))
    parser.add_argument("--train-cfg", default=str(REPO / "configs/train.yaml"))
    parser.add_argument("--train-limit", type=int, default=None, help="Use only first N train examples")
    parser.add_argument("--val-limit", type=int, default=None, help="Use only first N val examples")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.yaml epochs")
    parser.add_argument("--eval-every", type=int, default=None, help="Override eval.every_n_steps")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--no-save", action="store_true", help="Skip checkpoint writes")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    log = get_logger()
    data_cfg = load_yaml(args.data_cfg)
    student_cfg = load_yaml(args.student_cfg)
    train_cfg = load_yaml(args.train_cfg)

    seed = args.seed if args.seed is not None else int(train_cfg["seed"])
    seed_everything(seed)

    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    eval_every = int(
        args.eval_every if args.eval_every is not None else train_cfg["eval"]["every_n_steps"]
    )

    device = torch.device(student_cfg["device"] if torch.backends.mps.is_available() else "cpu")
    set_device(str(device))
    log.info(f"device: {device}")
    log.info(f"seed: {seed} | epochs: {epochs} | eval_every: {eval_every}")

    # ----- model + tokenizer -----
    model, tokenizer = load_student(student_cfg, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"student: {student_cfg['model_name']} | trainable params: {n_params/1e6:.1f}M")

    # ----- data -----
    ds_train = load_split(data_cfg, "train")
    ds_val = load_split(data_cfg, "val")
    if args.train_limit is not None:
        ds_train = ds_train.select(range(min(args.train_limit, len(ds_train))))
    if args.val_limit is not None:
        ds_val = ds_val.select(range(min(args.val_limit, len(ds_val))))
    log.info(f"train: {len(ds_train)} | val: {len(ds_val)}")

    ds_train_tok = tokenize_split(ds_train, tokenizer, data_cfg)
    ds_val_tok = tokenize_split(ds_val, tokenizer, data_cfg)
    train_loader = make_dataloader(
        ds_train_tok, tokenizer, batch_size=data_cfg["train_batch_size"],
        shuffle=True, num_workers=data_cfg["num_workers"],
    )
    val_loader = make_dataloader(
        ds_val_tok, tokenizer, batch_size=data_cfg["eval_batch_size"],
        shuffle=False, num_workers=data_cfg["num_workers"],
    )

    # ----- optimizer + scheduler -----
    opt_cfg = train_cfg["optimizer"]
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    total_optim_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(float(train_cfg["scheduler"]["warmup_ratio"]) * total_optim_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps,
    )
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    # ----- run dir -----
    run_dir = REPO / train_cfg["logging"]["out_dir"] / args.run_name
    if not args.no_save:
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.jsonl"
        metrics_path.write_text("")
    log.info(f"run_dir: {run_dir}")

    # ----- training loop -----
    global_step = 0
    optim_step = 0
    best_acc = -1.0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            ids = batch.input_ids.to(device)
            msk = batch.attention_mask.to(device)
            lbl = batch.labels.to(device)

            logits = model(input_ids=ids, attention_mask=msk).logits
            loss = F.cross_entropy(logits, lbl) / grad_accum
            loss.backward()
            running_loss += loss.item() * grad_accum * lbl.size(0)
            running_n += lbl.size(0)

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

                if optim_step % max(1, eval_every // grad_accum) == 0:
                    res, _, _ = evaluate(model, val_loader, device)
                    res.step = optim_step
                    log.info(f"[step {optim_step}] val acc={res.accuracy:.4f} f1={res.macro_f1:.4f}")
                    if not args.no_save:
                        with open(metrics_path, "a") as f:
                            f.write(json.dumps({"event": "eval", **asdict(res)}) + "\n")
                        if res.accuracy > best_acc:
                            best_acc = res.accuracy
                            torch.save(model.state_dict(), run_dir / "best.pt")
                            log.info(f"  -> new best ({best_acc:.4f}), saved best.pt")
                    model.train()

            global_step += 1
            if global_step % 50 == 0:
                pbar.set_postfix(loss=f"{running_loss/max(1,running_n):.4f}", step=optim_step)

        log.info(f"epoch {epoch+1} done | avg train loss = {running_loss/max(1,running_n):.4f}")

    # ----- final eval -----
    res, preds, labels = evaluate(model, val_loader, device)
    res.step = optim_step
    per_cls = per_class_accuracy(preds, labels, CANONICAL_LABEL_ORDER)
    log.info(f"final val acc={res.accuracy:.4f} f1={res.macro_f1:.4f}")
    for k, v in per_cls.items():
        log.info(f"  acc[{k:>13}] = {v:.4f}")
    log.info(f"wall-clock: {(time.time()-t0)/60:.1f} min")

    if not args.no_save:
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"event": "final", **asdict(res), "per_class": per_cls}) + "\n")
        torch.save(model.state_dict(), run_dir / "last.pt")
        if res.accuracy > best_acc:
            torch.save(model.state_dict(), run_dir / "best.pt")


if __name__ == "__main__":
    main()
