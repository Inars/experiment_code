"""Control run: RoBERTa-base fine-tuned on SetFit/mnli with hard labels only.

No distillation, no growing — establishes the lower-bound number every later
experiment should beat.

Usage:
    uv run python scripts/train_baseline.py [--train-limit N] [--epochs E] \
        [--run-name NAME] [--no-save]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from distill_nli.data.mnli import (
    MNLIBatch,
    load_split,
    make_dataloader,
    tokenize_split,
)
from distill_nli.models.student import load_student
from distill_nli.training.loop import train
from distill_nli.utils.config import load_yaml
from distill_nli.utils.logging import get_logger
from distill_nli.utils.seed import seed_everything

from gromo.utils.utils import set_device


REPO = Path(__file__).resolve().parents[1]


def make_hard_ce_loss():
    def _loss(model: torch.nn.Module, batch: MNLIBatch, device: torch.device) -> torch.Tensor:
        ids = batch.input_ids.to(device)
        msk = batch.attention_mask.to(device)
        lbl = batch.labels.to(device)
        logits = model(input_ids=ids, attention_mask=msk).logits
        return F.cross_entropy(logits, lbl)
    return _loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cfg", default=str(REPO / "configs/data.yaml"))
    parser.add_argument("--student-cfg", default=str(REPO / "configs/student.yaml"))
    parser.add_argument("--train-cfg", default=str(REPO / "configs/train.yaml"))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    log = get_logger()
    data_cfg = load_yaml(args.data_cfg)
    student_cfg = load_yaml(args.student_cfg)
    train_cfg = load_yaml(args.train_cfg)

    seed = args.seed if args.seed is not None else int(train_cfg["seed"])
    seed_everything(seed)

    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    eval_every = int(args.eval_every if args.eval_every is not None else train_cfg["eval"]["every_n_steps"])

    device = torch.device(student_cfg["device"] if torch.backends.mps.is_available() else "cpu")
    set_device(str(device))
    log.info(f"device: {device} | seed: {seed} | epochs: {epochs} | eval_every: {eval_every}")

    model, tokenizer = load_student(student_cfg, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"student: {student_cfg['model_name']} | trainable params: {n_params/1e6:.1f}M")

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

    run_dir: Path | None = None
    if not args.no_save:
        run_dir = REPO / train_cfg["logging"]["out_dir"] / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"run_dir: {run_dir}")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        compute_loss=make_hard_ce_loss(),
        device=device,
        epochs=epochs,
        grad_accum_steps=grad_accum,
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        eval_every_optim_steps=eval_every,
        run_dir=run_dir,
        log=log,
    )


if __name__ == "__main__":
    main()
