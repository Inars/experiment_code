"""Ablation: RoBERTa-base + TINY growing, hard-label CE only (no distillation).

Symmetric counterpart to train_distill.py: tests "growth alone" vs "distillation
alone" vs "distillation + growth" (the headline experiment).

Usage:
    uv run python scripts/train_grow.py [--train-limit N] [--epochs E] \
        [--run-name NAME] [--no-save]
"""

from __future__ import annotations

# MPS lacks torch.linalg.eigh (gromo uses it); set fallback BEFORE importing torch.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from distill_nli.data.mnli import (  # noqa: E402
    MNLIBatch,
    load_split,
    make_dataloader,
    tokenize_split,
)
from distill_nli.growing import grow_step  # noqa: E402
from distill_nli.growing.schedule import GrowthSchedule  # noqa: E402
from distill_nli.models.growing import make_student_growable  # noqa: E402
from distill_nli.models.student import load_student  # noqa: E402
from distill_nli.training.loop import train  # noqa: E402
from distill_nli.utils.config import load_yaml  # noqa: E402
from distill_nli.utils.logging import get_logger  # noqa: E402
from distill_nli.utils.seed import seed_everything  # noqa: E402

from gromo.utils.utils import set_device  # noqa: E402


REPO = Path(__file__).resolve().parents[1]


def make_hard_ce_loss():
    def _loss(model: torch.nn.Module, batch: MNLIBatch, device: torch.device) -> torch.Tensor:
        ids = batch.input_ids.to(device)
        msk = batch.attention_mask.to(device)
        lbl = batch.labels.to(device)
        logits = model(input_ids=ids, attention_mask=msk).logits
        return F.cross_entropy(logits, lbl)
    return _loss


def _build_optim_and_sched(model: torch.nn.Module, train_cfg: dict, total_optim_steps: int):
    opt_cfg = train_cfg["optimizer"]
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    warmup_steps = int(float(train_cfg["scheduler"]["warmup_ratio"]) * total_optim_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps,
    )
    return optimizer, scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cfg", default=str(REPO / "configs/data.yaml"))
    parser.add_argument("--student-cfg", default=str(REPO / "configs/student.yaml"))
    parser.add_argument("--train-cfg", default=str(REPO / "configs/train.yaml"))
    parser.add_argument("--grow-cfg", default=str(REPO / "configs/grow.yaml"))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--run-name", default="grow")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    log = get_logger()
    data_cfg = load_yaml(args.data_cfg)
    student_cfg = load_yaml(args.student_cfg)
    train_cfg = load_yaml(args.train_cfg)
    grow_cfg = load_yaml(args.grow_cfg)
    # This script does not use a teacher; growth uses the supervised loss.
    grow_cfg["signal_source"] = "supervised"

    seed = args.seed if args.seed is not None else int(train_cfg["seed"])
    seed_everything(seed)

    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    eval_every = int(args.eval_every if args.eval_every is not None else train_cfg["eval"]["every_n_steps"])

    device = torch.device(student_cfg["device"] if torch.backends.mps.is_available() else "cpu")
    set_device(str(device))
    log.info(f"device: {device} | seed: {seed} | epochs: {epochs}")
    log.info(
        f"growth: ffn={grow_cfg['ffn']['enabled']} attn={grow_cfg['attention']['enabled']} "
        f"warmup={grow_cfg['warmup_epochs']} interval={grow_cfg['interval_epochs']} max={grow_cfg['max_grows']}",
    )

    # ----- model + surgery -----
    student, tokenizer = load_student(student_cfg, device=device)
    registry = make_student_growable(student, grow_cfg)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info(f"student trainable: {n_params/1e6:.1f}M | growable ffn: {len(registry['ffn'])} | growable attn: {len(registry['attention'])}")

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
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    total_optim_steps = (len(train_loader) // grad_accum) * epochs
    optimizer, scheduler = _build_optim_and_sched(student, train_cfg, total_optim_steps)

    # ----- run dir -----
    run_dir: Path | None = None
    if not args.no_save:
        run_dir = REPO / train_cfg["logging"]["out_dir"] / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"run_dir: {run_dir}")

    # ----- growth hook closures -----
    compute_loss = make_hard_ce_loss()

    def grow_loss_fn(model: torch.nn.Module, batch: MNLIBatch) -> torch.Tensor:
        return compute_loss(model, batch, device)

    def do_grow(model: torch.nn.Module, probe_batches):
        return grow_step(model, registry, probe_batches, grow_loss_fn, grow_cfg)

    def rebuild_optimizer(model: torch.nn.Module):
        if not grow_cfg.get("optimizer", {}).get("reset_after_grow", True):
            return optimizer, scheduler  # unchanged; user opted out
        return _build_optim_and_sched(model, train_cfg, total_optim_steps)

    growth_schedule = GrowthSchedule(
        warmup_epochs=int(grow_cfg["warmup_epochs"]),
        interval_epochs=int(grow_cfg["interval_epochs"]),
        max_grows=int(grow_cfg["max_grows"]),
    )

    # ----- train -----
    train(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        compute_loss=compute_loss,
        device=device,
        epochs=epochs,
        grad_accum_steps=grad_accum,
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        eval_every_optim_steps=eval_every,
        run_dir=run_dir,
        log=log,
        growth_schedule=growth_schedule,
        do_grow=do_grow,
        rebuild_optimizer=rebuild_optimizer,
        num_probe_batches=int(grow_cfg["num_probe_batches"]),
    )


if __name__ == "__main__":
    main()
