"""Shared training loop for baseline, distillation, and (eventually) distill+grow.

The loop is parameterized by a `compute_loss(model, batch, device) -> Tensor`
callable so each script can plug in its own loss (hard CE, distillation mix, etc.)
without re-implementing the optimizer/scheduler/eval/checkpoint scaffolding.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from distill_nli.data.mnli import CANONICAL_LABEL_ORDER, MNLIBatch
from distill_nli.utils.metrics import accuracy, macro_f1, per_class_accuracy


LossFn = Callable[[nn.Module, "MNLIBatch", torch.device], torch.Tensor]


@dataclass
class EvalResult:
    step: int
    accuracy: float
    macro_f1: float


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[EvalResult, np.ndarray, np.ndarray]:
    was_training = model.training
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
    if was_training:
        model.train()
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return (
        EvalResult(step=-1, accuracy=accuracy(preds, labels), macro_f1=macro_f1(preds, labels)),
        preds,
        labels,
    )


def train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    compute_loss: LossFn,
    device: torch.device,
    epochs: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    eval_every_optim_steps: int,
    run_dir: Path | None,
    log,
) -> tuple[EvalResult, dict[str, float]]:
    metrics_path = run_dir / "metrics.jsonl" if run_dir is not None else None
    if metrics_path is not None:
        metrics_path.write_text("")

    optim_step = 0
    best_acc = -1.0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            loss = compute_loss(model, batch, device) / grad_accum_steps
            loss.backward()
            bs = batch.labels.size(0)
            running_loss += loss.item() * grad_accum_steps * bs
            running_n += bs

            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

                if optim_step % max(1, eval_every_optim_steps) == 0:
                    res, _, _ = evaluate(model, val_loader, device)
                    res.step = optim_step
                    log.info(f"[step {optim_step}] val acc={res.accuracy:.4f} f1={res.macro_f1:.4f}")
                    if metrics_path is not None:
                        with open(metrics_path, "a") as f:
                            f.write(json.dumps({"event": "eval", **asdict(res)}) + "\n")
                        if res.accuracy > best_acc:
                            best_acc = res.accuracy
                            torch.save(model.state_dict(), run_dir / "best.pt")
                            log.info(f"  -> new best ({best_acc:.4f}), saved best.pt")

            if (i + 1) % 50 == 0:
                pbar.set_postfix(loss=f"{running_loss/max(1,running_n):.4f}", step=optim_step)

        log.info(f"epoch {epoch+1} done | avg train loss = {running_loss/max(1,running_n):.4f}")

    final, preds, labels = evaluate(model, val_loader, device)
    final.step = optim_step
    per_cls = per_class_accuracy(preds, labels, CANONICAL_LABEL_ORDER)
    log.info(f"final val acc={final.accuracy:.4f} f1={final.macro_f1:.4f}")
    for k, v in per_cls.items():
        log.info(f"  acc[{k:>13}] = {v:.4f}")
    log.info(f"wall-clock: {(time.time()-t0)/60:.1f} min")

    if metrics_path is not None:
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"event": "final", **asdict(final), "per_class": per_cls}) + "\n")
        torch.save(model.state_dict(), run_dir / "last.pt")
        if final.accuracy > best_acc:
            torch.save(model.state_dict(), run_dir / "best.pt")

    return final, per_cls
