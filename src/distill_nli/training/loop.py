"""Shared training loop for baseline, distillation, and (eventually) distill+grow.

The loop is parameterized by a `compute_loss(model, batch, device) -> Tensor`
callable so each script can plug in its own loss (hard CE, distillation mix, etc.)
without re-implementing the optimizer/scheduler/eval/checkpoint scaffolding.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from distill_nli.data.mnli import CANONICAL_LABEL_ORDER, MNLIBatch
from distill_nli.growing.schedule import GrowthSchedule
from distill_nli.utils.metrics import accuracy, macro_f1, per_class_accuracy


LossFn = Callable[[nn.Module, "MNLIBatch", torch.device], torch.Tensor]
GrowStepFn = Callable[[nn.Module, list[Any]], dict[str, Any]]
RebuildOptimFn = Callable[[nn.Module], tuple[torch.optim.Optimizer, Any]]


def _summarize_grow(report: dict[str, Any]) -> str:
    """One-line summary of a grow_step return value for the training log."""
    parts: list[str] = []
    ffn = report.get("ffn")
    if ffn:
        total = sum(int(v) for v in ffn.values())
        parts.append(f"ffn={total} neurons across {len(ffn)} layers")
    attn = report.get("attention")
    if attn:
        applied = sum(1 for v in attn.values() if v.get("applied"))
        k_added = sum(int(v["k_added"]) for v in attn.values() if v.get("applied"))
        parts.append(f"attn={k_added} k_dim across {applied} heads")
    return "; ".join(parts) if parts else "no growth applied"


def _jsonify_grow_report(report: dict[str, Any]) -> dict[str, Any]:
    """Convert tuple keys (layer_idx, head_idx) to 'L{li}/H{hi}' strings."""
    out: dict[str, Any] = {}
    for kind, value in report.items():
        if not isinstance(value, dict):
            out[kind] = value
            continue
        if kind == "attention":
            out[kind] = {f"L{li}/H{hi}": v for (li, hi), v in value.items()}
        else:
            out[kind] = {str(k): v for k, v in value.items()}
    return out


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
    # Optional growth hook. All three must be set together for growth to fire.
    growth_schedule: GrowthSchedule | None = None,
    do_grow: GrowStepFn | None = None,
    rebuild_optimizer: RebuildOptimFn | None = None,
    num_probe_batches: int = 0,
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

        # ----- end-of-epoch growth hook -----
        if (
            growth_schedule is not None
            and do_grow is not None
            and growth_schedule.is_growth_epoch(epoch)
        ):
            probe = list(itertools.islice(val_loader, max(1, num_probe_batches)))
            log.info(f"[epoch {epoch+1}] growth attempt ({len(probe)} probe batches)...")
            grow_metrics = do_grow(model, probe)
            growth_schedule.record_grow()
            log.info(f"  growth done: {_summarize_grow(grow_metrics)}")
            if metrics_path is not None:
                jsonable = _jsonify_grow_report(grow_metrics)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"event": "grow", "epoch": epoch + 1, **jsonable}, default=str) + "\n")
            if rebuild_optimizer is not None:
                optimizer, scheduler = rebuild_optimizer(model)
                log.info("  optimizer + scheduler rebuilt after grow")

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
