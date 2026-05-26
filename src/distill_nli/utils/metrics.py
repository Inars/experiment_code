"""MNLI metric helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, preds, average="macro"))


def per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, name in enumerate(label_names):
        mask = labels == i
        out[name] = float((preds[mask] == labels[mask]).mean()) if mask.any() else float("nan")
    return out
