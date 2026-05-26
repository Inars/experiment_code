"""Distillation losses (Hinton et al., 2015).

L = alpha * T^2 * KL(softmax(s/T) || softmax(t/T))  +  (1 - alpha) * CE(s, y)

The T^2 factor preserves the gradient magnitude when T changes — without it the
soft-loss contribution shrinks as T grows.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total_loss, soft_loss, hard_loss).

    Both teacher and student logits are expected in the project's canonical label
    order (entailment=0, neutral=1, contradiction=2).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    T = temperature
    s_log_probs = F.log_softmax(student_logits / T, dim=-1)
    t_probs = F.softmax(teacher_logits / T, dim=-1)
    soft = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)

    hard = F.cross_entropy(student_logits, labels)

    total = alpha * soft + (1.0 - alpha) * hard
    return total, soft.detach(), hard.detach()
