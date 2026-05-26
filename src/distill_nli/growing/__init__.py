"""Public entry point for FFN + attention growth.

`grow_step` is the single call the training loop makes per growth epoch. It
inspects which kinds of growable modules are registered (via
`distill_nli.models.growing.make_student_growable`) and dispatches accordingly.

Ordering when both kinds are enabled: attention first, then FFN. The two grow
steps are independent on this codebase (no shared parameters), so the order
only affects which side sees the freshly-grown model for stats accumulation.
Attention-first matches the reference's "structure-shaping first" pattern.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn

from distill_nli.growing.attention import (
    AttentionGrowConfig,
    GrowableRobertaSelfAttention,
    grow_step_attention,
)
from distill_nli.growing.ffn import GrowableRobertaFFN, grow_step_ffn


LossFn = Callable[[nn.Module, Any], torch.Tensor]


def grow_step(
    student: nn.Module,
    registry: dict[str, list[nn.Module]],
    probe_batches: Iterable[Any],
    loss_fn: LossFn,
    grow_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run one combined grow step (attention + FFN) per the registry and config.

    Args:
        student: the model being trained (used for zero_grad).
        registry: output of make_student_growable; lists of growable modules
            keyed by "attention" and "ffn".
        probe_batches: small iterable of batches used to accumulate growth
            statistics. Materialized into a list so it can be reused by both
            growers.
        loss_fn: (student, batch) -> Tensor. Distillation loss when a teacher
            is present, hard-label CE otherwise.
        grow_cfg: the dict loaded from configs/grow.yaml. Reads `attention` and
            `ffn` sub-dicts.

    Returns:
        Combined report:
            {
              "attention": {(layer_idx, head_idx): {...}, ...} or None,
              "ffn":       {ffn_idx: n_added, ...}            or None,
            }
    """
    probe_list = list(probe_batches)
    report: dict[str, Any] = {"attention": None, "ffn": None}

    attn_cfg_dict = grow_cfg.get("attention", {})
    if attn_cfg_dict.get("enabled", False) and registry.get("attention"):
        report["attention"] = grow_step_attention(
            student=student,
            growable_attns=registry["attention"],
            probe_batches=probe_list,
            loss_fn=loss_fn,
            cfg=AttentionGrowConfig(
                p_per_grow=int(attn_cfg_dict.get("p_per_grow", 4)),
                max_k_per_head=int(attn_cfg_dict.get("max_k_per_head", 128)),
                top_k=int(attn_cfg_dict.get("top_k", 12)),
                lambda_reg=float(attn_cfg_dict.get("lambda_reg", 1e-3)),
                alpha=float(attn_cfg_dict.get("alpha", 1e-3)),
                adaptive_tau=float(attn_cfg_dict.get("adaptive_tau", 1e-4)),
                cg_tol=float(attn_cfg_dict.get("cg_tol", 1e-5)),
                cg_max_iter=int(attn_cfg_dict.get("cg_max_iter", 50)),
                precision=str(attn_cfg_dict.get("precision", "float64")),
            ),
        )

    ffn_cfg_dict = grow_cfg.get("ffn", {})
    if ffn_cfg_dict.get("enabled", False) and registry.get("ffn"):
        report["ffn"] = grow_step_ffn(
            student=student,
            growable_ffns=registry["ffn"],
            probe_batches=probe_list,
            loss_fn=loss_fn,
            neurons_per_grow=int(ffn_cfg_dict.get("neurons_per_grow", 64)),
            alpha=float(ffn_cfg_dict.get("alpha", 1e-3)),
            max_intermediate=int(ffn_cfg_dict.get("max_intermediate", 6144)),
        )

    return report


__all__ = [
    "AttentionGrowConfig",
    "GrowableRobertaFFN",
    "GrowableRobertaSelfAttention",
    "grow_step",
    "grow_step_attention",
    "grow_step_ffn",
]
