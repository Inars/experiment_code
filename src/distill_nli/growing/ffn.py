"""TINY-style FFN growth via gromo (PHASE 2 — currently a stub).

Targets RoBERTa's intermediate (Linear E -> 4E) -> GELU -> output (Linear 4E -> E)
pair around each transformer block. Growth adds neurons to the 4E hidden dim.

Public API (to be implemented in Phase 2):
- wrap_ffn(layer: RobertaLayer) -> GrowableRobertaFFN
    Replaces `.intermediate` + `.output.dense` with a gromo-backed growable pair,
    preserving pretrained weights.
- grow_step(growable_ffns, probe_batches, loss_fn, alpha, neurons_per_grow)
    Accumulates stats over probe_batches, computes the optimal added neurons via
    gromo, applies them in-place, returns a dict {layer_idx: n_added}.

The loss_fn is a loss-source-agnostic callable: (model, batch) -> Tensor. Pass the
distillation loss when a teacher is present, hard-label CE otherwise.
"""

from __future__ import annotations
