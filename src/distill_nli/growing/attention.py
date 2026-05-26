"""TINY-style attention growth via direct port of OneShotGrower (PHASE 3 — currently a stub).

Targets each transformer block's RobertaSelfAttention. The HF module keeps separate
`query`, `key`, `value` Linears already (verified at install time). The growable
replacement splits each per-head — Q and K become per-head Linears with their own
`k_dim[h]`; growth expands W_Q[h] and W_K[h] simultaneously via SVD of P + delta_P.
V and the output projection stay fused.

Math reference:
    growing-attention/src/growth/one_shot.py
    OneShotGrower.{accumulate_stats, grow, solve_update, apply_growth}

Public API (to be implemented in Phase 3):
- wrap_attention(layer: RobertaLayer) -> GrowableRobertaSelfAttention
    In-place replacement of `.attention.self`, preserving the pretrained Q/K/V.
- grow_step(growable_attns, probe_batches, loss_fn, alpha, p_per_grow)
    Accumulates H0 and C per head via forward hooks (input X and grad-retained
    Q@K^T output), solves (H0+alpha*I) vec(delta_P) = vec(C), applies via SVD,
    returns a dict {(layer_idx, head_idx): k_added}.

The loss_fn is loss-source-agnostic: same contract as growing/ffn.py.
"""

from __future__ import annotations
