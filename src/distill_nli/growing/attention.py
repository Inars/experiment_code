"""TINY-style attention growth (port of growing-attention's TwoShotsIterative).

The HF `RobertaSelfAttention` has fused `query`, `key`, `value` Linears. We
replace it with a per-head structure so each head's Q/K projection can have a
different rank (`k_dim[h]`), growing independently.

V projection stays a single fused Linear (not growable, matches the reference).

Math reference:
    growing-attention/src/growth/two_shots.py  (TwoShotsIterative, _solve_step_1,
    _solve_step_2, _compute_scores, _apply_growth)
    growing-attention/src/growth/math_utils.py  (vec, unvec, batched_cg)

Memory note (RoBERTa-base, hidden_size=768):
    ~30 MB per head for stats + cache. With 12 heads per layer × N wrapped layers,
    grow-step peak memory is roughly N * 350 MB. Wrap a subset of layers via
    cfg.attention.layers if memory is tight on a MacBook Air.

Grow-step math is implemented in this module (Phase 3c, below). The surgery
that swaps this in for HF's RobertaSelfAttention lives in models/growing.py.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _LogitComputation(nn.Module):
    """Separate module so we can register a forward hook on it to capture the
    pre-softmax logits and `retain_grad()` them for TINY stats accumulation."""

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        return torch.matmul(Q, K.transpose(-2, -1))


class GrowableAttentionHead(nn.Module):
    """One growable attention head.

    Holds W_Q ∈ ℝ^(E, K) and W_K ∈ ℝ^(E, K) as nn.Parameters, with K mutable
    (re-bound via nn.Parameter after each growth). W_V is held by the parent
    GrowableRobertaSelfAttention as a shared fused Linear and sliced into
    per-head views at forward time.
    """

    def __init__(
        self,
        hidden_size: int,
        k_dim: int,
        head_size: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size            # E (e.g. 768)
        self.k_dim = k_dim                        # K[h] (e.g. 64 initially)
        self.head_size = head_size                # V dim (e.g. 64, never grown)

        # Q/K projections (per-head, growable). Biases match HF's split Linear.
        self.W_Q = nn.Parameter(torch.empty(hidden_size, k_dim, device=device))
        self.W_K = nn.Parameter(torch.empty(hidden_size, k_dim, device=device))
        self.b_Q = nn.Parameter(torch.zeros(k_dim, device=device))
        self.b_K = nn.Parameter(torch.zeros(k_dim, device=device))

        # kappa is the inverse-sqrt scaling. Kept as a buffer (re-set on grow).
        self.register_buffer(
            "kappa", torch.tensor(math.sqrt(k_dim), device=device),
        )

        # Separate logit-computation module so growth-stat hooks have a clean target.
        self.logit_comp = _LogitComputation()


class GrowableRobertaSelfAttention(nn.Module):
    """Drop-in replacement for HF's RobertaSelfAttention with per-head growable Q/K.

    The output signature matches `RobertaSelfAttention.forward`: returns a
    `(attn_output, attn_weights | None)` tuple.

    The forward path uses a per-head Python loop. Slower than fused SDPA, but
    necessary because k_dim differs per head after growth.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: int,
        dropout_prob: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.all_head_size = num_heads * head_size
        self.dropout_prob = dropout_prob

        # Per-head Q/K state.
        self.heads = nn.ModuleList(
            [
                GrowableAttentionHead(hidden_size, head_size, head_size, device=device)
                for _ in range(num_heads)
            ]
        )

        # Shared V projection (not growable).
        self.value = nn.Linear(hidden_size, self.all_head_size, device=device)

        self.dropout = nn.Dropout(dropout_prob)

    @classmethod
    def from_hf(cls, hf_self_attn: nn.Module) -> "GrowableRobertaSelfAttention":
        """Build from an HF RobertaSelfAttention, copying Q/K/V weights and biases.

        Splits the fused query/key Linears column-wise into per-head slices, so
        head h's W_Q[h] = hf.query.weight[h*head_size:(h+1)*head_size, :].T
        (HF stores Linear weights as (out_features, in_features)).
        """
        hidden = hf_self_attn.query.in_features
        num_heads = hf_self_attn.num_attention_heads
        head_size = hf_self_attn.attention_head_size
        # HF's RobertaSelfAttention reads dropout from its own .dropout Module.
        dropout_prob = float(hf_self_attn.dropout.p)
        device = hf_self_attn.query.weight.device

        instance = cls(
            hidden_size=hidden,
            num_heads=num_heads,
            head_size=head_size,
            dropout_prob=dropout_prob,
            device=device,
        )

        with torch.no_grad():
            # query.weight: (all_head_size, hidden) -> split rows into 12 chunks of (head_size, hidden)
            # head h's W_Q[h]: (hidden, head_size) -> transpose of the chunk
            for h, head in enumerate(instance.heads):
                rs, re = h * head_size, (h + 1) * head_size
                head.W_Q.copy_(hf_self_attn.query.weight[rs:re, :].t().contiguous())
                head.W_K.copy_(hf_self_attn.key.weight[rs:re, :].t().contiguous())
                head.b_Q.copy_(hf_self_attn.query.bias[rs:re])
                head.b_K.copy_(hf_self_attn.key.bias[rs:re])

            # V stays fused.
            instance.value.weight.copy_(hf_self_attn.value.weight)
            instance.value.bias.copy_(hf_self_attn.value.bias)

        return instance

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        # hidden_states: (B, S, hidden)
        b, s, _ = hidden_states.shape

        # Shared V, sliced per head.
        v_all = self.value(hidden_states)              # (B, S, all_head_size)
        v_all = v_all.view(b, s, self.num_heads, self.head_size)  # (B, S, H, head_size)

        # Reduce HF's extended mask (B, 1, 1, S) to (B, 1, S) so it broadcasts
        # onto our (B, S, S) per-head logits.
        mask_2d = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask_2d = attention_mask.squeeze(1)    # (B, 1, S)
            elif attention_mask.dim() == 3:
                mask_2d = attention_mask
            else:
                raise ValueError(
                    f"Unsupported attention_mask shape {tuple(attention_mask.shape)}; "
                    "expected 4D (B, 1, 1, S) or 3D (B, 1, S).",
                )

        outputs: list[torch.Tensor] = []
        for h, head in enumerate(self.heads):
            Q = hidden_states @ head.W_Q + head.b_Q    # (B, S, K[h])
            K = hidden_states @ head.W_K + head.b_K    # (B, S, K[h])
            V_h = v_all[:, :, h, :]                    # (B, S, head_size)

            logits = head.logit_comp(Q, K)             # (B, S, S)
            # Match HF's per-head scaling = 1/sqrt(head_size).
            # Our kappa = sqrt(K[h]) plays the same role and is updated after growth.
            logits = logits / head.kappa

            if mask_2d is not None:
                logits = logits + mask_2d

            # softmax in fp32 then cast back, matching HF eager_attention_forward.
            attn = F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
            attn = self.dropout(attn)

            outputs.append(attn @ V_h)                 # (B, S, head_size)

        # Concat across heads: (B, S, num_heads * head_size = hidden)
        attn_output = torch.cat(outputs, dim=-1)
        return attn_output, None
