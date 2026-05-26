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
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from distill_nli.growing._math import batched_cg, unvec, vec


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


# =============================================================================
# Growth step (port of growing-attention/src/growth/two_shots.py::TwoShotsIterative)
# =============================================================================


LossFn = Callable[[nn.Module, Any], torch.Tensor]


@dataclass
class AttentionGrowConfig:
    """Hyperparameters for one attention grow step. Mirrors configs/grow.yaml."""

    p_per_grow: int = 4
    max_k_per_head: int = 128
    top_k: int = 12
    lambda_reg: float = 1e-3
    alpha: float = 1e-3
    adaptive_tau: float = 1e-4
    cg_tol: float = 1e-5
    cg_max_iter: int = 50
    precision: str = "float64"  # "float64" (recommended) or "float32"


# ---- candidate book-keeping ------------------------------------------------


@dataclass
class _HeadCandidate:
    """Per-head working state during a grow step."""

    layer_idx: int
    head_idx: int
    head: GrowableAttentionHead
    actual_p: int
    delta_Q: torch.Tensor | None = None
    delta_K: torch.Tensor | None = None
    tilde_Q: torch.Tensor | None = None
    tilde_K: torch.Tensor | None = None
    score: float = -float("inf")
    rmse_pre: float = float("nan")
    rmse_post: float = float("nan")
    rmse_target: float = float("nan")


@dataclass
class _StatBundle:
    """Per-head accumulators + per-layer X cache shared across heads in the layer."""

    layer_idx: int
    E: int                   # embed dim (hidden_size)
    K: int                   # current k_dim of this head
    dtype: torch.dtype
    device: torch.device
    C: torch.Tensor = field(init=False)
    mean_norm_T_sq: torch.Tensor = field(init=False)
    diag_S: torch.Tensor = field(init=False)
    diag_G_Q: torch.Tensor = field(init=False)
    diag_G_K: torch.Tensor = field(init=False)
    diag_B_Q_sq: torch.Tensor = field(init=False)
    diag_B_K_sq: torch.Tensor = field(init=False)
    count: int = 0
    X_cache: list[torch.Tensor] = field(default_factory=list)  # CPU tensors

    def __post_init__(self) -> None:
        e, k, d, dev = self.E, self.K, self.dtype, self.device
        self.C = torch.zeros(e, e, dtype=d, device=dev)
        self.mean_norm_T_sq = torch.zeros((), dtype=d, device=dev)
        self.diag_S = torch.zeros(e, dtype=d, device=dev)
        self.diag_G_Q = torch.zeros(k, dtype=d, device=dev)
        self.diag_G_K = torch.zeros(k, dtype=d, device=dev)
        self.diag_B_Q_sq = torch.zeros(e, k, dtype=d, device=dev)
        self.diag_B_K_sq = torch.zeros(e, k, dtype=d, device=dev)


# ---- hooks ----------------------------------------------------------------


def _register_growth_hooks(
    growable_attns: list[GrowableRobertaSelfAttention],
) -> tuple[
    list[torch.utils.hooks.RemovableHandle],
    dict[int, torch.Tensor],            # layer_idx -> last batch's X (B, S, E)
    dict[tuple[int, int], torch.Tensor],  # (layer_idx, head_idx) -> last batch's L (B, S, S)
]:
    """Hook each attention's input and each head's logit_comp output.

    Returns the handles (to remove on teardown) and dict objects that the hooks
    write into. The dicts are mutated in place per forward pass.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    layer_inputs: dict[int, torch.Tensor] = {}
    head_logits: dict[tuple[int, int], torch.Tensor] = {}

    for layer_idx, attn in enumerate(growable_attns):

        def _input_hook(_mod, inputs, _layer_idx=layer_idx):
            # `inputs[0]` is the hidden_states arg.
            layer_inputs[_layer_idx] = inputs[0].detach()

        handles.append(attn.register_forward_pre_hook(_input_hook))

        for head_idx, head in enumerate(attn.heads):

            def _logit_hook(_mod, _inputs, output, _li=layer_idx, _hi=head_idx):
                head_logits[(_li, _hi)] = output
                output.retain_grad()

            handles.append(head.logit_comp.register_forward_hook(_logit_hook))

    return handles, layer_inputs, head_logits


# ---- stat accumulation ----------------------------------------------------


def _accumulate_attention_stats(
    student: nn.Module,
    growable_attns: list[GrowableRobertaSelfAttention],
    probe_batches: Iterable[Any],
    loss_fn: LossFn,
    *,
    dtype: torch.dtype,
) -> dict[tuple[int, int], _StatBundle]:
    """Run probe batches, capturing per-head stats and a per-layer X cache.

    The cache is on CPU; the per-head accumulators live on the head's device.
    """
    handles, layer_inputs, head_logits = _register_growth_hooks(growable_attns)

    # MPS does not support fp64 natively, so all grow-step math runs on CPU.
    # The model forward/backward stays on its real device (e.g. MPS); we move
    # captured tensors to CPU+target-dtype before accumulating.
    stats_device = torch.device("cpu")

    stats: dict[tuple[int, int], _StatBundle] = {}
    layer_X_cache: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(growable_attns))}
    for li, attn in enumerate(growable_attns):
        for hi, head in enumerate(attn.heads):
            stats[(li, hi)] = _StatBundle(
                layer_idx=li,
                E=head.hidden_size,
                K=head.k_dim,
                dtype=dtype,
                device=stats_device,
            )

    was_training = student.training
    student.eval()
    try:
        for batch in probe_batches:
            student.zero_grad(set_to_none=True)
            loss = loss_fn(student, batch)
            loss.backward()

            # Cache X per layer in CPU+fp32 (once per batch, shared across heads in the layer).
            for li, X in layer_inputs.items():
                layer_X_cache[li].append(X.detach().to("cpu", dtype=torch.float32))

            # Per-head accumulation in CPU+target_dtype.
            for (li, hi), sb in stats.items():
                L = head_logits.get((li, hi))
                if L is None or L.grad is None:
                    continue

                # MPS lacks fp64 support, so move to CPU before casting dtype.
                X_in = layer_inputs[li].to(stats_device).to(dtype=dtype)        # (B, S, E)
                T_batch = (-L.grad).to(stats_device).to(dtype=dtype)             # (B, S, S)
                B = X_in.shape[0]

                # S = X^T X per sample, (B, E, E)
                S_batch = torch.bmm(X_in.transpose(1, 2), X_in)

                # C = X^T T X per sample, summed across batch -> (E, E)
                TX = torch.bmm(T_batch, X_in)                    # (B, S, E)
                C_batch_sum = torch.bmm(X_in.transpose(1, 2), TX).sum(dim=0)

                head = growable_attns[li].heads[hi]
                W_Q = head.W_Q.data.to(stats_device).to(dtype=dtype)            # (E, K)
                W_K = head.W_K.data.to(stats_device).to(dtype=dtype)

                # B = S W (per sample), (B, E, K)
                W_Q_b = W_Q.unsqueeze(0).expand(B, -1, -1)
                W_K_b = W_K.unsqueeze(0).expand(B, -1, -1)
                B_Q_per = torch.bmm(S_batch, W_Q_b)
                B_K_per = torch.bmm(S_batch, W_K_b)
                # G = W^T S W (per sample), (B, K, K)
                G_Q_per = torch.bmm(W_Q.t().unsqueeze(0).expand(B, -1, -1), B_Q_per)
                G_K_per = torch.bmm(W_K.t().unsqueeze(0).expand(B, -1, -1), B_K_per)

                sb.count += B
                sb.C += C_batch_sum
                sb.mean_norm_T_sq += (T_batch.norm(dim=(1, 2)) ** 2).sum()
                sb.diag_S += torch.diagonal(S_batch, dim1=1, dim2=2).sum(dim=0)
                sb.diag_G_Q += torch.diagonal(G_Q_per, dim1=1, dim2=2).sum(dim=0)
                sb.diag_G_K += torch.diagonal(G_K_per, dim1=1, dim2=2).sum(dim=0)
                sb.diag_B_Q_sq += (B_Q_per ** 2).sum(dim=0)
                sb.diag_B_K_sq += (B_K_per ** 2).sum(dim=0)

                # Clear so a missed-grad on the next batch is detected.
                head_logits[(li, hi)] = None

            layer_inputs.clear()
    finally:
        for h in handles:
            h.remove()
        if was_training:
            student.train()

    # Finalize: divide by count, attach the per-layer X cache to each head.
    for (li, hi), sb in stats.items():
        if sb.count > 0:
            sb.C /= sb.count
            sb.mean_norm_T_sq /= sb.count
            sb.diag_S /= sb.count
            sb.diag_G_Q /= sb.count
            sb.diag_G_K /= sb.count
            sb.diag_B_Q_sq /= sb.count
            sb.diag_B_K_sq /= sb.count
        sb.X_cache = layer_X_cache[sb.layer_idx]

    return stats


# ---- Step 1: fixed-architecture update ------------------------------------


def _solve_step_1(
    head: GrowableAttentionHead,
    sb: _StatBundle,
    cfg: AttentionGrowConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best fixed-architecture update via matrix-free Conjugate Gradient.

    Solves Ω · vec([δW_Q; δW_K]) = b, where Ω is the Hessian of
    ||T − ∂L(W)/∂W · ΔW||² + λ‖ΔW‖² wrt (ΔW_Q, ΔW_K), ridge-regularized.

    Implementation mirrors growing-attention's TwoShotsIterative._solve_step_1:
    Ω·v is computed by iterating the cached batches (X tensors on CPU) and
    summing the gradient-like terms.
    """
    E, K = head.hidden_size, head.k_dim
    dtype = sb.dtype
    device = sb.device

    W_Q = head.W_Q.data.to(device).to(dtype=dtype)
    W_K = head.W_K.data.to(device).to(dtype=dtype)

    # RHS: b_Q = vec(C @ W_K); b_K = vec(C^T @ W_Q).
    b = torch.cat([vec(sb.C @ W_K), vec(sb.C.t() @ W_Q)], dim=0).unsqueeze(0)  # (1, 2EK)

    # Adaptive ridge: scale lambda by the mean diagonal of the implicit Ω.
    tr_A11 = sb.diag_G_K.sum() * sb.diag_S.sum()
    tr_A22 = sb.diag_G_Q.sum() * sb.diag_S.sum()
    mean_diag = (tr_A11 + tr_A22) / (2 * E * K)
    actual_lambda = max(cfg.lambda_reg, cfg.adaptive_tau * mean_diag.item())

    n_total = sb.count

    def matvec(w: torch.Tensor) -> torch.Tensor:
        w = w.squeeze(0)
        Delta_Q = unvec(w[: E * K], (E, K))
        Delta_K = unvec(w[E * K:], (E, K))

        sum_vQ = torch.zeros(E, K, device=device, dtype=dtype)
        sum_vK = torch.zeros(E, K, device=device, dtype=dtype)

        for X_cpu in sb.X_cache:
            X = X_cpu.to(device=device, dtype=dtype)
            B = X.shape[0]
            S = torch.bmm(X.transpose(1, 2), X)
            W_Q_b = W_Q.unsqueeze(0).expand(B, -1, -1)
            W_K_b = W_K.unsqueeze(0).expand(B, -1, -1)
            B_Q = torch.bmm(S, W_Q_b)
            B_K = torch.bmm(S, W_K_b)
            G_Q = torch.bmm(W_Q.t().unsqueeze(0).expand(B, -1, -1), B_Q)
            G_K = torch.bmm(W_K.t().unsqueeze(0).expand(B, -1, -1), B_K)

            Delta_Q_b = Delta_Q.unsqueeze(0).expand(B, -1, -1)
            Delta_K_b = Delta_K.unsqueeze(0).expand(B, -1, -1)
            Delta_Q_t_b = Delta_Q.t().unsqueeze(0).expand(B, -1, -1)
            Delta_K_t_b = Delta_K.t().unsqueeze(0).expand(B, -1, -1)

            # v_Q = S ΔW_Q G_K + B_Q ΔW_K^T B_K
            sum_vQ += (torch.bmm(torch.bmm(S, Delta_Q_b), G_K)
                       + torch.bmm(torch.bmm(B_Q, Delta_K_t_b), B_K)).sum(dim=0)
            # v_K = B_K ΔW_Q^T B_Q + S ΔW_K G_Q
            sum_vK += (torch.bmm(torch.bmm(B_K, Delta_Q_t_b), B_Q)
                       + torch.bmm(torch.bmm(S, Delta_K_b), G_Q)).sum(dim=0)

        vQ = sum_vQ / n_total + actual_lambda * Delta_Q
        vK = sum_vK / n_total + actual_lambda * Delta_K
        return torch.cat([vec(vQ), vec(vK)], dim=0).unsqueeze(0)

    # Jacobi preconditioner: diag(Ω) ≈ kron(diag_G, diag_S) on each diag block.
    diag_A11 = torch.kron(sb.diag_G_K, sb.diag_S) + actual_lambda
    diag_A22 = torch.kron(sb.diag_G_Q, sb.diag_S) + actual_lambda
    M_diag = torch.cat([diag_A11, diag_A22], dim=0).unsqueeze(0)

    def precond(r: torch.Tensor) -> torch.Tensor:
        return r / M_diag

    w_sol = batched_cg(
        matvec, b, M=precond, rtol=cfg.cg_tol, max_iter=cfg.cg_max_iter,
    ).squeeze(0)

    # Solver runs in fp64 on CPU; keep results in fp64 on CPU for downstream
    # math (step_2, score). The cast back to head device/dtype happens once
    # in _apply_head_growth.
    delta_Q = unvec(w_sol[: E * K], (E, K))
    delta_K = unvec(w_sol[E * K:], (E, K))
    return delta_Q, delta_K


# ---- Step 2: growth update (new columns) ----------------------------------


def _solve_step_2(
    head: GrowableAttentionHead,
    delta_Q: torch.Tensor,
    delta_K: torch.Tensor,
    sb: _StatBundle,
    cfg: AttentionGrowConfig,
    p: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Growth update via matrix-free CG + SVD truncation.

    Solves (H_0 + α I) · vec(Y) = vec(C) + α · vec(Z_0), where
    Z_0 = δW_Q W_K^T + W_Q δW_K^T and H_0 v = E[S v S] (no E²×E² materialization).
    Then SVD(Y − Z_0), take top-p singular components → new (~W_Q, ~W_K).
    """
    E = head.hidden_size
    dtype = sb.dtype
    device = sb.device

    dQ = delta_Q.to(device).to(dtype=dtype)
    dK = delta_K.to(device).to(dtype=dtype)
    W_Q = head.W_Q.data.to(device).to(dtype=dtype)
    W_K = head.W_K.data.to(device).to(dtype=dtype)

    Z0 = dQ @ W_K.t() + W_Q @ dK.t()

    # Adaptive ridge.
    tr_H0 = sb.diag_S.sum() ** 2
    mean_diag = tr_H0 / (E * E)
    actual_alpha = max(cfg.alpha, cfg.adaptive_tau * mean_diag.item())

    rhs = (vec(sb.C) + actual_alpha * vec(Z0)).unsqueeze(0)
    n_total = sb.count

    def matvec(y: torch.Tensor) -> torch.Tensor:
        y = y.squeeze(0)
        Y = unvec(y, (E, E))
        sum_v = torch.zeros(E, E, device=device, dtype=dtype)
        for X_cpu in sb.X_cache:
            X = X_cpu.to(device=device, dtype=dtype)
            B = X.shape[0]
            S = torch.bmm(X.transpose(1, 2), X)
            sum_v += torch.bmm(torch.bmm(S, Y.unsqueeze(0).expand(B, -1, -1)), S).sum(dim=0)
        return vec(sum_v / n_total + actual_alpha * Y).unsqueeze(0)

    diag_H0 = torch.kron(sb.diag_S, sb.diag_S)
    M_diag = (diag_H0 + actual_alpha).unsqueeze(0)

    def precond(r: torch.Tensor) -> torch.Tensor:
        return r / M_diag

    y_sol = batched_cg(
        matvec, rhs, M=precond, rtol=cfg.cg_tol, max_iter=cfg.cg_max_iter,
    ).squeeze(0)

    Y_star = unvec(y_sol, (E, E))
    Z_star = Y_star - Z0
    U, S_vals, Vh = torch.linalg.svd(Z_star)
    sqrt_S = torch.diag(torch.sqrt(S_vals.clamp_min(0)))
    tilde_Q = (U @ sqrt_S)[:, :p].contiguous()
    tilde_K = (Vh.t().contiguous() @ sqrt_S)[:, :p].contiguous()
    # Stay in CPU+sb.dtype; the cast back to head device/dtype happens in _apply_head_growth.
    return tilde_Q, tilde_K


# ---- Score (used to rank heads for top-k selection) -----------------------


def _score_head(
    head: GrowableAttentionHead,
    delta_Q: torch.Tensor,
    delta_K: torch.Tensor,
    tilde_Q: torch.Tensor,
    tilde_K: torch.Tensor,
    sb: _StatBundle,
) -> tuple[float, float, float, float]:
    """Return (score, rmse_pre, rmse_post, rmse_target).

    Score is (rmse_pre / rmse_post) * rmse_target — favors heads where new
    dimensions reduce the residual error AND that have significant target error.
    """
    dtype = sb.dtype
    device = sb.device

    dQ = delta_Q.to(device).to(dtype=dtype)
    dK = delta_K.to(device).to(dtype=dtype)
    tQ = tilde_Q.to(device).to(dtype=dtype)
    tK = tilde_K.to(device).to(dtype=dtype)
    W_Q = head.W_Q.data.to(device).to(dtype=dtype)
    W_K = head.W_K.data.to(device).to(dtype=dtype)

    Z0 = dQ @ W_K.t() + W_Q @ dK.t()
    Z_total = Z0 + tQ @ tK.t()
    C = sb.C
    n_total = sb.count
    mean_norm_T_sq = sb.mean_norm_T_sq.item()

    def mse(Z: torch.Tensor) -> float:
        # ||T − X Z X^T||²_F per-sample, averaged over the cache.
        sum_norm_XZX = 0.0
        for X_cpu in sb.X_cache:
            X = X_cpu.to(device=device, dtype=dtype)
            B = X.shape[0]
            XZ = torch.bmm(X, Z.unsqueeze(0).expand(B, -1, -1))
            XZXt = torch.bmm(XZ, X.transpose(1, 2))
            sum_norm_XZX += (XZXt.norm(dim=(1, 2)) ** 2).sum().item()
        return max(0.0, mean_norm_T_sq - 2.0 * (C * Z).sum().item() + sum_norm_XZX / n_total)

    rmse_pre = math.sqrt(mse(Z0))
    rmse_post = math.sqrt(mse(Z_total))
    rmse_target = math.sqrt(mean_norm_T_sq)
    score = (rmse_pre / (rmse_post + 1e-6)) * rmse_target
    return score, rmse_pre, rmse_post, rmse_target


# ---- Apply growth ---------------------------------------------------------


def _apply_head_growth(
    head: GrowableAttentionHead,
    delta_Q: torch.Tensor,
    delta_K: torch.Tensor,
    tilde_Q: torch.Tensor,
    tilde_K: torch.Tensor,
    p: int,
) -> None:
    """In-place: W_Q ← cat([W_Q + δW_Q, ~W_Q]) and W_K likewise; k_dim += p."""
    with torch.no_grad():
        # Solver outputs live on CPU in fp64; cast back to the head's device/dtype.
        target_device = head.W_Q.device
        target_dtype = head.W_Q.dtype
        # Solver outputs live on CPU+fp64; cast dtype FIRST on CPU, then move
        # to MPS (MPS lacks fp64; chaining the other way would error).
        dQ = delta_Q.to(dtype=target_dtype).to(device=target_device)
        dK = delta_K.to(dtype=target_dtype).to(device=target_device)
        tQ = tilde_Q.to(dtype=target_dtype).to(device=target_device)
        tK = tilde_K.to(dtype=target_dtype).to(device=target_device)

        new_W_Q = torch.cat([head.W_Q.data + dQ, tQ], dim=1)
        new_W_K = torch.cat([head.W_K.data + dK, tK], dim=1)

        head.W_Q = nn.Parameter(new_W_Q)
        head.W_K = nn.Parameter(new_W_K)

        head.k_dim += p
        head.kappa.data = torch.tensor(
            math.sqrt(head.k_dim), device=head.kappa.device, dtype=head.kappa.dtype,
        )


# ---- Orchestration --------------------------------------------------------


def grow_step_attention(
    student: nn.Module,
    growable_attns: list[GrowableRobertaSelfAttention],
    probe_batches: Iterable[Any],
    loss_fn: LossFn,
    *,
    cfg: AttentionGrowConfig,
) -> dict[tuple[int, int], dict[str, Any]]:
    """One TINY-style attention grow step (top-k selection across all heads).

    Returns a per-head report keyed by (layer_idx, head_idx) recording
    {k_added, score, rmse_pre, rmse_post, rmse_target, applied}.
    """
    dtype = torch.float64 if cfg.precision == "float64" else torch.float32

    # 1. Stats accumulation.
    stats = _accumulate_attention_stats(
        student=student,
        growable_attns=growable_attns,
        probe_batches=list(probe_batches),
        loss_fn=loss_fn,
        dtype=dtype,
    )

    # 2. Per-candidate Step 1, Step 2, score.
    candidates: list[_HeadCandidate] = []
    for li, attn in enumerate(growable_attns):
        for hi, head in enumerate(attn.heads):
            actual_p = min(
                cfg.p_per_grow,
                cfg.max_k_per_head - head.k_dim,
                head.hidden_size - head.k_dim,
            )
            if actual_p <= 0:
                continue

            sb = stats[(li, hi)]
            delta_Q, delta_K = _solve_step_1(head, sb, cfg)
            tilde_Q, tilde_K = _solve_step_2(head, delta_Q, delta_K, sb, cfg, actual_p)
            score, rmse_pre, rmse_post, rmse_target = _score_head(
                head, delta_Q, delta_K, tilde_Q, tilde_K, sb,
            )
            candidates.append(_HeadCandidate(
                layer_idx=li, head_idx=hi, head=head, actual_p=actual_p,
                delta_Q=delta_Q, delta_K=delta_K, tilde_Q=tilde_Q, tilde_K=tilde_K,
                score=score, rmse_pre=rmse_pre, rmse_post=rmse_post, rmse_target=rmse_target,
            ))

    # 3. Top-k selection across ALL candidates.
    candidates.sort(key=lambda c: c.score, reverse=True)
    selected = candidates[: max(0, cfg.top_k)]
    selected_keys = {(c.layer_idx, c.head_idx) for c in selected}

    # 4. Apply growth to selected heads.
    for c in selected:
        _apply_head_growth(c.head, c.delta_Q, c.delta_K, c.tilde_Q, c.tilde_K, c.actual_p)

    # 5. Build report (all candidates, with `applied` flag for selected ones).
    report: dict[tuple[int, int], dict[str, Any]] = {}
    for c in candidates:
        key = (c.layer_idx, c.head_idx)
        report[key] = {
            "k_added": c.actual_p if key in selected_keys else 0,
            "applied": key in selected_keys,
            "score": c.score,
            "rmse_pre": c.rmse_pre,
            "rmse_post": c.rmse_post,
            "rmse_target": c.rmse_target,
        }
    return report
