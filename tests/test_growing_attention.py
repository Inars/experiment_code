"""Tests for attention growth: forward equivalence + grow step + math sanity."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from distill_nli.growing._math import unvec, vec
from distill_nli.growing.attention import (
    AttentionGrowConfig,
    _solve_step_1,
    grow_step_attention,
)
from distill_nli.models.growing import make_student_growable


STUDENT_NAME = "FacebookAI/roberta-base"


requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine.",
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("mps")


@pytest.fixture
def student(device):
    # function-scope: attention surgery is destructive.
    config = AutoConfig.from_pretrained(STUDENT_NAME, num_labels=3)
    # Force eager attention so per-head fp32 softmax matches HF's eager_attention_forward
    # bit-for-bit. (SDPA on MPS may reorder reductions and break a tight allclose.)
    config._attn_implementation = "eager"
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_NAME, config=config)
    model.to(device).eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(STUDENT_NAME)


@pytest.fixture(scope="module")
def batch(tokenizer, device):
    enc = tokenizer(
        ["A man eats a sandwich.", "The cat sleeps on the mat."],
        ["A person consumes food.", "An animal rests."],
        return_tensors="pt", padding=True, truncation=True, max_length=32,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _forward(model, batch) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return out.logits


@requires_mps
def test_attention_wrap_preserves_forward(student, batch):
    """After wrapping attention.self in all 12 layers, logits match pre-surgery to fp32 tolerance."""
    logits_before = _forward(student, batch).clone()

    grow_cfg = {"attention": {"enabled": True, "layers": "all"}}
    registry = make_student_growable(student, grow_cfg)
    assert len(registry["attention"]) == 12

    logits_after = _forward(student, batch)
    max_diff = (logits_before - logits_after).abs().max().item()
    assert torch.allclose(logits_before, logits_after, atol=1e-4, rtol=1e-4), (
        f"max diff = {max_diff:.3e}"
    )


@requires_mps
def test_attention_and_ffn_wrap_together_preserves_forward(student, batch):
    """Wrapping BOTH attention and FFN in all 12 layers preserves forward output."""
    logits_before = _forward(student, batch).clone()

    grow_cfg = {
        "ffn": {"enabled": True, "layers": "all"},
        "attention": {"enabled": True, "layers": "all"},
    }
    registry = make_student_growable(student, grow_cfg)
    assert len(registry["ffn"]) == 12
    assert len(registry["attention"]) == 12

    logits_after = _forward(student, batch)
    max_diff = (logits_before - logits_after).abs().max().item()
    assert torch.allclose(logits_before, logits_after, atol=1e-4, rtol=1e-4), (
        f"max diff = {max_diff:.3e}"
    )


# ---- Math sanity: CG matches a dense solve on a tiny problem ----------------


def _dense_step_1_solve(head, sb, cfg: AttentionGrowConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the full Ω matrix explicitly and solve Ω · vec([δW_Q; δW_K]) = b.

    Only used in tests — has O(E²K²) memory cost, which is fine at E=16, K=4
    but would be 1.4 TB at RoBERTa-base scale (hence the CG version in prod).
    """
    E, K = head.hidden_size, head.k_dim
    dtype = sb.dtype
    device = sb.device

    W_Q = head.W_Q.data.to(dtype=dtype)
    W_K = head.W_K.data.to(dtype=dtype)

    # Adaptive ridge (must match _solve_step_1).
    tr_A11 = sb.diag_G_K.sum() * sb.diag_S.sum()
    tr_A22 = sb.diag_G_Q.sum() * sb.diag_S.sum()
    mean_diag = (tr_A11 + tr_A22) / (2 * E * K)
    actual_lambda = max(cfg.lambda_reg, cfg.adaptive_tau * mean_diag.item())
    n_total = sb.count
    n_X = sum(X.shape[0] for X in sb.X_cache)
    assert n_X == n_total, f"cache size {n_X} != count {n_total}"

    D = 2 * E * K
    Omega = torch.zeros(D, D, dtype=dtype, device=device)
    eye_D = torch.eye(D, dtype=dtype, device=device)
    # Build column-by-column via the matvec.
    from distill_nli.growing.attention import _solve_step_1  # noqa: F401 (re-import for clarity)
    # Reproduce the matvec inline — same algebra as _solve_step_1 but applied to basis vectors.
    for col in range(D):
        e_col = eye_D[col].unsqueeze(0)  # (1, D)
        w = e_col.squeeze(0)
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
            DQ_b = Delta_Q.unsqueeze(0).expand(B, -1, -1)
            DK_b = Delta_K.unsqueeze(0).expand(B, -1, -1)
            DQ_t_b = Delta_Q.t().unsqueeze(0).expand(B, -1, -1)
            DK_t_b = Delta_K.t().unsqueeze(0).expand(B, -1, -1)
            sum_vQ += (torch.bmm(torch.bmm(S, DQ_b), G_K)
                       + torch.bmm(torch.bmm(B_Q, DK_t_b), B_K)).sum(dim=0)
            sum_vK += (torch.bmm(torch.bmm(B_K, DQ_t_b), B_Q)
                       + torch.bmm(torch.bmm(S, DK_b), G_Q)).sum(dim=0)

        vQ = sum_vQ / n_total + actual_lambda * Delta_Q
        vK = sum_vK / n_total + actual_lambda * Delta_K
        Omega[:, col] = torch.cat([vec(vQ), vec(vK)], dim=0)

    b = torch.cat([vec(sb.C @ W_K), vec(sb.C.t() @ W_Q)], dim=0)
    w_sol = torch.linalg.solve(Omega, b)
    return unvec(w_sol[: E * K], (E, K)), unvec(w_sol[E * K:], (E, K))


def test_step1_cg_matches_dense_solve():
    """CG-based Step 1 must agree with an explicit dense Ω solve at fp64."""
    torch.manual_seed(0)
    device = torch.device("cpu")   # math correctness — CPU is deterministic and fast at this size
    dtype = torch.float64

    # Tiny problem so Ω = (2EK, 2EK) is small.
    E, K, B, S = 16, 4, 2, 6
    from distill_nli.growing.attention import GrowableAttentionHead, _StatBundle

    head = GrowableAttentionHead(hidden_size=E, k_dim=K, head_size=K, device=device)
    with torch.no_grad():
        head.W_Q.copy_(torch.randn_like(head.W_Q))
        head.W_K.copy_(torch.randn_like(head.W_K))

    # Hand-build a StatBundle with a small X cache and corresponding C, diag_S, etc.
    sb = _StatBundle(layer_idx=0, E=E, K=K, dtype=dtype, device=device)
    sb.X_cache = [torch.randn(B, S, E, dtype=torch.float32)]

    # Compute the "true" C consistent with the cached X and a random T.
    X = sb.X_cache[0].to(dtype=dtype)
    T = torch.randn(B, S, S, dtype=dtype) * 0.1   # small magnitude for numerical sanity
    Sm = torch.bmm(X.transpose(1, 2), X)
    TX = torch.bmm(T, X)
    sb.C = torch.bmm(X.transpose(1, 2), TX).sum(dim=0) / B
    sb.count = B

    # Fill diagonals from the same X.
    W_Q = head.W_Q.data.to(dtype=dtype)
    W_K = head.W_K.data.to(dtype=dtype)
    B_Q = torch.bmm(Sm, W_Q.unsqueeze(0).expand(B, -1, -1))
    B_K = torch.bmm(Sm, W_K.unsqueeze(0).expand(B, -1, -1))
    G_Q = torch.bmm(W_Q.t().unsqueeze(0).expand(B, -1, -1), B_Q)
    G_K = torch.bmm(W_K.t().unsqueeze(0).expand(B, -1, -1), B_K)
    sb.diag_S = torch.diagonal(Sm, dim1=1, dim2=2).sum(dim=0) / B
    sb.diag_G_Q = torch.diagonal(G_Q, dim1=1, dim2=2).sum(dim=0) / B
    sb.diag_G_K = torch.diagonal(G_K, dim1=1, dim2=2).sum(dim=0) / B
    sb.diag_B_Q_sq = (B_Q ** 2).sum(dim=0) / B
    sb.diag_B_K_sq = (B_K ** 2).sum(dim=0) / B
    sb.mean_norm_T_sq = (T.norm(dim=(1, 2)) ** 2).mean().detach().clone().to(dtype=dtype)

    cfg = AttentionGrowConfig(lambda_reg=1e-3, adaptive_tau=1e-4, cg_tol=1e-10, cg_max_iter=500)

    delta_Q_cg, delta_K_cg = _solve_step_1(head, sb, cfg)
    delta_Q_dense, delta_K_dense = _dense_step_1_solve(head, sb, cfg)

    err_Q = (delta_Q_cg - delta_Q_dense).abs().max().item()
    err_K = (delta_K_cg - delta_K_dense).abs().max().item()
    # CG tol is 1e-10 of relative residual — solution should agree to ~that level.
    assert err_Q < 1e-6, f"δW_Q diff CG vs dense = {err_Q:.3e}"
    assert err_K < 1e-6, f"δW_K diff CG vs dense = {err_K:.3e}"


# ---- One full grow step on a real RoBERTa student ---------------------------


@requires_mps
def test_attention_grow_step_increases_k_dim(student, batch, device):
    """Run one grow_step_attention with 2 layers wrapped; selected heads should
    each gain `p_per_grow` columns in W_Q and W_K. Smoke-level cap of 2 grows."""
    grow_cfg = {"attention": {"enabled": True, "layers": [0, 6]}}
    registry = make_student_growable(student, grow_cfg)
    attns = registry["attention"]
    assert len(attns) == 2

    # Snapshot k_dim per head.
    pre_k = [[h.k_dim for h in a.heads] for a in attns]

    def loss_fn(model, b):
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        targets = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
        return F.cross_entropy(out, targets)

    cfg = AttentionGrowConfig(
        p_per_grow=4,
        max_k_per_head=128,
        top_k=2,                  # small top_k to keep runtime tight
        lambda_reg=1e-3,
        alpha=1e-3,
        adaptive_tau=1e-4,
        cg_tol=1e-4,              # loosen for a fast test
        cg_max_iter=20,
        precision="float64",
    )

    report = grow_step_attention(
        student=student,
        growable_attns=attns,
        probe_batches=[batch, batch],
        loss_fn=loss_fn,
        cfg=cfg,
    )

    applied = [k for k, v in report.items() if v["applied"]]
    assert len(applied) == cfg.top_k, f"expected {cfg.top_k} grows, got {len(applied)}"

    # For each applied head, k_dim must have gone up by exactly p_per_grow and
    # W_Q / W_K must reflect the new column count.
    for (li, hi) in applied:
        layer_pos = [i for i, a in enumerate(attns) if a is attns[0 if li == 0 else 1]][0]
        head = attns[layer_pos].heads[hi]
        assert head.k_dim == pre_k[layer_pos][hi] + cfg.p_per_grow
        assert head.W_Q.shape[1] == head.k_dim
        assert head.W_K.shape[1] == head.k_dim
