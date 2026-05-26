"""Math helpers for TINY-style growth: column-major vec, unvec, batched CG.

Ported from growing-attention/src/growth/math_utils.py — kept variable-for-variable
identical so debugging by side-by-side comparison is straightforward.
"""

from __future__ import annotations

from typing import Callable

import torch


def vec(matrix: torch.Tensor) -> torch.Tensor:
    """Column-major (Fortran-order) vectorization: stacks columns of `matrix`."""
    return matrix.t().reshape(-1)


def unvec(vector: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of `vec`: reshapes a vector back into (rows, cols) column-major."""
    rows, cols = shape
    return vector.reshape(cols, rows).t()


def batched_cg(
    matvec_fn: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    M: Callable[[torch.Tensor], torch.Tensor] | None = None,
    rtol: float = 1e-5,
    max_iter: int | None = None,
) -> torch.Tensor:
    """Solve A x = b for multiple systems via Preconditioned Conjugate Gradient.

    The leading dimension of `b` is the "batch of independent systems" axis. CG
    iterates each system in lockstep; we stop when ALL systems' relative residuals
    are below `rtol`.

    Args:
        matvec_fn: takes x of shape (B, D) and returns A x of shape (B, D).
        b: right-hand side, shape (B, D).
        x0: initial guess, shape (B, D); defaults to zeros.
        M: optional preconditioner; takes r (B, D) and returns M^{-1} r (B, D).
        rtol: relative-residual tolerance per system.
        max_iter: max iterations; defaults to 2 * D.

    Returns:
        x: solution of shape (B, D).
    """
    B, D = b.shape
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    if max_iter is None:
        max_iter = 2 * D

    rhs_norm = torch.norm(b, dim=1)
    epsilon = torch.tensor(1e-12, dtype=b.dtype, device=b.device)
    tiny = torch.tensor(torch.finfo(b.dtype).tiny, dtype=b.dtype, device=b.device)

    Ax = matvec_fn(x)
    r = b - Ax

    z = M(r) if M is not None else r
    p = z.clone()
    rs_old = (r * z).sum(dim=1)  # (B,)

    for _ in range(max_iter):
        Ap = matvec_fn(p)
        pAp = (p * Ap).sum(dim=1)
        # Sign-preserving safe division; flipping sign in CG would change the step direction.
        safe_pAp = torch.where(
            torch.abs(pAp) > tiny, pAp, torch.where(pAp >= 0, tiny, -tiny),
        )
        alpha = rs_old / safe_pAp

        x = x + alpha.unsqueeze(1) * p
        r = r - alpha.unsqueeze(1) * Ap

        residual_norm = torch.sqrt((r * r).sum(dim=1))
        threshold = rtol * (rhs_norm + epsilon)
        if torch.all(residual_norm <= threshold):
            break

        z = M(r) if M is not None else r
        rs_new = (r * z).sum(dim=1)
        safe_rs_old = torch.where(
            torch.abs(rs_old) > tiny, rs_old, torch.where(rs_old >= 0, tiny, -tiny),
        )
        beta = rs_new / safe_rs_old
        p = z + beta.unsqueeze(1) * p
        rs_old = rs_new

    return x
