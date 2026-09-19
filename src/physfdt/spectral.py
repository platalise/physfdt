"""Spectral observables of a weight matrix: tail exponent ``alpha`` and IPR.

For a weight matrix ``W`` of shape ``(N, M)`` the empirical spectral density
(ESD) is the eigenvalue distribution of ``X = W W^T``. Randomly initialised
layers follow Marchenko-Pastur; trained layers grow a heavy tail well described
by ``rho(lambda) ~ lambda^{-alpha}``.

Two estimators of ``alpha`` are provided because they answer different
questions:

``method="mle"`` (default)
    Clauset-Shalizi-Newman maximum likelihood for a continuous power law,
    ``alpha = 1 + k / sum_i ln(lambda_i / lambda_min)``, using the ``k`` largest
    eigenvalues. Standard, has a known sampling error ``(alpha-1)/sqrt(k)``, and
    is what WeightWatcher-style analyses report.

``method="window"``
    Log-log linear regression over a central window of the sorted spectrum.
    Reproduces the convention used in the author's Physica A 692 (2026) 131474
    (default window: drop the top 10% as outliers and the bottom 20% as noise
    floor, fit the central 70%).

The two do not have to agree; a large disagreement usually means the tail is not
well described by a single power law, which is itself worth knowing.

This module is NumPy-only. :func:`spectral_report_torch` accepts a torch tensor
and converts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "SpectralReport",
    "esd",
    "powerlaw_alpha",
    "ipr",
    "spectral_report",
    "spectral_report_torch",
]


@dataclass
class SpectralReport:
    alpha: float
    """Tail exponent of ``rho(lambda) ~ lambda^{-alpha}``."""
    alpha_err: float
    """Standard error. For the MLE this is ``(alpha - 1) / sqrt(k)``; for the
    window fit it is the regression standard error of the slope."""
    r2: float
    """Goodness of fit of the power law on log-log axes."""
    lambda_min: float
    """Lower cut-off of the fitted tail."""
    n_tail: int
    """Number of eigenvalues in the fit."""
    ipr_max: float
    """Largest inverse participation ratio ``Y2 = sum_j |u_j|^4`` over
    eigenvectors. Order ``1/N`` means delocalised, order ``1`` means a feature
    has localised onto a few neurons."""
    ipr_mean: float
    method: str


def esd(W: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``W W^T``, sorted descending.

    Computed from singular values (``lambda = s^2``), which is both faster and
    numerically better conditioned than forming ``W W^T`` explicitly.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {W.shape}")
    s = np.linalg.svd(W, compute_uv=False)
    return np.sort(s**2)[::-1]


def _r2_loglog_ccdf(tail: np.ndarray, alpha: float, lam_min: float) -> float:
    """R^2 of the fitted power law against the empirical CCDF, on log-log axes."""
    x = np.sort(tail)
    n = len(x)
    if n < 3:
        return float("nan")
    # empirical CCDF at each point, excluding the last (which would be 0)
    ccdf = 1.0 - (np.arange(n) + 0.5) / n
    keep = ccdf > 0
    x, ccdf = x[keep], ccdf[keep]
    pred = -(alpha - 1.0) * (np.log(x) - np.log(lam_min))
    obs = np.log(ccdf)
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def powerlaw_alpha(
    eigs: np.ndarray,
    method: str = "mle",
    tail_frac: float = 0.1,
    window: tuple[float, float] = (0.10, 0.80),
) -> tuple[float, float, float, float, int]:
    """Estimate the tail exponent.

    Returns ``(alpha, alpha_err, r2, lambda_min, n_tail)``.

    Parameters
    ----------
    eigs:
        Eigenvalues, any order. Non-positive values are dropped.
    method:
        ``"mle"`` or ``"window"``.
    tail_frac:
        MLE only. Fraction of the largest eigenvalues used as the tail.
    window:
        Window fit only. ``(lo, hi)`` as fractions of the sorted-descending
        spectrum; the default ``(0.10, 0.80)`` fits ranks 10%..80%, i.e. drops
        the top 10% and the bottom 20%.
    """
    lam = np.sort(np.asarray(eigs, dtype=np.float64))[::-1]
    lam = lam[lam > 0]
    n = len(lam)
    if n < 10:
        raise ValueError(f"need at least 10 positive eigenvalues, got {n}")

    if method == "mle":
        k = max(10, int(round(tail_frac * n)))
        k = min(k, n - 1)
        lam_min = lam[k]
        tail = lam[:k]
        xi = float(np.mean(np.log(tail / lam_min)))
        if xi <= 0:
            raise ValueError("degenerate spectrum: cannot estimate a tail exponent")
        alpha = 1.0 + 1.0 / xi
        err = (alpha - 1.0) / np.sqrt(k)
        r2 = _r2_loglog_ccdf(tail, alpha, lam_min)
        return float(alpha), float(err), float(r2), float(lam_min), int(k)

    if method == "window":
        lo, hi = window
        i0, i1 = int(round(lo * n)), int(round(hi * n))
        i1 = max(i1, i0 + 5)
        i1 = min(i1, n)
        seg = lam[i0:i1]
        # rank-frequency: rank r among the sorted-descending spectrum is an
        # (unnormalised) CCDF, so slope on log-log is -(alpha - 1).
        rank = np.arange(i0 + 1, i0 + 1 + len(seg), dtype=np.float64)
        x = np.log(seg)
        y = np.log(rank / n)
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope = float(coef[0])
        alpha = 1.0 - slope
        yhat = A @ coef
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        dof = max(len(seg) - 2, 1)
        sxx = float(np.sum((x - x.mean()) ** 2))
        err = float(np.sqrt(ss_res / dof / sxx)) if sxx > 0 else float("nan")
        return float(alpha), err, float(r2), float(seg[-1]), int(len(seg))

    raise ValueError("method must be 'mle' or 'window'")


def ipr(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, Y2)`` for ``W W^T``, sorted by descending eigenvalue.

    ``Y2_k = sum_j |u_{k,j}|^4`` for the unit-norm eigenvector ``u_k``.
    ``Y2 ~ 1/N`` is a delocalised, noise-like mode; ``Y2 ~ O(1)`` is a mode
    localised on a few components.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {W.shape}")
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    order = np.argsort(s)[::-1]
    return (s[order] ** 2), np.sum(U[:, order] ** 4, axis=0)


def spectral_report(W: np.ndarray, method: str = "mle", **kwargs) -> SpectralReport:
    """Compute ``alpha`` and IPR for one weight matrix in a single SVD."""
    W = np.asarray(W, dtype=np.float64)
    eigs, y2 = ipr(W)
    alpha, err, r2, lam_min, n_tail = powerlaw_alpha(eigs, method=method, **kwargs)
    return SpectralReport(
        alpha=alpha,
        alpha_err=err,
        r2=r2,
        lambda_min=lam_min,
        n_tail=n_tail,
        ipr_max=float(np.max(y2)),
        ipr_mean=float(np.mean(y2)),
        method=method,
    )


def spectral_report_torch(tensor, method: str = "mle", **kwargs) -> SpectralReport:
    """As :func:`spectral_report`, for a 2-D ``torch.Tensor``.

    Tensors with more than two dimensions (e.g. conv kernels) are reshaped to
    ``(out_channels, -1)``, which is the usual convention.
    """
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return spectral_report(arr, method=method, **kwargs)
