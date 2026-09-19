"""NumPy backend.

Useful for toy models, for the analytic validation in ``tests/`` and
``examples/01_validate_quadratic.py``, and for any optimiser you write by hand.

You supply ``w`` (parameters *before* the step), ``u`` (the update direction
before the learning rate multiplies it) and ``eta``. Everything else is handled
by :class:`physfdt.core.FDRAccumulator`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .core import FDRAccumulator, FDRConfig, FDRState

__all__ = ["NumpyFDRMonitor", "fdr_terms"]

ArrayLike = Union[np.ndarray, Sequence[np.ndarray]]


def _flatten(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.ravel()
    return np.concatenate([np.asarray(a).ravel() for a in x])


def fdr_terms(w: ArrayLike, u: ArrayLike, eta: float) -> tuple[float, float]:
    """Return ``(lhs, rhs)`` of FDR-1 for one step.

    ``lhs = 2 * (w . u)`` and ``rhs = eta * |u|^2``, following the convention
    ``w_next = w - eta * u``.
    """
    wf = _flatten(w)
    uf = _flatten(u)
    if wf.shape != uf.shape:
        raise ValueError(f"w and u must have the same size, got {wf.shape} vs {uf.shape}")
    lhs = 2.0 * float(np.dot(wf, uf))
    rhs = float(eta) * float(np.dot(uf, uf))
    return lhs, rhs


class NumpyFDRMonitor:
    """Thin NumPy adapter around :class:`~physfdt.core.FDRAccumulator`.

    Examples
    --------
    >>> import numpy as np
    >>> mon = NumpyFDRMonitor(FDRConfig(half_life=50, min_steps=10, patience=5))
    >>> w = np.array([1.0, 1.0])
    >>> eta = 0.05
    >>> rng = np.random.default_rng(0)
    >>> for _ in range(4000):
    ...     u = w + 0.3 * rng.standard_normal(2)   # grad of 0.5|w|^2 plus noise
    ...     state = mon.observe(w, u, eta)
    ...     w = w - eta * u
    >>> bool(abs(state.rho - 1.0) < 0.1)
    True
    """

    def __init__(self, config: Optional[FDRConfig] = None) -> None:
        self.acc = FDRAccumulator(config)

    def observe(self, w: ArrayLike, u: ArrayLike, eta: float) -> FDRState:
        lhs, rhs = fdr_terms(w, u, eta)
        wf = _flatten(w)
        return self.acc.observe(lhs, rhs, norm2=float(np.dot(wf, wf)))

    def reset(self) -> None:
        """Clear history. Call after every learning-rate change."""
        self.acc.reset()

    @property
    def config(self) -> FDRConfig:
        return self.acc.config
