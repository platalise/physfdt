"""Framework-free core of the FDR stationarity diagnostic.

Physics
-------
Any first-order optimiser can be written as

    w_{t+1} = w_t - eta * u_t

where ``u_t`` is whatever direction the optimiser produces before the learning
rate is applied (plain gradient, momentum buffer, Adam's preconditioned step,
gradient plus weight decay, ...).

Take the observable ``O(w) = 0.5 * |w|^2`` and impose stationarity of its
expectation, ``<O(w_{t+1})> = <O(w_t)>``:

    |w_{t+1}|^2 = |w_t|^2 - 2 eta <w_t . u_t> + eta^2 <|u_t|^2>

    =>   2 <w_t . u_t>  =  eta <|u_t|^2>                              (FDR-1)

Define the dimensionless ratio

    rho = 2 <w_t . u_t> / (eta <|u_t|^2>)

At stationarity ``rho = 1``. Continuing to train at that learning rate buys
nothing further.

Interpretation depends on the optimiser:

* **Plain SGD** (``u = grad``): this is exactly the fluctuation-dissipation
  relation of Yaida (arXiv:1810.00004). The left side is dissipation, the right
  side is the fluctuation (second moment of the mini-batch gradient, noise
  included). It assumes neither Gaussian nor isotropic gradient noise -- only
  stationarity. This is the regime where the physical reading holds.
* **Any other optimiser**: the identity is still algebraically exact and remains
  a valid *stationarity diagnostic*, but the fluctuation-dissipation reading is
  not established. ``physfdt`` reports this distinction rather than hiding it.

Two independent signals
-----------------------
``rho`` is a ratio of two smoothed averages, and its denominator
``eta <|u|^2>`` can be tiny and dominated by a few samples (near-separable
classification is the standard example). There ``rho`` swings through large
positive *and negative* values while the weight norm is perfectly stationary.
So the sign of ``rho`` is **not** used to decide whether the norm is drifting.
That verdict comes from the norm itself: backends pass ``|w|^2`` alongside the
two FDR-1 terms, and the accumulator tests for a trend in it directly. ``rho``
then answers the narrower question it is good at -- *given* a stationary norm,
has the FDR balance been reached.

Notes
-----
The ratio is formed from separately smoothed numerator and denominator
(ratio of averages), never from an average of per-step ratios: the per-step
ratio has heavy tails and its mean is not the quantity of interest.

All timescales in :class:`FDRConfig` count *measured* steps of the accumulator.
The PyTorch monitor converts them from optimiser steps when ``every > 1``.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional

__all__ = ["FDRConfig", "FDRState", "FDRAccumulator", "NonStationaryWarning"]


class NonStationaryWarning(RuntimeWarning):
    """Raised once, when the weight norm has been growing for
    ``nonstationary_patience`` consecutive measured steps -- i.e. FDR-1 does
    not (yet) apply to this run.

    Subclasses ``RuntimeWarning`` so existing ``except RuntimeWarning`` or
    ``filterwarnings("...", RuntimeWarning)`` code keeps working. But prefer
    filtering on ``NonStationaryWarning`` specifically: ``RuntimeWarning`` is
    also the category NumPy itself uses for floating-point warnings (divide
    by zero, overflow, invalid value -- these can come from a BLAS backend on
    perfectly finite input, e.g. a known Apple Accelerate artifact), so
    catching the bare category can silently pick up noise that has nothing to
    do with stationarity.
    """

#: Number of |w|^2 samples kept for the trend test. The window spans
#: ``norm_window_factor * half_life`` measured steps, subsampled to this many
#: points, so the cost is bounded regardless of half_life.
_NORM_POINTS = 200


@dataclass
class FDRConfig:
    """Settings for the equilibrium detector.

    Parameters
    ----------
    half_life:
        Measured steps over which the exponential moving averages of the two
        FDR-1 terms lose half their weight. Sets the scatter of ``rho``
        (``~ 1/sqrt(half_life)``) and every other timescale below.
    tol:
        ``rho`` counts as balanced when ``|rho - 1| < tol``.
    patience:
        ``|rho - 1| < tol`` must hold for this many consecutive measured steps.
        Default: ``half_life``. Anything much shorter lets ``rho`` qualify while
        it is still drifting through the band at the tail of a transient,
        because the smoothed estimator cannot move appreciably in fewer steps
        than its own half-life.
    min_steps:
        No equilibrium can be declared this soon after a reset. Default:
        ``half_life``.
    norm_window_factor:
        The trend test on ``|w|^2`` looks back over
        ``norm_window_factor * half_life`` measured steps. Longer windows are
        less easily fooled by slow fluctuations; shorter ones react faster.
    norm_z:
        The norm is called drifting when the difference between the means of
        the two halves of the window exceeds ``norm_z`` standard deviations of
        ``|w|^2`` over the window. A pure linear trend scores ``sqrt(3) ~ 1.73``,
        so the default of 1 catches trend-dominated windows and ignores
        fluctuation-dominated ones.
    norm_rel_tol:
        ...and the relative change of ``|w|^2`` across the window must also
        exceed this. Statistical significance alone is not enough: at a
        weight-decay equilibrium ``|w|`` fluctuates so little that a physically
        negligible creep (0.03%) can still score ``z > 1``. Measured on
        separable logistic regression: with no weight decay the late-time trend
        is ~1.5% per window; with weight decay 1e-2 its 95th percentile is
        ~0.36%. The default 0.5% sits between them.
    warn_nonstationary:
        Emit a one-time warning after the norm has been growing for
        ``nonstationary_patience`` consecutive measured steps.
    nonstationary_patience:
        Default: ``4 * half_life``.
    """

    # Defaults are chosen to be SELF-CONSISTENT: the sampling scatter of rho is
    # ~1/sqrt(half_life) (measured: 3 sigma ~ 0.059 at half_life=200, ~0.025 at
    # 500), so tol must sit comfortably above it or the detector is thresholding
    # its own noise. Check yours with FDRState.tol_is_achievable.
    half_life: int = 500
    tol: float = 0.10
    patience: Optional[int] = None
    min_steps: Optional[int] = None
    norm_window_factor: int = 4
    norm_z: float = 1.0
    norm_rel_tol: float = 0.005
    warn_nonstationary: bool = True
    nonstationary_patience: Optional[int] = None

    def __post_init__(self) -> None:
        if self.half_life < 1:
            raise ValueError("half_life must be >= 1")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")
        if self.patience is None:
            self.patience = self.half_life
        if self.min_steps is None:
            self.min_steps = self.half_life
        if self.nonstationary_patience is None:
            self.nonstationary_patience = 4 * self.half_life
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if self.norm_window_factor < 1:
            raise ValueError("norm_window_factor must be >= 1")
        if self.norm_z <= 0:
            raise ValueError("norm_z must be > 0")
        if self.norm_rel_tol < 0:
            raise ValueError("norm_rel_tol must be >= 0")

    def rescaled(self, every: int) -> "FDRConfig":
        """Return a copy with every timescale divided by ``every``.

        Use when the accumulator is fed only one step in ``every``, so that
        ``half_life=500`` keeps meaning 500 *optimiser* steps.
        """
        if every <= 1:
            return self

        def s(x: int) -> int:
            return max(1, int(round(x / every)))

        return FDRConfig(
            half_life=s(self.half_life),
            tol=self.tol,
            patience=s(self.patience),
            min_steps=s(self.min_steps),
            norm_window_factor=self.norm_window_factor,
            norm_z=self.norm_z,
            norm_rel_tol=self.norm_rel_tol,
            warn_nonstationary=self.warn_nonstationary,
            nonstationary_patience=s(self.nonstationary_patience),
        )


@dataclass
class FDRState:
    """One measurement, as returned by :meth:`FDRAccumulator.observe`."""

    step: int
    steps_since_reset: int
    lhs: float
    """Smoothed dissipation term, ``2 <w . u>``."""
    rhs: float
    """Smoothed fluctuation term, ``eta <|u|^2>``."""
    rho: float
    """``lhs / rhs``. Equals 1 at stationarity."""
    residual: float
    """``lhs - rhs``, in the raw units of the loss."""
    regime: str
    """What the run is doing.

    ``"norm_growing"`` / ``"norm_shrinking"``
        ``|w|^2`` has a significant trend over the trend window. Not
        equilibrated, whatever ``rho`` says.
    ``"equilibrated"``
        Norm has no significant trend *and* ``|rho - 1| < tol``.
    ``"stationary_noisy"``
        Norm has no significant trend but ``rho`` is outside the band. Usually
        means ``rho`` is too noisy at this ``half_life`` -- compare
        ``rho_std`` with ``tol``. Common in near-separable classification.
    ``"warming_up"``
        Not enough history yet for a norm verdict.
    ``"unknown"``
        ``rho`` undefined (zero denominator).

    When no ``norm2`` is supplied to :meth:`FDRAccumulator.observe`, the norm
    labels fall back to the sign of ``1 - rho``, which is unreliable when
    ``rho`` is noisy. Both shipped backends supply it.
    """
    norm_trend: float
    """Relative change of ``|w|^2`` between the two halves of the trend window,
    ``(mean_late - mean_early) / mean``. NaN until the window is full."""
    norm_trend_z: float
    """The same difference in units of the window's ``|w|^2`` standard
    deviation. Drifting requires ``|norm_trend_z| > norm_z`` *and*
    ``|norm_trend| > norm_rel_tol``."""
    nonstationary_run: int
    """Consecutive measured steps with the norm judged to be growing. A long run
    with no weight decay usually means there is no stationary state at all:
    cross-entropy on separable data drives ``|w| -> inf`` (Soudry et al., JMLR
    2018). With weight decay present it more often means a slow approach to a
    larger equilibrium norm, e.g. after a learning-rate decay."""
    rho_std: float
    """Measured scatter of ``rho`` about its own mean over a trailing window --
    the noise floor of the estimator. NaN until the window is full."""
    in_band: bool
    """Whether ``|rho - 1| < tol`` on this step."""
    band_run: int
    """How many consecutive steps ``in_band`` has held."""
    equilibrated: bool
    """Band held for ``patience`` steps, at least ``min_steps`` since reset, and
    the norm is not drifting."""
    _tol: float = 0.0

    @property
    def tol_is_achievable(self) -> bool:
        """Whether ``tol`` sits above this estimator's own noise floor.

        False means the band is narrower than the scatter of ``rho``, so
        ``equilibrated`` is being decided by sampling noise. Raise ``half_life``
        (scatter falls as ``1/sqrt(half_life)``) or widen ``tol``.
        """
        if self.rho_std != self.rho_std or self.rho_std == 0.0:
            return True
        return self._tol > 3.0 * self.rho_std


class FDRAccumulator:
    """Exponential moving averages of the two sides of FDR-1, plus a direct
    trend test on the weight norm.

    This class knows nothing about tensors. Feed it scalars each step. The
    PyTorch and NumPy backends are thin adapters that compute them.

    Examples
    --------
    >>> acc = FDRAccumulator(FDRConfig(half_life=10, patience=2, min_steps=3))
    >>> for _ in range(20):
    ...     s = acc.observe(lhs=2.0, rhs=2.0)
    >>> round(s.rho, 6)
    1.0
    >>> s.equilibrated
    True
    """

    def __init__(self, config: Optional[FDRConfig] = None) -> None:
        self.config = config or FDRConfig()
        self._decay = 0.5 ** (1.0 / self.config.half_life)
        span = self.config.norm_window_factor * self.config.half_life
        self._norm_stride = max(1, span // _NORM_POINTS)
        # Number of points the window actually holds. When span >= _NORM_POINTS
        # this is _NORM_POINTS (bounded cost, subsampled by stride above). When
        # span < _NORM_POINTS (small half_life), stride is 1 and the window must
        # hold only `span` points -- NOT _NORM_POINTS. Getting this wrong means
        # the window needs _NORM_POINTS=200 raw ticks to fill no matter how
        # small half_life is, silently overriding the documented contract that
        # the trend test looks back over `norm_window_factor * half_life`
        # measured steps, and making `min_steps`/`patience` unable to produce a
        # verdict faster than 200 steps even when set to 1.
        self._norm_win_len = max(2, min(_NORM_POINTS, span // self._norm_stride))
        self.step = 0
        self._warned = False
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all history.

        Call this whenever the learning rate changes: FDR-1 is a statement
        about the stationary state *at a given eta*, so the old history is
        about a different equilibrium and must not be mixed in.
        """
        self._s_lhs = 0.0
        self._s_rhs = 0.0
        self._n = 0
        self._band_run = 0
        self._nonstat_run = 0
        self._rho_window: deque = deque(
            maxlen=min(20 * self.config.half_life, 20_000)
        )
        self._w_s1 = 0.0
        self._w_s2 = 0.0
        self._norm_win: deque = deque(maxlen=self._norm_win_len)
        self._norm_tick = 0
        self._norm_trend = float("nan")
        self._norm_z = float("nan")
        self._saw_norm = False
        self.steps_since_reset = 0

    # ------------------------------------------------------------------
    def _update_norm(self, norm2: float) -> None:
        self._saw_norm = True
        self._norm_tick += 1
        if self._norm_tick % self._norm_stride:
            return
        self._norm_win.append(float(norm2))
        win = self._norm_win
        if len(win) < win.maxlen:
            return
        vals = list(win)
        half = len(vals) // 2
        m1 = sum(vals[:half]) / half
        m2 = sum(vals[half:]) / (len(vals) - half)
        mean = (m1 + m2) / 2.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        self._norm_trend = (m2 - m1) / mean if mean > 0 else float("nan")
        if sd > 0:
            self._norm_z = (m2 - m1) / sd
        else:
            self._norm_z = 0.0 if m2 == m1 else math.copysign(math.inf, m2 - m1)

    # ------------------------------------------------------------------
    def observe(self, lhs: float, rhs: float, norm2: Optional[float] = None) -> FDRState:
        """Record one measurement and update the state.

        Parameters
        ----------
        lhs:
            ``2 * sum_p (w_p . u_p)`` summed over all parameters.
        rhs:
            ``sum_p eta_p * |u_p|^2`` summed over all parameters. ``eta`` sits
            inside the sum so per-group learning rates are handled correctly.
        norm2:
            ``|w|^2`` before the step. Strongly recommended: it is what decides
            whether the norm is drifting. Without it the regime falls back to
            the sign of ``1 - rho``, which noise can flip.
        """
        cfg = self.config
        d = self._decay
        self._s_lhs = d * self._s_lhs + (1.0 - d) * float(lhs)
        self._s_rhs = d * self._s_rhs + (1.0 - d) * float(rhs)
        self._n += 1
        self.step += 1
        self.steps_since_reset += 1

        bias = 1.0 - d**self._n
        lhs_hat = self._s_lhs / bias
        rhs_hat = self._s_rhs / bias
        rho = lhs_hat / rhs_hat if rhs_hat != 0.0 else float("nan")

        # --- noise floor of rho (trailing window, O(1) running sums) -------
        if rho == rho:
            win = self._rho_window
            x = rho - 1.0
            if len(win) == win.maxlen:
                old = win[0]
                self._w_s1 -= old
                self._w_s2 -= old * old
            win.append(x)
            self._w_s1 += x
            self._w_s2 += x * x
        n = len(self._rho_window)
        if n == self._rho_window.maxlen and n > 1:
            var = (self._w_s2 - self._w_s1 * self._w_s1 / n) / (n - 1)
            rho_std = var**0.5 if var > 0.0 else 0.0
        else:
            rho_std = float("nan")

        # --- norm trend ------------------------------------------------------
        if norm2 is not None:
            self._update_norm(norm2)

        in_band = (rho == rho) and abs(rho - 1.0) < cfg.tol
        self._band_run = self._band_run + 1 if in_band else 0

        # --- regime ----------------------------------------------------------
        if rho != rho:
            regime = "unknown"
            drifting = None
        elif self._saw_norm:
            z, rel = self._norm_z, self._norm_trend
            big = rel == rel and abs(rel) > cfg.norm_rel_tol
            if z != z:
                regime, drifting = "warming_up", None
            elif z > cfg.norm_z and big:
                regime, drifting = "norm_growing", "up"
            elif z < -cfg.norm_z and big:
                regime, drifting = "norm_shrinking", "down"
            elif in_band:
                regime, drifting = "equilibrated", "no"
            else:
                regime, drifting = "stationary_noisy", "no"
        else:
            # Fallback: E[d|w|^2] = eta^2 <|u|^2> (1 - rho). Reliable only
            # when rho is well above its noise floor.
            drifting = "no" if in_band else ("up" if rho < 1.0 else "down")
            regime = {
                "no": "equilibrated",
                "up": "norm_growing",
                "down": "norm_shrinking",
            }[drifting]

        self._nonstat_run = self._nonstat_run + 1 if drifting == "up" else 0
        if (
            cfg.warn_nonstationary
            and not self._warned
            and self._nonstat_run >= cfg.nonstationary_patience
        ):
            self._warned = True
            trend = (
                f" (|w|^2 up {100 * self._norm_trend:.1f}% across the trend window)"
                if self._norm_trend == self._norm_trend
                else ""
            )
            warnings.warn(
                f"The weight norm has been growing for {self._nonstat_run} "
                f"consecutive measured steps{trend}, so this run is not stationary "
                f"and FDR-1 does not yet apply. Two common causes: (1) no weight "
                f"decay with cross-entropy on separable data -- max-margin "
                f"dynamics drives |w| -> inf and there is NO stationary state; add "
                f"weight decay. (2) weight decay present but the equilibrium norm "
                f"has moved, e.g. after a learning-rate decay -- this is a "
                f"transient; train longer. Silence with "
                f"FDRConfig(warn_nonstationary=False).",
                NonStationaryWarning,
                stacklevel=3,
            )

        equilibrated = (
            self._band_run >= cfg.patience
            and self.steps_since_reset >= cfg.min_steps
            and (drifting == "no" if self._saw_norm else True)
        )

        return FDRState(
            step=self.step,
            steps_since_reset=self.steps_since_reset,
            lhs=lhs_hat,
            rhs=rhs_hat,
            rho=rho,
            residual=lhs_hat - rhs_hat,
            regime=regime,
            norm_trend=self._norm_trend,
            norm_trend_z=self._norm_z,
            nonstationary_run=self._nonstat_run,
            rho_std=rho_std,
            in_band=in_band,
            band_run=self._band_run,
            equilibrated=equilibrated,
            _tol=cfg.tol,
        )
