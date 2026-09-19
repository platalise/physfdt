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

``rho -> 1`` iff the weight norm has stopped drifting, i.e. the system has
equilibrated *at the current learning rate*. Continuing to train at that
learning rate buys nothing further.

Interpretation depends on the optimiser:

* **Plain SGD** (``u = grad``): this is exactly the fluctuation-dissipation
  relation of Yaida (arXiv:1810.00004). The left side is dissipation, the right
  side is the fluctuation (second moment of the mini-batch gradient, noise
  included). It assumes neither Gaussian nor isotropic gradient noise -- only
  stationarity. This is the regime where the physical reading holds.
* **Any other optimiser**: the identity is still algebraically exact and remains
  a valid *stationarity diagnostic*, but the fluctuation-dissipation reading is
  not established. ``physfdt`` reports this distinction rather than hiding it.

Notes
-----
The ratio is formed from separately smoothed numerator and denominator
(ratio of averages), never from an average of per-step ratios: the per-step
ratio has heavy tails and its mean is not the quantity of interest.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional

__all__ = ["FDRConfig", "FDRState", "FDRAccumulator"]


@dataclass
class FDRConfig:
    """Settings for the equilibrium detector.

    Parameters
    ----------
    half_life:
        Number of *measured* steps over which the exponential moving average
        loses half its weight. Larger = smoother, slower to react. A useful
        default is a few times the mini-batch correlation time.
    tol:
        Equilibrium is declared when ``|rho - 1| < tol``.
    patience:
        ``|rho - 1| < tol`` must hold for this many consecutive measured steps.
        Guards against a transient crossing of 1.
    min_steps:
        No equilibrium can be declared until this many steps have been measured
        since the last reset. Prevents firing on the bias-corrected garbage of
        the first few observations.
    warn_nonstationary:
        Emit a one-time warning once ``rho`` has been negative for
        ``nonstationary_patience`` consecutive measured steps. A sustained
        negative ``rho`` means the run has no stationary state, so nothing the
        detector reports is meaningful; silence here would look like a broken
        library rather than a diagnosed setup.
    nonstationary_patience:
        How many consecutive negative-``rho`` steps trigger that warning.
    """

    # Defaults are chosen to be SELF-CONSISTENT: the sampling scatter of rho is
    # ~1/sqrt(half_life) (measured: 3 sigma ~ 0.059 at half_life=200, ~0.025 at
    # 500), so tol must sit comfortably above it or the detector is thresholding
    # its own noise. half_life=500 with tol=0.10 leaves 3 sigma four times
    # inside the band. Check yours with FDRState.tol_is_achievable.
    half_life: int = 500
    tol: float = 0.10
    patience: int = 50
    min_steps: int = 500
    warn_nonstationary: bool = True
    nonstationary_patience: int = 500

    def __post_init__(self) -> None:
        if self.half_life < 1:
            raise ValueError("half_life must be >= 1")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")


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
    """``lhs - rhs``, in the raw units of the loss. Sign carries information:
    positive means the weight norm is still contracting."""
    regime: str
    """What the weight norm is doing, from the exact identity
    ``E[d|w|^2] = eta^2 <|u|^2> (1 - rho)``:

    ``"norm_growing_fast"``
        ``rho < 0``, i.e. ``<w . u> < 0``. The update points *away* from the
        origin on average, so the norm grows on both terms at once. There is no
        stationary state and FDR-1 does not apply. See
        :attr:`FDRState.nonstationary`.
    ``"norm_growing"``
        ``0 <= rho < 1 - tol``. Norm still growing, but the dissipative term is
        pulling back.
    ``"equilibrated"``
        ``|rho - 1| < tol``.
    ``"norm_shrinking"``
        ``rho > 1 + tol``. The usual transient when starting away from the
        minimum; the norm is contracting toward its stationary value.
    """
    nonstationary_run: int
    """Consecutive measured steps spent in ``"norm_growing_fast"``. A sustained
    run means the run has no stationary state at all -- usually cross-entropy on
    separable data with no weight decay, where max-margin dynamics drives
    ``|w| -> inf`` (Soudry et al., JMLR 2018). Add weight decay (or any
    confining term) and ``rho`` becomes meaningful."""
    rho_std: float
    """Measured scatter of ``rho`` about its own slow mean -- the noise floor of
    this estimator at this ``half_life``. ``tol`` below ``3 * rho_std`` means the
    detector is thresholding its own sampling noise; see
    :attr:`tol_is_achievable`."""
    in_band: bool
    """Whether ``|rho - 1| < tol`` on this step."""
    band_run: int
    """How many consecutive steps ``in_band`` has held."""
    equilibrated: bool
    """Whether the full criterion (band + patience + min_steps) is met."""

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

    _tol: float = 0.0
    """Copy of the configured tolerance, so the state is self-describing."""


class FDRAccumulator:
    """Exponential moving averages of the two sides of FDR-1.

    This class knows nothing about tensors. Feed it two scalars per step and it
    tracks the ratio and the equilibrium criterion. The PyTorch and NumPy
    backends are thin adapters that compute those two scalars.

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
        self.step = 0
        self._warned = False
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the moving averages.

        Call this whenever the learning rate changes: FDR-1 is a statement
        about the stationary state *at a given eta*, so the old history is
        about a different equilibrium and must not be mixed in.
        """
        self._s_lhs = 0.0
        self._s_rhs = 0.0
        self._n = 0
        self._band_run = 0
        self._nonstat_run = 0
        self._rho_window = deque(maxlen=min(20 * self.config.half_life, 20_000))
        self._w_s1 = 0.0
        self._w_s2 = 0.0
        self.steps_since_reset = 0

    # ------------------------------------------------------------------
    def observe(self, lhs: float, rhs: float) -> FDRState:
        """Record one measurement of the two sides and update the state.

        Parameters
        ----------
        lhs:
            ``2 * sum_p (w_p . u_p)`` summed over all parameters.
        rhs:
            ``sum_p eta_p * |u_p|^2`` summed over all parameters. Note ``eta``
            sits inside the sum so that per-parameter-group learning rates are
            handled correctly.
        """
        d = self._decay
        self._s_lhs = d * self._s_lhs + (1.0 - d) * float(lhs)
        self._s_rhs = d * self._s_rhs + (1.0 - d) * float(rhs)
        self._n += 1
        self.step += 1
        self.steps_since_reset += 1

        # Bias correction (Adam-style). It is the same factor on both sides so
        # it cancels in rho, but the individual terms are reported to the user
        # and should be unbiased.
        bias = 1.0 - d**self._n
        lhs_hat = self._s_lhs / bias
        rhs_hat = self._s_rhs / bias

        rho = lhs_hat / rhs_hat if rhs_hat != 0.0 else float("nan")

        # Noise floor of rho: the scatter of the smoothed estimator over a
        # trailing window. A window is used rather than another EMA because an
        # EMA never fully forgets the opening transient -- where rho can be in
        # the tens -- and would report that transient as noise for tens of
        # thousands of steps afterwards. The window must span many fluctuation
        # times of the estimator itself (timescale ~half_life), hence 20x;
        # rho_std stays NaN until it is full rather than reporting a number it
        # cannot yet support.
        #
        # Running sums keep this O(1) per step. Values are accumulated as
        # (rho - 1) so that the sum of squares stays small and the variance
        # does not come out of a cancellation between two large numbers.
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

        in_band = (rho == rho) and abs(rho - 1.0) < self.config.tol
        self._band_run = self._band_run + 1 if in_band else 0

        # E[d|w|^2] = eta^2 <|u|^2> (1 - rho), so the sign of (1 - rho) says
        # what the norm is doing, and rho < 0 means <w . u> < 0: the update
        # points away from the origin and no stationary state exists.
        if rho != rho:  # NaN
            regime = "unknown"
        elif rho < 0.0:
            regime = "norm_growing_fast"
        elif in_band:
            regime = "equilibrated"
        elif rho < 1.0:
            regime = "norm_growing"
        else:
            regime = "norm_shrinking"

        self._nonstat_run = (
            self._nonstat_run + 1 if regime == "norm_growing_fast" else 0
        )
        if (
            self.config.warn_nonstationary
            and not self._warned
            and self._nonstat_run >= self.config.nonstationary_patience
        ):
            self._warned = True
            warnings.warn(
                f"rho has been negative for {self._nonstat_run} consecutive steps "
                f"(currently {rho:.3g}): the weight norm is growing without bound, so "
                f"this run has NO stationary state and FDR-1 does not apply to it. The "
                f"usual cause is cross-entropy on separable/interpolating data with no "
                f"weight decay, where max-margin dynamics drives |w| -> inf. Add weight "
                f"decay (1e-2 is a good starting point) or another confining term, and "
                f"rho becomes meaningful. Silence this with "
                f"FDRConfig(warn_nonstationary=False).",
                RuntimeWarning,
                stacklevel=3,
            )

        equilibrated = (
            self._band_run >= self.config.patience
            and self.steps_since_reset >= self.config.min_steps
        )

        return FDRState(
            step=self.step,
            steps_since_reset=self.steps_since_reset,
            lhs=lhs_hat,
            rhs=rhs_hat,
            rho=rho,
            residual=lhs_hat - rhs_hat,
            regime=regime,
            nonstationary_run=self._nonstat_run,
            rho_std=rho_std,
            _tol=self.config.tol,
            in_band=in_band,
            band_run=self._band_run,
            equilibrated=equilibrated,
        )
