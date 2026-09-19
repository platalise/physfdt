"""PyTorch backend: :class:`FDRMonitor` and :class:`FDREquilibriumLR`.

Two measurement modes:

``mode="delta"`` (default)
    Snapshot the parameters before ``optimizer.step()`` and recover the update
    direction from the actual parameter change, ``u = -(w_next - w) / eta``.
    Exact for *any* optimiser, including Adam, momentum and weight decay,
    because it reads what the optimiser really did. Costs one parameter-sized
    buffer (fp32: 4 bytes per parameter).

``mode="grad"``
    Read ``u`` straight from ``p.grad`` (plus decoupled weight decay if
    configured). No extra memory, but only correct for plain SGD without
    momentum. The monitor refuses to run in this mode if it detects momentum.

Overhead is one pass over the parameters per measured step. Use ``every=N`` to
measure once every ``N`` steps; with ``every=20`` the cost is negligible even
for large models. **If you are benchmarking wall-clock or energy savings, keep
the monitor enabled in the baseline arm too, or subtract its cost explicitly.**
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable, Optional

from .core import FDRAccumulator, FDRConfig, FDRState

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["FDRMonitor", "FDREquilibriumLR"]


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "physfdt.torch_backend requires PyTorch. Install it with "
            "`pip install physfdt[torch]` or `pip install torch`."
        ) from exc
    return torch


class FDRMonitor:
    """Measure the FDR-1 stationarity ratio during PyTorch training.

    Parameters
    ----------
    optimizer:
        The optimiser being monitored. Learning rates are read from
        ``optimizer.param_groups`` on every measured step, so external
        schedulers are picked up automatically.
    config:
        Detector settings, see :class:`~physfdt.core.FDRConfig`.
    mode:
        ``"delta"`` or ``"grad"``. See module docstring.
    every:
        Measure once every ``every`` optimiser steps. Untouched steps cost
        nothing.

    Examples
    --------
    ::

        monitor = FDRMonitor(optimizer, FDRConfig(half_life=200))

        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()

            with monitor.measure():
                optimizer.step()

            if monitor.last is not None and monitor.last.equilibrated:
                ...  # decay the learning rate, then monitor.reset()
    """

    def __init__(
        self,
        optimizer: "torch.optim.Optimizer",
        config: Optional[FDRConfig] = None,
        mode: str = "delta",
        every: int = 1,
    ) -> None:
        _require_torch()
        if mode not in {"delta", "grad"}:
            raise ValueError("mode must be 'delta' or 'grad'")
        if every < 1:
            raise ValueError("every must be >= 1")

        self.optimizer = optimizer
        self.acc = FDRAccumulator(config)
        self.mode = mode
        self.every = every

        self._calls = 0
        self._snapshot: Optional[list] = None
        self._armed = False
        self.last: Optional[FDRState] = None

        self._warn_about_optimizer()

    # ------------------------------------------------------------------
    def _warn_about_optimizer(self) -> None:
        name = type(self.optimizer).__name__
        has_momentum = any(
            g.get("momentum", 0) not in (0, None) or "betas" in g
            for g in self.optimizer.param_groups
        )
        if self.mode == "grad" and has_momentum:
            raise ValueError(
                f"mode='grad' is only valid for plain SGD without momentum, but "
                f"{name} carries momentum state. Use mode='delta'."
            )
        if name not in ("SGD",) or has_momentum:
            warnings.warn(
                f"FDR-1 is exact as a stationarity identity for any optimiser, but its "
                f"fluctuation-dissipation reading (Yaida 2019) is established only for "
                f"plain SGD. With {name} treat rho as a stationarity diagnostic, not as "
                f"a measured effective temperature.",
                stacklevel=3,
            )

    def _params(self) -> Iterable:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None or self.mode == "delta":
                    yield group, p

    # ------------------------------------------------------------------
    def pre_step(self) -> None:
        """Call immediately *before* ``optimizer.step()``."""
        self._calls += 1
        self._armed = self._calls % self.every == 0
        if not self._armed:
            return
        if self.mode == "delta":
            torch = _require_torch()
            with torch.no_grad():
                self._snapshot = [p.detach().clone() for _, p in self._params()]

    def post_step(self) -> Optional[FDRState]:
        """Call immediately *after* ``optimizer.step()``. Returns the new state,
        or ``None`` on steps that were skipped because of ``every``."""
        if not self._armed:
            return None
        self._armed = False
        torch = _require_torch()

        lhs = 0.0
        rhs = 0.0
        with torch.no_grad():
            if self.mode == "delta":
                if self._snapshot is None:
                    raise RuntimeError("post_step() called without a matching pre_step()")
                for (group, p), w0 in zip(self._params(), self._snapshot):
                    eta = float(group["lr"])
                    if eta == 0.0:
                        continue
                    delta = p.detach().sub(w0)          # w_next - w
                    # u = -delta / eta
                    lhs += -2.0 * float(torch.sum(w0 * delta)) / eta
                    rhs += float(torch.sum(delta * delta)) / eta
                self._snapshot = None
            else:  # grad mode: plain SGD, u = grad + wd * w
                for group, p in self._params():
                    if p.grad is None:
                        continue
                    eta = float(group["lr"])
                    wd = float(group.get("weight_decay", 0.0))
                    w = p.detach()
                    u = p.grad.detach()
                    if wd:
                        u = u.add(w, alpha=wd)
                    lhs += 2.0 * float(torch.sum(w * u))
                    rhs += eta * float(torch.sum(u * u))

        self.last = self.acc.observe(lhs, rhs)
        return self.last

    # ------------------------------------------------------------------
    def measure(self):
        """Context manager wrapping ``optimizer.step()``.

        ``with monitor.measure(): optimizer.step()`` is equivalent to
        ``monitor.pre_step(); optimizer.step(); monitor.post_step()``.

        In ``grad`` mode the measurement happens on entry, before the step, so
        the gradients are still the ones belonging to ``w_t``.
        """
        return _MeasureCtx(self)

    def reset(self) -> None:
        """Clear the moving averages. Call after every learning-rate change."""
        self.acc.reset()
        self.last = None

    @property
    def config(self) -> FDRConfig:
        return self.acc.config


class _MeasureCtx:
    def __init__(self, monitor: FDRMonitor) -> None:
        self.m = monitor

    def __enter__(self):
        self.m.pre_step()
        return self.m

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.m.post_step()
        else:
            self.m._armed = False
            self.m._snapshot = None
        return False


class FDREquilibriumLR:
    """Decay the learning rate when FDR-1 says the system has equilibrated.

    This reproduces the adaptive scheduler of Yaida (arXiv:1810.00004) on top of
    :class:`FDRMonitor`. It is not a novel algorithm and should not be presented
    as one; it is here so that the diagnostic is usable end to end and so that
    published results can be reproduced.

    Parameters
    ----------
    factor:
        Multiply the learning rate by this on each trigger.
    min_lr:
        Never decay below this. Once reached, the scheduler stops firing.
    max_decays:
        Optional cap on the number of decays.

    Examples
    --------
    ::

        monitor = FDRMonitor(optimizer)
        sched = FDREquilibriumLR(optimizer, monitor, factor=0.5, min_lr=1e-6)

        with monitor.measure():
            optimizer.step()
        sched.step()          # decays and resets the monitor when triggered
    """

    def __init__(
        self,
        optimizer: "torch.optim.Optimizer",
        monitor: FDRMonitor,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        max_decays: Optional[int] = None,
    ) -> None:
        if not 0.0 < factor < 1.0:
            raise ValueError("factor must be in (0, 1)")
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.min_lr = min_lr
        self.max_decays = max_decays
        self.n_decays = 0
        self.history: list[tuple[int, float, float]] = []
        """``(step, lr_before, rho_at_trigger)`` for each decay."""

    def step(self) -> bool:
        """Decay if the monitor reports equilibrium. Returns whether it fired."""
        state = self.monitor.last
        if state is None or not state.equilibrated:
            return False
        if self.max_decays is not None and self.n_decays >= self.max_decays:
            return False

        fired = False
        for group in self.optimizer.param_groups:
            lr = float(group["lr"])
            if lr <= self.min_lr:
                continue
            new_lr = max(lr * self.factor, self.min_lr)
            group["lr"] = new_lr
            fired = True

        if fired:
            self.n_decays += 1
            self.history.append((state.step, lr, state.rho))
            # The equilibrium is defined per learning rate: the old averages
            # describe a different stationary state and must be discarded.
            self.monitor.reset()
        return fired
