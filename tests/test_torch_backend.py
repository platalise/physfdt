"""Verify the PyTorch backend's arithmetic against the validated NumPy path.

If real PyTorch is installed this exercises it directly. If not, a minimal shim
implementing exactly the handful of tensor operations `torch_backend` uses is
installed into ``sys.modules`` instead. Either way the assertion is the same:
``delta`` mode and ``grad`` mode must both reproduce
:func:`physfdt.fdr_terms`, which is itself checked against a closed-form
stationary state in ``test_validation_quadratic.py``.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types

import numpy as np
import pytest

from physfdt import fdr_terms

try:  # pragma: no cover
    import torch as _real_torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# --------------------------------------------------------------------------
# Minimal stand-in for the torch surface that torch_backend touches.
# --------------------------------------------------------------------------
class _T:
    __slots__ = ("a", "grad")

    def __init__(self, a, grad=None):
        self.a = np.asarray(a, dtype=np.float64)
        self.grad = grad

    def detach(self):
        return _T(self.a)

    def clone(self):
        return _T(self.a.copy())

    def sub(self, other):
        return _T(self.a - other.a)

    def add(self, other, alpha=1.0):
        return _T(self.a + alpha * other.a)

    def __mul__(self, other):
        return _T(self.a * (other.a if isinstance(other, _T) else other))

    __rmul__ = __mul__


def _make_shim():
    m = types.ModuleType("torch")
    m.no_grad = contextlib.nullcontext
    m.sum = lambda t: float(np.sum(t.a))
    m.Tensor = _T
    optim = types.ModuleType("torch.optim")

    class Optimizer:  # only needed for the type annotation
        pass

    optim.Optimizer = Optimizer
    m.optim = optim
    return m


@pytest.fixture
def backend(monkeypatch):
    """Return the torch_backend module, with real torch or the shim behind it."""
    if HAVE_TORCH:
        import physfdt.torch_backend as tb

        return tb, _real_torch
    shim = _make_shim()
    monkeypatch.setitem(sys.modules, "torch", shim)
    monkeypatch.setitem(sys.modules, "torch.optim", shim.optim)
    import physfdt.torch_backend as tb

    importlib.reload(tb)
    return tb, shim


class FakeSGD:
    """A hand-rolled plain-SGD optimiser over the shim tensors."""

    def __init__(self, params, lr, weight_decay=0.0):
        self.param_groups = [
            {"params": params, "lr": lr, "weight_decay": weight_decay, "momentum": 0}
        ]

    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                u = p.grad.a + g["weight_decay"] * p.a
                p.a = p.a - g["lr"] * u


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(HAVE_TORCH, reason="shim path only; real torch covered below")
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_delta_mode_matches_fdr_terms(backend, weight_decay):
    tb, _ = backend
    rng = np.random.default_rng(0)
    lr = 0.03
    w = [_T(rng.standard_normal((4, 5))), _T(rng.standard_normal(7))]
    for p in w:
        p.grad = _T(rng.standard_normal(p.a.shape))
    opt = FakeSGD(w, lr=lr, weight_decay=weight_decay)

    # expected, from the validated NumPy path
    u = [p.grad.a + weight_decay * p.a for p in w]
    want_lhs, want_rhs = fdr_terms([p.a for p in w], u, lr)

    mon = tb.FDRMonitor.__new__(tb.FDRMonitor)      # bypass the torch import guard
    tb.FDRMonitor.__init__(mon, opt, mode="delta", every=1)
    mon.pre_step()
    opt.step()
    state = mon.post_step()

    assert state.lhs == pytest.approx(want_lhs, rel=1e-9)
    assert state.rhs == pytest.approx(want_rhs, rel=1e-9)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(HAVE_TORCH, reason="shim path only")
def test_grad_mode_matches_delta_mode(backend):
    tb, _ = backend
    rng = np.random.default_rng(1)
    lr = 0.03

    def fresh():
        r = np.random.default_rng(1)
        ps = [_T(r.standard_normal((4, 5))), _T(r.standard_normal(7))]
        for p in ps:
            p.grad = _T(r.standard_normal(p.a.shape))
        return ps

    a, b = fresh(), fresh()
    opt_a, opt_b = FakeSGD(a, lr), FakeSGD(b, lr)

    m1 = tb.FDRMonitor.__new__(tb.FDRMonitor)
    tb.FDRMonitor.__init__(m1, opt_a, mode="delta")
    m1.pre_step()
    opt_a.step()
    s1 = m1.post_step()

    m2 = tb.FDRMonitor.__new__(tb.FDRMonitor)
    tb.FDRMonitor.__init__(m2, opt_b, mode="grad")
    m2.pre_step()
    s2 = m2.post_step()
    opt_b.step()

    assert s1.lhs == pytest.approx(s2.lhs, rel=1e-9)
    assert s1.rhs == pytest.approx(s2.rhs, rel=1e-9)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(HAVE_TORCH, reason="shim path only")
def test_every_skips_measurements(backend):
    tb, _ = backend
    rng = np.random.default_rng(2)
    w = [_T(rng.standard_normal(6))]
    w[0].grad = _T(rng.standard_normal(6))
    opt = FakeSGD(w, lr=0.01)
    mon = tb.FDRMonitor.__new__(tb.FDRMonitor)
    tb.FDRMonitor.__init__(mon, opt, mode="delta", every=3)

    measured = 0
    for _ in range(9):
        mon.pre_step()
        opt.step()
        if mon.post_step() is not None:
            measured += 1
    assert measured == 3


# --------------------------------------------------------------------------
# Real-torch path: an actual nn.Module, if torch happens to be installed.
# --------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_TORCH, reason="requires PyTorch")
def test_real_torch_delta_matches_fdr_terms():
    import torch
    import torch.nn as nn

    import physfdt.torch_backend as tb

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 5), nn.ReLU(), nn.Linear(5, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.02)
    x, y = torch.randn(16, 8), torch.randint(0, 3, (16,))

    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()

    w = [p.detach().clone().numpy() for p in model.parameters()]
    u = [p.grad.detach().clone().numpy() for p in model.parameters()]
    want_lhs, want_rhs = fdr_terms(w, u, 0.02)

    mon = tb.FDRMonitor(opt, mode="delta")
    mon.pre_step()
    opt.step()
    state = mon.post_step()

    assert state.lhs == pytest.approx(want_lhs, rel=1e-4)
    assert state.rhs == pytest.approx(want_rhs, rel=1e-4)


@pytest.mark.skipif(not HAVE_TORCH, reason="requires PyTorch")
def test_scheduler_decays_and_resets():
    import torch
    import torch.nn as nn

    import physfdt.torch_backend as tb
    from physfdt import FDRConfig

    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    mon = tb.FDRMonitor(opt, FDRConfig(half_life=5, tol=1e9, patience=1, min_steps=1))
    sched = tb.FDREquilibriumLR(opt, mon, factor=0.5, min_lr=1e-3)

    x, y = torch.randn(8, 4), torch.randn(8, 2)
    for _ in range(3):
        opt.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        with mon.measure():
            opt.step()
        sched.step()

    assert sched.n_decays >= 1
    assert opt.param_groups[0]["lr"] < 0.1
    assert mon.acc.steps_since_reset <= 1     # reset fired with the decay
