"""Validation against an analytically solvable case.

Loss ``L(w) = 0.5 w^T A w`` with an unbiased mini-batch gradient
``u = A w + xi``, ``xi ~ N(0, sigma^2 I)``. SGD is then a discrete
Ornstein-Uhlenbeck process whose stationary covariance ``Sigma`` obeys

    Sigma = (I - eta A) Sigma (I - eta A)^T + eta^2 sigma^2 I

Taking the trace after expanding gives

    2 tr(A Sigma) = eta [ tr(A Sigma A) + N sigma^2 ]

whose left and right sides are exactly the two sides of FDR-1. So ``rho -> 1``
is not an approximation here: it is an identity of the stationary state, and
these tests check that the estimator recovers it.
"""

from __future__ import annotations

import numpy as np
import pytest

from physfdt import FDRConfig, NumpyFDRMonitor


def run_sgd(
    eigenvalues,
    eta=0.02,
    sigma=0.5,
    steps=40_000,
    w0_scale=1.0,
    seed=0,
    config=None,
):
    """Run SGD on a diagonal quadratic and return (monitor, rho trace)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(eigenvalues, dtype=np.float64)
    n = len(a)
    w = w0_scale * np.ones(n)
    mon = NumpyFDRMonitor(config or FDRConfig(half_life=500, min_steps=500, patience=100))
    rhos = np.empty(steps)
    for t in range(steps):
        u = a * w + sigma * rng.standard_normal(n)
        rhos[t] = mon.observe(w, u, eta).rho
        w = w - eta * u
    return mon, rhos


def test_rho_converges_to_one():
    """At stationarity rho must equal 1 to within sampling error."""
    a = np.linspace(0.5, 5.0, 20)
    _, rhos = run_sgd(a, eta=0.02, sigma=0.5, steps=60_000, seed=1)
    tail = rhos[-20_000:]
    assert abs(np.mean(tail) - 1.0) < 0.02, f"mean rho = {np.mean(tail)}"


def test_rho_converges_for_several_learning_rates():
    """The identity must hold at every stable learning rate, not just one."""
    a = np.linspace(0.5, 5.0, 20)
    for eta in (0.005, 0.01, 0.05, 0.1):
        _, rhos = run_sgd(a, eta=eta, sigma=0.5, steps=60_000, seed=2)
        m = float(np.mean(rhos[-20_000:]))
        assert abs(m - 1.0) < 0.05, f"eta={eta}: mean rho = {m}"


def test_rho_is_large_during_transient():
    """Starting far from the minimum, dissipation dominates and rho >> 1."""
    a = np.linspace(0.5, 5.0, 20)
    _, rhos = run_sgd(a, eta=0.02, sigma=0.05, steps=40_000, w0_scale=50.0, seed=3)
    assert rhos[200] > 3.0, f"early rho = {rhos[200]}"
    assert abs(np.mean(rhos[-10_000:]) - 1.0) < 0.05


def test_equilibrium_is_detected_and_not_too_early():
    a = np.linspace(0.5, 5.0, 20)
    cfg = FDRConfig(half_life=300, tol=0.05, patience=100, min_steps=1000)
    rng = np.random.default_rng(4)
    n, eta, sigma = len(a), 0.02, 0.3
    w = 30.0 * np.ones(n)
    mon = NumpyFDRMonitor(cfg)
    first_eq = None
    for t in range(60_000):
        u = a * w + sigma * rng.standard_normal(n)
        st = mon.observe(w, u, eta)
        if st.equilibrated and first_eq is None:
            first_eq = t
        w = w - eta * u
    assert first_eq is not None, "equilibrium was never detected"
    assert first_eq >= cfg.min_steps
    # It must not fire while the weight norm is still collapsing by orders of
    # magnitude. Relaxation time here is ~1/(eta*a_min) = 100 steps.
    assert first_eq > 1000, f"fired at step {first_eq}, too early"


def test_reset_clears_history():
    a = np.linspace(0.5, 5.0, 20)
    mon, _ = run_sgd(a, steps=20_000, seed=5)
    assert mon.acc.steps_since_reset == 20_000
    mon.reset()
    assert mon.acc.steps_since_reset == 0
    assert mon.acc._n == 0


@pytest.mark.parametrize("sigma", [0.05, 0.5, 2.0])
def test_rho_independent_of_noise_scale(sigma):
    """FDR-1 assumes nothing about the noise magnitude."""
    a = np.linspace(0.5, 5.0, 20)
    _, rhos = run_sgd(a, eta=0.02, sigma=sigma, steps=60_000, seed=6)
    m = float(np.mean(rhos[-20_000:]))
    assert abs(m - 1.0) < 0.05, f"sigma={sigma}: mean rho = {m}"


def test_rho_with_anisotropic_noise():
    """Nor does it assume isotropic noise -- the usual weak point of
    effective-temperature definitions."""
    rng = np.random.default_rng(7)
    a = np.linspace(0.5, 5.0, 20)
    scale = np.geomspace(0.05, 5.0, 20)  # strongly anisotropic
    n, eta = len(a), 0.02
    w = np.ones(n)
    mon = NumpyFDRMonitor(FDRConfig(half_life=500, min_steps=500, patience=100))
    rhos = []
    for _ in range(60_000):
        u = a * w + scale * rng.standard_normal(n)
        rhos.append(mon.observe(w, u, eta).rho)
        w = w - eta * u
    m = float(np.mean(rhos[-20_000:]))
    assert abs(m - 1.0) < 0.05, f"anisotropic: mean rho = {m}"
