"""The non-stationary regime: when FDR-1 does not apply, and how we say so.

Cross-entropy on separable data has no stationary weight norm. Once the model
separates the data, max-margin dynamics drives ``|w| -> inf`` (Soudry et al.,
JMLR 2018), the update points away from the origin on average, and the
identity FDR-1 was derived from -- stationarity of ``0.5|w|^2`` -- has no
solution. ``rho`` then goes negative and stays there.

This is the single most likely thing a new user hits, because the default
PyTorch recipe (cross-entropy, no weight decay) is exactly that case. These
tests pin the behaviour: report it, name it, and say what to do about it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from physfdt import FDRAccumulator, FDRConfig, NumpyFDRMonitor


def separable_logistic(weight_decay, steps=40_000, eta=0.05, seed=0,
                       d=20, n=3000, batch=32, config=None):
    """SGD on logistic regression over linearly separable data."""
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(d)
    w_true /= np.linalg.norm(w_true)
    X = rng.standard_normal((n, d))
    margin = X @ w_true
    keep = np.abs(margin) > 0.3          # a clean separation gap
    X, margin = X[keep], margin[keep]
    y = (margin > 0).astype(float)

    w = 0.1 * rng.standard_normal(d)
    mon = NumpyFDRMonitor(
        config or FDRConfig(half_life=500, min_steps=500, patience=100,
                            warn_nonstationary=False)
    )
    rhos, norms, states = [], [], []
    for _ in range(steps):
        i = rng.integers(0, len(X), batch)
        Xb, yb = X[i], y[i]
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -60, 60)))
        u = Xb.T @ (p - yb) / batch + weight_decay * w
        st = mon.observe(w, u, eta)
        rhos.append(st.rho)
        norms.append(np.linalg.norm(w))
        states.append(st)
        w = w - eta * u
    return np.array(rhos), np.array(norms), states


def test_no_weight_decay_gives_negative_rho_and_growing_norm():
    """The reported failure mode, pinned as expected behaviour."""
    rho, norm, states = separable_logistic(weight_decay=0.0)
    assert norm[-1] > 5 * norm[0], f"|w| {norm[0]:.3f} -> {norm[-1]:.3f}"
    assert rho[-1] < 0, f"final rho = {rho[-1]}"
    assert states[-1].regime == "norm_growing_fast"
    assert not states[-1].equilibrated


def test_weight_decay_restores_stationarity():
    """The same run, with a confining term, has a bounded norm and rho ~ 1."""
    rho, norm, states = separable_logistic(weight_decay=1e-2)
    tail = float(np.mean(rho[-5000:]))
    assert abs(tail - 1.0) < 0.25, f"mean rho = {tail}"
    assert norm[-1] < 5.0, f"|w| ended at {norm[-1]:.3f}"
    assert states[-1].regime != "norm_growing_fast"


def test_logistic_is_a_hard_case_and_says_so():
    """Near-separable classification is hard for this estimator, and the
    library reports that rather than firing anyway.

    As the margin grows, almost every sample's gradient collapses towards zero,
    so ``<|u|^2>`` is small and dominated by a few hard examples. The
    denominator of ``rho`` is then very noisy: here the scatter is O(1), far
    above any sensible ``tol``. ``tol_is_achievable`` is the guard against
    reading anything into ``equilibrated`` in that situation.
    """
    _, _, states = separable_logistic(weight_decay=1e-2)
    hard = states[-1]
    assert hard.rho_std > 0.3, f"rho_std = {hard.rho_std}"
    assert not hard.tol_is_achievable

    # Contrast: on a well-conditioned quadratic the same machinery sits four
    # times inside its own noise floor at the same settings.
    rng = np.random.default_rng(0)
    a = np.linspace(0.5, 5.0, 20)
    w, eta = np.ones(20), 0.02
    mon = NumpyFDRMonitor(FDRConfig(half_life=500, min_steps=500, patience=100,
                                    warn_nonstationary=False))
    for _ in range(40_000):
        u = a * w + 0.5 * rng.standard_normal(20)
        easy = mon.observe(w, u, eta)
        w = w - eta * u
    assert easy.rho_std < 0.02, f"rho_std = {easy.rho_std}"
    assert easy.tol_is_achievable
    assert easy.equilibrated


def test_stationary_norm_shrinks_with_more_confinement():
    """More weight decay -> smaller stationary norm. Sanity on the mechanism."""
    finals = []
    for wd in (1e-2, 5e-2):
        _, norm, _ = separable_logistic(weight_decay=wd, steps=20_000)
        finals.append(norm[-1])
    assert finals[0] > finals[1], f"|w| did not shrink with wd: {finals}"


def test_warning_fires_once_and_is_actionable():
    cfg = FDRConfig(half_life=200, nonstationary_patience=200,
                    warn_nonstationary=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        separable_logistic(weight_decay=0.0, steps=5000, config=cfg)
    msgs = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(msgs) == 1, f"expected exactly one warning, got {len(msgs)}"
    text = str(msgs[0].message).lower()
    assert "no stationary state" in text
    assert "weight decay" in text


def test_warning_can_be_silenced():
    cfg = FDRConfig(half_life=200, nonstationary_patience=200,
                    warn_nonstationary=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        separable_logistic(weight_decay=0.0, steps=5000, config=cfg)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


# --------------------------------------------------------------------------
# Regime classification, driven directly rather than through a training loop.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lhs,rhs,want",
    [
        (2.0, 2.0, "equilibrated"),        # rho = 1
        (10.0, 2.0, "norm_shrinking"),     # rho = 5
        (1.0, 2.0, "norm_growing"),        # rho = 0.5
        (-4.0, 2.0, "norm_growing_fast"),  # rho = -2
    ],
)
def test_regime_labels(lhs, rhs, want):
    acc = FDRAccumulator(FDRConfig(half_life=2, warn_nonstationary=False))
    for _ in range(50):
        st = acc.observe(lhs, rhs)
    assert st.regime == want, f"rho={st.rho:.3f} labelled {st.regime}"


def test_nonstationary_run_resets_when_rho_turns_positive():
    acc = FDRAccumulator(FDRConfig(half_life=2, warn_nonstationary=False))
    for _ in range(30):
        st = acc.observe(-4.0, 2.0)
    assert st.nonstationary_run == 30
    for _ in range(30):
        st = acc.observe(2.0, 2.0)
    assert st.nonstationary_run == 0
    assert st.regime == "equilibrated"


def test_norm_drift_identity():
    """E[d|w|^2] = eta^2 <|u|^2> (1 - rho) -- the relation that makes the
    regime labels exact rather than heuristic. Checked directly."""
    rng = np.random.default_rng(3)
    eta, d = 0.05, 12
    for _ in range(200):
        w = rng.standard_normal(d)
        u = rng.standard_normal(d)
        rho = 2.0 * float(w @ u) / (eta * float(u @ u))
        predicted = eta**2 * float(u @ u) * (1.0 - rho)
        actual = float(np.sum((w - eta * u) ** 2) - np.sum(w**2))
        assert predicted == pytest.approx(actual, rel=1e-9, abs=1e-12)
