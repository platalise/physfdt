"""The non-stationary regime: when FDR-1 does not apply, and how we say so.

Cross-entropy on separable data has no stationary weight norm. Once the model
separates the data, max-margin dynamics drives ``|w| -> inf`` (Soudry et al.,
JMLR 2018), the update points away from the origin on average, and the
identity FDR-1 was derived from -- stationarity of ``0.5|w|^2`` -- has no
solution. ``rho`` then goes negative and stays there.

This is the single most likely thing a new user hits, because the default
PyTorch recipe (cross-entropy, no weight decay) is exactly that case.

The converse matters as much: a negative ``rho`` is NOT by itself evidence of
a growing norm. In near-separable problems ``rho`` is noisy enough to go
negative while ``|w|`` is perfectly stationary. The verdict on the norm is
therefore taken from ``|w|^2`` directly, and these tests pin both directions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from physfdt import FDRAccumulator, FDRConfig, NonStationaryWarning, NumpyFDRMonitor


def separable_logistic(weight_decay, steps=40_000, eta=0.05, seed=0,
                       d=20, n=3000, batch=32, config=None):
    """SGD on logistic regression over linearly separable data."""
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(d)
    w_true /= np.linalg.norm(w_true)
    X = rng.standard_normal((n, d))
    # w_true is a unit vector and X is standard-normal, so X @ w_true cannot
    # genuinely overflow float64. Some BLAS backends (observed: numpy 2.3 +
    # Apple Accelerate) raise spurious divide-by-zero/overflow/invalid-value
    # RuntimeWarnings from matmul on perfectly finite input -- an artifact of
    # the backend, not of this data. errstate suppresses it at the source
    # rather than letting it pollute anyone's warnings.catch_warnings().
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
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
    assert states[-1].regime == "norm_growing"
    assert states[-1].norm_trend > 0
    assert states[-1].nonstationary_run > 1000
    assert not any(s.equilibrated for s in states)


def test_weight_decay_restores_stationarity():
    """The same run, with a confining term, has a bounded norm and rho ~ 1."""
    rho, norm, states = separable_logistic(weight_decay=1e-2)
    tail = float(np.mean(rho[-5000:]))
    assert abs(tail - 1.0) < 0.25, f"mean rho = {tail}"
    assert norm[-1] < 5.0, f"|w| ended at {norm[-1]:.3f}"
    # The norm verdict must say "not drifting", even though rho is noisy here.
    late = states[-5000:]
    drifting = sum(s.regime in ("norm_growing", "norm_shrinking") for s in late)
    assert drifting / len(late) < 0.2, f"{drifting}/{len(late)} late steps flagged"


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
    # Filter on NonStationaryWarning specifically, not the bare RuntimeWarning
    # category: RuntimeWarning is also what NumPy itself uses for
    # floating-point warnings, so a blanket filter can catch noise that has
    # nothing to do with physfdt (see the errstate note above).
    msgs = [w for w in caught if issubclass(w.category, NonStationaryWarning)]
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
    assert not [w for w in caught if issubclass(w.category, NonStationaryWarning)]


# --------------------------------------------------------------------------
# Regime classification, driven directly rather than through a training loop.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lhs,rhs,want",
    [
        (2.0, 2.0, "equilibrated"),     # rho = 1
        (10.0, 2.0, "norm_shrinking"),  # rho = 5
        (1.0, 2.0, "norm_growing"),     # rho = 0.5
        (-4.0, 2.0, "norm_growing"),    # rho = -2
    ],
)
def test_regime_fallback_labels_without_norm(lhs, rhs, want):
    """With no |w|^2 supplied, labels fall back to the sign of 1 - rho."""
    acc = FDRAccumulator(FDRConfig(half_life=2, warn_nonstationary=False))
    for _ in range(50):
        st = acc.observe(lhs, rhs)
    assert st.regime == want, f"rho={st.rho:.3f} labelled {st.regime}"


def test_negative_rho_with_stationary_norm_is_not_called_growing():
    """The bug behind the quickstart log: rho < 0 while |w| is flat must not be
    reported as a growing norm."""
    acc = FDRAccumulator(FDRConfig(half_life=5, warn_nonstationary=False))
    rng = np.random.default_rng(0)
    for _ in range(3000):
        st = acc.observe(-4.0, 2.0, norm2=10.0 + 0.1 * rng.standard_normal())
    assert st.rho < 0
    assert st.regime == "stationary_noisy"
    assert st.nonstationary_run == 0
    assert not st.equilibrated


def test_growing_norm_is_detected_from_norm_not_rho():
    """And the converse: a trending |w|^2 is flagged even when rho sits at 1.

    The trend window spans `norm_window_factor * half_life` measured steps
    (20 here) -- a short, fast-reacting window by design. For a steady linear
    drift, the relative signal visible *within* that window shrinks like
    ~window / (2 * elapsed) the longer the run has already gone on, since the
    baseline itself has grown too. So this must be checked while still close
    to the window filling, not arbitrarily late in a long run: at t=3000 the
    same drift that's obvious over the run's full history is only ~0.3%
    within the last 20 steps -- correctly below norm_rel_tol, and invisible
    to a window that short by construction, not a bug.
    """
    acc = FDRAccumulator(FDRConfig(half_life=5, warn_nonstationary=False))
    for t in range(100):
        st = acc.observe(2.0, 2.0, norm2=1.0 + 0.01 * t)
    assert abs(st.rho - 1.0) < 1e-9
    assert st.regime == "norm_growing"
    assert not st.equilibrated


def test_lr_decay_with_weight_decay_is_not_flagged_as_growing():
    """The quickstart log: after an LR halving with weight decay present, rho
    went negative and the old code said 'no stationary state; add weight
    decay'. With weight decay present |w| stays put after the decay, so the
    norm must never be judged to be growing -- however negative rho gets.

    (The opening phase, from a small initialisation, *is* genuine growth and
    is allowed to warn; this test is about what happens after the decay.)
    """
    cfg = FDRConfig(half_life=500, warn_nonstationary=False)
    rng = np.random.default_rng(0)
    d = 20
    wt = rng.standard_normal(d)
    wt /= np.linalg.norm(wt)
    X = rng.standard_normal((3000, d))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = X @ wt
    keep = np.abs(m) > 0.3
    X, m = X[keep], m[keep]
    y = (m > 0).astype(float)
    w = 0.1 * rng.standard_normal(d)
    mon = NumpyFDRMonitor(cfg)
    eta, wd = 0.05, 1e-2

    def run(n):
        nonlocal w
        out = []
        for _ in range(n):
            i = rng.integers(0, len(X), 32)
            p = 1.0 / (1.0 + np.exp(-np.clip(X[i] @ w, -60, 60)))
            u = X[i].T @ (p - y[i]) / 32 + wd * w
            out.append(mon.observe(w, u, eta))
            w = w - eta * u
        return out

    run(40_000)
    eta = 0.025
    mon.reset()
    after = run(40_000)

    assert sum(s.rho < 0 for s in after) > 1000, "precondition: rho goes negative"
    assert max(s.nonstationary_run for s in after) == 0
    late = after[-20_000:]
    assert sum(s.regime == "stationary_noisy" for s in late) / len(late) > 0.8
    # rho is far too noisy here to certify equilibrium, and it must say so.
    assert not late[-1].tol_is_achievable


def test_patience_default_blocks_firing_mid_transient():
    """The first log's premature decay: rho was still sliding down through the
    band (1.43 -> 1.21 -> 1.08) when patience=50 let it fire. Default
    patience is now half_life."""
    assert FDRConfig(half_life=500).patience == 500
    assert FDRConfig(half_life=500).min_steps == 500
    # A rho that drifts linearly from 1.5 to 0.5 over 2000 steps spends ~400
    # steps in the band [0.9, 1.1]; with patience = half_life = 500 it must not
    # fire, even though the norm is supplied as flat.
    acc = FDRAccumulator(FDRConfig(half_life=500, warn_nonstationary=False))
    fired = False
    for t in range(2000):
        r = 1.5 - t / 2000
        st = acc.observe(2.0 * r, 2.0, norm2=5.0)
        fired |= st.equilibrated
    assert not fired


def test_config_rescaled_by_every():
    cfg = FDRConfig(half_life=500).rescaled(20)
    assert cfg.half_life == 25
    assert cfg.patience == 25
    assert cfg.min_steps == 25
    assert cfg.nonstationary_patience == 100
    assert FDRConfig(half_life=500).rescaled(1).half_life == 500


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
