from __future__ import annotations

import numpy as np
import pytest

from physfdt import FDRAccumulator, FDRConfig, esd, fdr_terms, ipr, powerlaw_alpha, spectral_report


# --------------------------------------------------------------------- core


def test_perfect_balance_gives_rho_one():
    acc = FDRAccumulator(FDRConfig(half_life=10, patience=2, min_steps=3))
    for _ in range(20):
        st = acc.observe(2.0, 2.0)
    assert st.rho == pytest.approx(1.0)
    assert st.residual == pytest.approx(0.0)
    assert st.equilibrated


def test_bias_correction_makes_first_step_unbiased():
    acc = FDRAccumulator(FDRConfig(half_life=100))
    st = acc.observe(3.0, 7.0)
    assert st.lhs == pytest.approx(3.0)
    assert st.rhs == pytest.approx(7.0)
    assert st.rho == pytest.approx(3.0 / 7.0)


def test_patience_blocks_a_transient_crossing():
    acc = FDRAccumulator(FDRConfig(half_life=2, tol=0.05, patience=5, min_steps=1))
    for _ in range(30):
        acc.observe(10.0, 1.0)          # far from equilibrium
    st = acc.observe(1.0, 1.0)          # one lucky step
    assert not st.equilibrated


def test_min_steps_blocks_early_firing():
    acc = FDRAccumulator(FDRConfig(half_life=2, tol=0.05, patience=1, min_steps=50))
    for i in range(49):
        st = acc.observe(1.0, 1.0)
        assert not st.equilibrated, f"fired at step {i}"
    assert acc.observe(1.0, 1.0).equilibrated


def test_reset_forgets_the_old_equilibrium():
    acc = FDRAccumulator(FDRConfig(half_life=5, patience=2, min_steps=3))
    for _ in range(30):
        acc.observe(1.0, 1.0)
    acc.reset()
    st = acc.observe(1.0, 1.0)
    assert not st.equilibrated
    assert st.steps_since_reset == 1


def test_ratio_of_averages_not_average_of_ratios():
    """Alternating 10:1 and 1:10 has mean ratio 5.05 but ratio of means 1.0.
    The estimator must report 1.0."""
    acc = FDRAccumulator(FDRConfig(half_life=200))
    for i in range(4000):
        st = acc.observe(10.0, 1.0) if i % 2 else acc.observe(1.0, 10.0)
    assert st.rho == pytest.approx(1.0, abs=0.05)


def test_fdr_terms_sign_and_magnitude():
    w = np.array([2.0, 0.0])
    u = np.array([1.0, 0.0])
    lhs, rhs = fdr_terms(w, u, eta=0.5)
    assert lhs == pytest.approx(4.0)     # 2 * (2*1)
    assert rhs == pytest.approx(0.5)     # 0.5 * 1


def test_fdr_terms_accepts_parameter_lists():
    w = [np.ones((2, 2)), np.ones(3)]
    u = [np.full((2, 2), 2.0), np.full(3, 2.0)]
    lhs, rhs = fdr_terms(w, u, eta=1.0)
    assert lhs == pytest.approx(2 * 2.0 * 7)
    assert rhs == pytest.approx(4.0 * 7)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fdr_terms(np.ones(3), np.ones(4), eta=0.1)


def test_invalid_config():
    for kwargs in ({"half_life": 0}, {"tol": 0}, {"patience": 0}, {"min_steps": 0}):
        with pytest.raises(ValueError):
            FDRConfig(**kwargs)


# ----------------------------------------------------------------- spectral


def _powerlaw_matrix(n=400, m=600, alpha=2.5, seed=0):
    """Build a matrix whose W W^T spectrum has a prescribed power-law tail."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # Clamp the inverse-CDF draw away from 1. Unclamped, a draw within float64
    # eps of 1 sends lam to inf. The clamp keeps four decades of power-law
    # range, which is ample for the fit -- and it is enforced below, not just
    # hoped for.
    u = np.clip(rng.uniform(size=n), 0.0, 1.0 - 1e-6)
    lam = (1.0 - u) ** (-1.0 / (alpha - 1.0))      # density ~ lam^-alpha
    assert np.isfinite(lam).all()
    R, _ = np.linalg.qr(rng.standard_normal((m, n)))
    # lam is finite (asserted above) and Q, R are orthonormal, so this product
    # cannot genuinely overflow. Some BLAS backends (observed: numpy 2.3 +
    # Apple Accelerate) raise spurious divide-by-zero/overflow/invalid-value
    # RuntimeWarnings from matmul regardless -- a backend artifact, not a
    # property of this data. Suppress at the source rather than let it leak
    # into anyone's warnings.catch_warnings().
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return Q @ np.diag(np.sqrt(lam)) @ R.T


def test_esd_matches_eigenvalues_of_wwT():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((30, 50))
    got = esd(W)
    want = np.sort(np.linalg.eigvalsh(W @ W.T))[::-1]
    assert np.allclose(got, want, atol=1e-8)


@pytest.mark.parametrize("alpha_true", [2.0, 2.5, 3.5])
def test_mle_recovers_the_planted_exponent(alpha_true):
    W = _powerlaw_matrix(alpha=alpha_true, seed=1)
    alpha, err, r2, _, k = powerlaw_alpha(esd(W), method="mle", tail_frac=0.2)
    assert abs(alpha - alpha_true) < 4 * err, f"got {alpha:.3f} +- {err:.3f}, want {alpha_true}"
    assert r2 > 0.9


def test_window_method_runs_and_is_in_the_right_ballpark():
    W = _powerlaw_matrix(alpha=2.5, seed=2)
    alpha, _, r2, _, n = powerlaw_alpha(esd(W), method="window")
    assert 1.5 < alpha < 4.0
    assert r2 > 0.8
    assert n > 100


def test_ipr_bounds():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((64, 128))
    eigs, y2 = ipr(W)
    n = W.shape[0]
    assert len(eigs) == len(y2) == n
    assert np.all(y2 >= 1.0 / n - 1e-9)   # delocalised lower bound
    assert np.all(y2 <= 1.0 + 1e-9)       # fully localised upper bound
    assert np.all(np.diff(eigs) <= 1e-9)  # sorted descending


def test_ipr_detects_a_localised_mode():
    """A rank-one spike on a single coordinate must localise: Y2 ~ 1."""
    rng = np.random.default_rng(4)
    W = 0.01 * rng.standard_normal((64, 128))
    W[0, :] += 10.0
    _, y2 = ipr(W)
    assert y2[0] > 0.9


def test_spectral_report_is_self_consistent():
    W = _powerlaw_matrix(alpha=2.5, seed=5)
    rep = spectral_report(W, method="mle")
    alpha, *_ = powerlaw_alpha(esd(W), method="mle")
    assert rep.alpha == pytest.approx(alpha)
    assert rep.ipr_mean <= rep.ipr_max
    assert rep.method == "mle"


def test_too_few_eigenvalues_raises():
    with pytest.raises(ValueError):
        powerlaw_alpha(np.arange(1, 5, dtype=float))


def test_non_2d_input_raises():
    with pytest.raises(ValueError):
        esd(np.ones((2, 2, 2)))
