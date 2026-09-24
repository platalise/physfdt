# Changelog

## 0.4.0

New capability: a second fluctuation–dissipation observable, `X = T_eff/T_bath`,
measured by a twin-trajectory response probe — and, unlike `ρ`, a *leading*
indicator of generalisation.

### `physfdt.response.fd_probe`

Measures the full fluctuation–dissipation ratio `X = T_eff/T_bath` at a frozen
checkpoint of any `torch.nn.Module`, by comparing the network's response to a
small constant force against its fluctuations in the same direction. Returns a
`ResponseState(X, T_eff, T_bath, chi, Delta, T_bath_dirs)` and restores the
model afterwards, so it leaves an in-progress training loop undisturbed. Both
are exported from the top level (`from physfdt import fd_probe, ResponseState`).

This is an *occasional, checkpoint-time* measurement (~29 000 extra
SGD-equivalent steps per call at the defaults), orthogonal to the per-step
`FDRMonitor`: `ρ` is the cheap continuous signal, `X` the expensive forecast.

### Why it matters

On the canonical grokking test-bed (2-layer network, modular addition, MSE
readout), `X` rises `0.12 → 0.53` and **saturates ~3–5× before held-out
accuracy moves** — a label-free, validation-free leading indicator that a run is
on track to generalise. `examples/05_leading_indicator.py` reproduces this and
writes `examples/figures/leading_indicator.png`. The README section
"Forecasting a run" describes the kill-or-continue / early-ranking use that
saves failed runs, sweep compute, and energy — with the scope stated plainly
(clean on quadratic/MSE losses; numerically fragile on saturating cross-entropy).

### Validation

`tests/test_response.py` checks the estimator against a closed-form multi-mode
Ornstein–Uhlenbeck system with a known `X` (equilibrium and two anisotropic
cases), to within Monte-Carlo error. Independently, the estimator reproduces the
reference protocol of Nguyen (2026) on the exact 2-layer MSE modular-addition
task to within 0.01 at every checkpoint.

## 0.3.1

Found by the user's own `pytest -q` on macOS: two tests that were green in
this project's own sandbox failed there -- `test_warning_fires_once_and_is_actionable`
got 4 warnings instead of 1, `test_warning_can_be_silenced` got 3 instead of
0. The warning-firing logic itself was not the bug; the test's filter was.

### `physfdt`'s own warning is now its own class

`observe()` used to raise a plain `RuntimeWarning`. That category is also
what NumPy itself uses for floating-point warnings (divide by zero, overflow,
invalid value), including spurious ones a BLAS backend can raise on input
that is perfectly finite -- observed on the user's machine (numpy + Apple
Accelerate) from a plain `X @ w_true` matmul that cannot mathematically
overflow. `warnings.catch_warnings()` filtered on the bare `RuntimeWarning`
category therefore counted both: physfdt's one real warning plus three
BLAS-backend artifacts, and the two tests above were checking exactly that
count.

* New `NonStationaryWarning(RuntimeWarning)`. `observe()` now raises this
  instead of `RuntimeWarning` directly (still catchable as `RuntimeWarning`
  for existing code, but filterable precisely).
* The two tests now filter on `NonStationaryWarning`.
* The test helpers' own matmuls (`separable_logistic`'s `X @ w_true`, the LR-decay
  regression test's `X @ wt`, and `test_core_and_spectral.py`'s `_powerlaw_matrix`)
  are now wrapped in `np.errstate(over="ignore", invalid="ignore", divide="ignore")`,
  so the spurious BLAS warnings are suppressed at the source rather than left
  to leak into whatever a test happens to be recording.
* Correction to the 0.2.0 entry below: the fix credited there ("fixed a float
  overflow in the power-law test fixture") clamped `lam`, which
  `assert np.isfinite(lam).all()` already showed was never actually infinite.
  The real source was the matmul itself, not `lam` -- which is why the same
  warning resurfaced in this round at the same line. Fixed properly here.

### The norm trend window ignored `half_life` below 50

Fixing the test above meant finally installing real PyTorch in a sandbox that
runs this suite (previously only reachable via the hand-written shim,
`tests/test_torch_backend.py`'s own fallback for exactly this gap) and
running `test_scheduler_decays_and_resets` for the first time. It failed:
zero decays fired in 3 steps with `patience=1, min_steps=1` -- settings that
should make equilibrium near-instant.

Root cause: the trend test on `|w|^2` subsamples its lookback window to at
most 200 points for bounded cost, via `stride = max(1, span // 200)` where
`span = norm_window_factor * half_life`. Correct when `span >= 200`. But the
window itself was hard-coded to hold exactly 200 points regardless of
`stride`, so for `span < 200` (any `half_life` under 50 at the default
`norm_window_factor=4`) the window still needed 200 raw measured steps to
fill -- ten times `span` in this test's case (`half_life=5` -> `span=20`) --
silently overriding the documented contract that the window looks back over
`norm_window_factor * half_life` steps, and any `min_steps`/`patience` a user
sets to react faster than that.

* Window length is now `min(200, span // stride)`, matching the documented
  span exactly in both regimes. Verified directly: with `half_life=5,
  norm_window_factor=4`, equilibrium is now reachable at the 20th measured
  step (previously the 200th), matching `span` on the nose.
* Default config (`half_life=500`, `span=2000 >= 200`) is untouched by this --
  `examples/01_validate_quadratic.py`'s decay steps (8076, 9676, 11276,
  13238) are unchanged byte-for-byte. This only affects `half_life < 50`.
* `test_scheduler_decays_and_resets` updated to run `norm_window_factor *
  half_life` steps (the minimum physically required, not a guess) instead of
  a fixed 3, and to break as soon as the scheduler fires rather than assuming
  a fixed step count; also seeded (`torch.manual_seed(0)`) -- it wasn't
  before, so it was flaky by construction, coincidentally never caught
  because it also always failed before this fix.
* `test_growing_norm_is_detected_from_norm_not_rho` adjusted: it checked a
  linear drift 3000 steps in with a 20-step window, where the drift's
  relative signal *within that window* has shrunk to ~0.3% (window / (2 *
  elapsed), a real property of short windows, not a bug) -- now checks near
  where the window first fills, where the same drift is clearly visible.

51 tests (49 run without torch) -- same count as 0.3.0, no tests added or
removed, only made to test what they claim to, and to actually run.

## 0.3.0

0.2.0 met a real PyTorch run and failed in three ways. All three were found by
running it; none would have been found by reading it.

### The sign of rho no longer decides whether the norm is drifting

0.2.0 labelled every negative-rho step `norm_growing_fast` and told the user
"no stationary state; add weight decay" -- including on a run that *had*
weight decay, right after an LR decay. Reproducing it showed why that reading
is wrong: on near-separable logistic regression with weight decay, rho was
negative on 16,451 of 40,000 steps after an LR halving while `|w|` stayed at
3.67 to three decimals. rho's denominator is small and noisy there; its sign
carries no information about the norm.

* Backends now pass `|w|^2`, and the verdict comes from a trend test on it:
  the late-half vs early-half difference over `4 * half_life` steps must be
  both significant (`norm_z`, default 1 sd) and non-negligible
  (`norm_rel_tol`, default 0.5%). The effect-size floor is needed because at a
  weight-decay equilibrium `|w|` barely fluctuates, so a 0.03% creep can still
  be "significant". Measured: no weight decay -> ~1.5% per window; weight
  decay 1e-2 -> 95th percentile 0.36%.
* Regimes are now `norm_growing`, `norm_shrinking`, `equilibrated`,
  `stationary_noisy` (norm flat, rho too noisy to certify), `warming_up`.
  `norm_growing_fast` is gone.
* `equilibrated` now also requires the norm to be flat.
* The non-stationarity warning names both causes (no confinement vs. a moved
  equilibrium after an LR decay or small init) instead of assuming the first.

### `patience` defaulted far too short

The quickstart fired an LR decay while rho was still sliding down through the
band (1.43 -> 1.21 -> 1.08). With `patience=50` and `half_life=500`, the
smoothed estimator physically cannot move much in 50 steps, so 50 in-band
steps are almost automatic once rho crosses 1.1. `patience` and `min_steps`
now default to `half_life`.

### `every > 1` silently stretched every timescale 20x

Config timescales count measured steps, but the monitor only measures one step
in `every`. The benchmark ran `every=20` with a hardcoded 0.1.0-era config
(`half_life=200, tol=0.05, patience=50, min_steps=300`): `min_steps` alone
meant 6,000 optimiser steps -- the entire run -- so the fdr arm could never
fire (the harness's F0 check caught this). `FDRMonitor` now treats config
timescales as optimiser steps and converts internally (`FDRConfig.rescaled`);
the benchmark uses library defaults and a reachable `--target-loss 0.45`.

51 tests (49 run without torch).

## 0.2.0

First release that survives contact with real training runs. Three things that
0.1.0 got wrong, all found by running it rather than by reading it.

### Non-stationary runs are now detected and named

Cross-entropy with no weight decay has **no stationary weight norm**: once the
data is separated, max-margin dynamics drives `|w| -> inf`, and `rho` goes
negative and stays there. 0.1.0 printed that negative number with no
explanation, which reads as a broken library rather than a diagnosed setup.

* `FDRState.regime` labels every measurement: `norm_growing_fast` (rho < 0, no
  stationary state), `norm_growing`, `equilibrated`, `norm_shrinking`.
* `FDRState.nonstationary_run` counts consecutive negative-rho steps.
* A one-time `RuntimeWarning` names the cause and the fix once that run reaches
  `FDRConfig.nonstationary_patience`.
* `examples/02` and `examples/03` now default to `weight_decay=1e-2` and say
  plainly what `weight_decay=0` does.

### `tol` is now checked against the estimator's own noise floor

`rho` is an EMA, so it has sampling scatter `~1/sqrt(half_life)`. The old
defaults (`half_life=200`, `tol=0.05`) put `tol` *below* 3 sigma of that
scatter, so the detector could not reliably fire at all.

* `FDRState.rho_std` reports the measured scatter, over a trailing window
  (O(1) per step) rather than an EMA, so the opening transient does not
  masquerade as noise.
* `FDRState.tol_is_achievable` is False when `tol < 3 * rho_std`.
* Defaults changed to `half_life=500`, `tol=0.10`, `min_steps=500`, which are
  self-consistent: 3 sigma sits four times inside the band.

### Monitor overhead is real and is now amortised

Measured at `every=1`: **~39% of wall clock** on a small MLP on CPU. The
benchmark now defaults to `every=20`, reports per-arm overhead as a percentage,
widens the learning-rate grid, and warns when the selected learning rate sits at
the edge of the grid (the optimum was not bracketed) or when the FDR arm never
fired a decay (the comparison is vacuous).

### Also

* Fixed a float overflow in the power-law test fixture that produced
  divide-by-zero warnings on some BLAS backends.
* 47 tests, up from 29.

## 0.1.0

Initial release.
