# Changelog

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
