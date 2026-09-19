# physfdt

**Fluctuation–dissipation diagnostics for stochastic gradient descent.**

`physfdt` answers one question during training: *has this run equilibrated at
the current learning rate?* It answers it from the optimiser's own arithmetic —
no held-out validation split, no assumption about the shape of the gradient
noise.

```python
from physfdt import FDRMonitor, FDRConfig, FDREquilibriumLR

monitor = FDRMonitor(optimizer, FDRConfig(half_life=500))
sched   = FDREquilibriumLR(optimizer, monitor, factor=0.5)

loss.backward()
with monitor.measure():
    optimizer.step()
sched.step()                      # decays the LR when rho settles at 1
```

---

## The relation

Any first-order optimiser can be written `w_{t+1} = w_t − η·u_t`, where `u_t` is
whatever direction it produces before the learning rate is applied. Take the
observable `O(w) = ½|w|²` and require its expectation to be stationary:

```
|w_{t+1}|² = |w_t|² − 2η ⟨w_t · u_t⟩ + η² ⟨|u_t|²⟩

  ⟹   2⟨w · u⟩ = η ⟨|u|²⟩                                          (FDR-1)
```

Define

```
ρ  =  2⟨w · u⟩ / (η ⟨|u|²⟩)
```

At stationarity `ρ = 1` exactly: the system has equilibrated at that η, and
further steps at that η buy nothing. (The converse needs care — see *Reading ρ*
below.) `physfdt` tracks `ρ` with exponentially smoothed numerator and denominator (a ratio of
averages, never an average of ratios) and reports when it settles.

For **plain SGD** (`u = g`) this is the fluctuation–dissipation relation of
Yaida, [arXiv:1810.00004](https://arxiv.org/abs/1810.00004): the left side is
dissipation, the right side is the second moment of the mini-batch gradient,
noise included. It assumes neither Gaussian nor isotropic noise — only
stationarity. That is what makes it different from effective-temperature
definitions that need a noise model.

For **other optimisers** the identity remains algebraically exact and is still a
valid stationarity diagnostic, but the fluctuation–dissipation reading is not
established. `physfdt` warns rather than hides this.

## Reading ρ

The same algebra gives an exact drift law, `E[Δ|w|²] = η²⟨|u|²⟩(1 − ρ)`. It
is tempting to read the norm's behaviour off the sign of `1 − ρ`. **Don't.**
ρ is a ratio whose denominator `η⟨|u|²⟩` can be tiny and dominated by a few
samples; measured on near-separable logistic regression with weight decay, ρ
was negative on 16,451 of 40,000 steps while `|w|` sat at 3.67 to three
decimals. So `physfdt` takes the verdict on the norm from `|w|²` directly — a
trend test over the last `4 × half_life` steps that must be both statistically
significant (`norm_z`) and physically non-negligible (`norm_rel_tol`, 0.5%) —
and uses ρ only for what it is good at: *given* a stationary norm, has the FDR
balance been reached.

| `regime` | meaning |
|---|---|
| `norm_growing` / `norm_shrinking` | `|w|²` is trending. Not equilibrated, whatever ρ says. |
| `equilibrated` | norm flat **and** `|ρ − 1| < tol` for `patience` steps |
| `stationary_noisy` | norm flat but ρ outside the band — ρ is too noisy at this `half_life`; compare `rho_std` with `tol` |
| `warming_up` | not enough history for a norm verdict yet |

### If the norm keeps growing

Two different situations produce a sustained `norm_growing`, and they need
opposite responses:

- **No weight decay, cross-entropy.** Once the data is separated, max-margin
  dynamics drives `|w| → ∞`
  ([Soudry et al., JMLR 2018](https://jmlr.org/papers/v19/18-188.html)). There
  is **no stationary state**; FDR-1 never applies. Add weight decay.
- **Weight decay present.** The equilibrium norm exists but has moved — for
  instance after a learning-rate decay, or when starting from a small
  initialisation. This is a transient. Train longer.

Measured on logistic regression over separable data, 40k steps:

| weight decay | \|w\| start → end | final ρ | |
|---|---|---|---|
| 0 | 0.46 → **11.07** | **−14294** | diverging, no stationary state |
| 1e-4 | 0.46 → 10.15 | −4614 | diverging |
| 1e-3 | 0.46 → 6.90 | 0.39 | not yet settled |
| **1e-2** | 0.46 → **3.67** | **1.07** | stationary |
| **5e-2** | 0.46 → **2.07** | **1.02** | stationary |

`physfdt` raises a one-time `RuntimeWarning` naming both causes once the norm
has grown for `nonstationary_patience` (default `4 × half_life`) steps.

### Check `tol` against the noise floor

ρ is an exponentially smoothed estimator, so it has sampling scatter of order
`1/√half_life`. If `tol` is below that scatter, `equilibrated` is being decided
by noise — and the detector typically never fires at all. Every state reports
its own measured noise floor:

```python
state.rho_std              # measured scatter of rho
state.tol_is_achievable    # False => tol is below 3 * rho_std
```

Measured on the solvable quadratic:

| `half_life` | `rho_std` | 3σ | `tol=0.05` usable? |
|---|---|---|---|
| 100 | 0.036 | 0.107 | no |
| 200 | 0.018 | 0.055 | no |
| 500 | 0.0077 | 0.023 | yes |
| 2000 | 0.0019 | 0.006 | yes |

The defaults (`half_life=500`, `tol=0.10`) are self-consistent by construction.
Badly conditioned problems need much more smoothing: near-separable
classification collapses most per-sample gradients, leaving `⟨|u|²⟩` small and
dominated by a few hard examples, and ρ can carry O(1) scatter there. Check
`tol_is_achievable` before believing `equilibrated`.

## Attribution

The FDR relations and the adaptive scheduler built on them are Yaida's, not
ours. This package exists to make them easy to measure and to reproduce, and to
pair them with the spectral observables (`α`, IPR) from
[Physica A **692** (2026) 131474](https://doi.org/10.1016/j.physa.2026.131474).
If you use the scheduler, cite Yaida.

---

## Install

```bash
pip install physfdt              # core, NumPy only
pip install "physfdt[torch]"     # + the PyTorch monitor and scheduler
```

From source:

```bash
git clone https://github.com/platalise/physfdt && cd physfdt
pip install -e ".[dev]"
pytest -q
```

The core has **no torch dependency**. `import physfdt` works in a NumPy-only
environment; the torch symbols are imported lazily.

## Validate it before you trust it

```bash
python examples/01_validate_quadratic.py
```

Runs SGD on `L(w) = ½ wᵀAw` with noisy gradients. That system is an
Ornstein–Uhlenbeck process whose stationary covariance provably satisfies FDR-1,
so `ρ → 1` is an exact prediction rather than a fit. Expected output:

```
     eta    mean rho (last 20k)        std   verdict
   0.005                 1.0002     0.0204   OK
   0.010                 1.0000     0.0119   OK
   0.020                 0.9999     0.0055   OK
   0.050                 1.0000     0.0024   OK
   0.100                 1.0000     0.0011   OK
```

Part 2 of the same script starts far from the minimum (`ρ ≈ 36`), watches the
detector fire four learning-rate cuts, and shows `ρ` returning to 1 after each.

## Monitor a real run

```bash
python examples/02_torch_quickstart.py                    # offline, synthetic
python examples/02_torch_quickstart.py --dataset mnist    # real data
python examples/02_torch_quickstart.py --schedule fdr     # detector drives the LR
```

Writes a CSV with one row per measured step: `rho`, both sides of FDR-1, the
learning rate, the loss, and the spectral exponent `α` of a probe layer.

## Benchmark it honestly

```bash
python examples/03_benchmark_arms.py --seeds 5
```

Four arms — constant, **tuned** cosine, ReduceLROnPlateau, FDR — on an identical
learning-rate tuning grid, with the monitor's overhead measured and subtracted.
The script prints three pre-registered failure criteria and the numbers needed
to check them. Read them before you read the result.

---

## API

| | |
|---|---|
| `FDRConfig(half_life, tol, patience, min_steps)` | detector settings; defaults are self-consistent (see *Reading ρ*) |
| `state.regime`, `state.norm_trend`, `state.rho_std`, `state.tol_is_achievable` | is this measurement meaningful? |
| `FDRMonitor(optimizer, config, mode, every)` | torch monitor; `mode="delta"` (any optimiser) or `"grad"` (plain SGD, zero extra memory). Config timescales are in **optimiser** steps whatever `every` is. |
| `FDREquilibriumLR(optimizer, monitor, factor, min_lr)` | decay on equilibrium |
| `NumpyFDRMonitor(config)` | toy models and hand-written optimisers |
| `spectral_report(W, method="mle"\|"window")` | `α`, its standard error, R², IPR |
| `TraceWriter(path, extra=[...])` | CSV trace, one row per measured step |

**Cost.** One pass over the parameters per measured step, plus one
parameter-sized buffer in `delta` mode. **Measured: at `every=1` the monitor took
~39% of wall clock** on a small MLP on CPU — the per-step tensor work is
comparable to the training step itself when the model is tiny. Use `every=20`
(the benchmark default) to amortise it to ~2%. If you are benchmarking
wall-clock or energy, keep the monitor on in the baseline arm too
(`--monitor-in-baseline`), or subtract it explicitly.

**Always `reset()` after a learning-rate change.** FDR-1 describes the stationary
state *at a given η*; mixing histories across learning rates is meaningless.
`FDREquilibriumLR` does this for you.

---

## Scope and limitations

Read this before putting a number in a paper.

- **It measures stationarity, not progress.** `ρ = 1` says the weight norm has
  stopped drifting at this η. It does not say the loss is low or that the model
  generalises.
- **Single-epoch pre-training is out of scope.** Large language models are
  trained under a fixed compute budget for roughly one pass, and stop because the
  budget ran out, not because they equilibrated. Such a run is never stationary
  and FDR-1 does not apply to it. The regime where this tool is useful is
  **multi-epoch fine-tuning on small data** — where runs do reach stationarity,
  and where a held-out split is expensive precisely because labelled data is
  scarce.
- **Momentum and Adam.** The identity holds; the physics does not transfer
  unchanged. Treat `ρ` as a stationarity diagnostic there, not as a measured
  effective temperature.
- **It has hyperparameters.** `tol`, `patience` and `half_life` are real choices,
  and they are *coupled*: `tol` must sit above the `1/√half_life` noise floor or
  the detector never fires. A method that needs them retuned per workload is not
  hyperparameter-free, and its net saving is smaller than a naive comparison
  suggests. This is failure criterion F2 in the benchmark script.
- **It needs a confining term.** Plain cross-entropy with no weight decay has no
  stationary state at all. See *Reading ρ* above.
- **Small `η` inflates the smoothing timescale.** After a decay, allow at least
  a few `half_life` before trusting `ρ` again. `min_steps` enforces a floor.
- **No claim of compute savings is made here.** Whether FDR-triggered decay beats
  a properly tuned cosine schedule is an empirical question that
  `examples/03_benchmark_arms.py` exists to settle, in either direction.

## Tests

```bash
pytest -q          # 51 tests (2 need torch)
```

The physics tests are the ones that matter. `tests/test_validation_quadratic.py`
checks `ρ → 1` against a case with a closed-form answer, across five learning
rates, three noise magnitudes, and — the one that matters most — strongly
anisotropic noise, where model-based effective temperatures break down.
`tests/test_nonstationary_regime.py` pins the opposite case: that a run with no
stationary state is detected and named rather than silently mis-reported.

## Citing

```bibtex
@software{physfdt,
  title  = {physfdt: fluctuation-dissipation diagnostics for SGD},
  author = {Nguyen, Quang},
  year   = {2026},
  url    = {https://github.com/platalise/physfdt}
}
```

Please also cite Yaida (arXiv:1810.00004) for the FDR relations and the
scheduler.

## Licence

MIT.
