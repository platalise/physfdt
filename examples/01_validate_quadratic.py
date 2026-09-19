#!/usr/bin/env python3
"""Validate the FDR estimator on an analytically solvable system. No torch needed.

    python examples/01_validate_quadratic.py

Loss L(w) = 0.5 w^T A w with mini-batch gradient u = A w + xi. SGD is then a
discrete Ornstein-Uhlenbeck process, and at stationarity the two sides of FDR-1
are provably equal, so rho -> 1 is an exact prediction rather than a fit.

Part 2 shows what the scheduler is built on: after a learning-rate cut, the
system leaves its old equilibrium, rho departs from 1, and then returns.
"""

from __future__ import annotations

import numpy as np

from physfdt import FDRConfig, NumpyFDRMonitor


def sparkline(values, lo, hi, width=60):
    blocks = "▁▂▃▄▅▆▇█"
    idx = np.clip(((np.asarray(values) - lo) / (hi - lo) * 7).round(), 0, 7).astype(int)
    step = max(1, len(idx) // width)
    return "".join(blocks[i] for i in idx[::step][:width])


def main() -> None:
    rng = np.random.default_rng(0)
    a = np.linspace(0.5, 5.0, 32)          # curvature spectrum
    n, sigma = len(a), 0.4

    print("=" * 70)
    print("PART 1  rho -> 1 at stationarity, for every stable learning rate")
    print("=" * 70)
    print(f"{'eta':>8} {'mean rho (last 20k)':>22} {'std':>10}   verdict")
    for eta in (0.005, 0.01, 0.02, 0.05, 0.10):
        w = np.ones(n)
        mon = NumpyFDRMonitor(FDRConfig(half_life=500, min_steps=500, patience=100))
        rhos = []
        for _ in range(60_000):
            u = a * w + sigma * rng.standard_normal(n)
            rhos.append(mon.observe(w, u, eta).rho)
            w = w - eta * u
        tail = np.array(rhos[-20_000:])
        ok = "OK" if abs(tail.mean() - 1) < 0.05 else "FAIL"
        print(f"{eta:>8.3f} {tail.mean():>22.4f} {tail.std():>10.4f}   {ok}")

    print()
    print("=" * 70)
    print("PART 2  transient -> equilibrium -> LR cut -> new equilibrium")
    print("=" * 70)

    eta = 0.02
    w = 40.0 * np.ones(n)                  # start far from the minimum
    cfg = FDRConfig(half_life=400, tol=0.05, patience=100, min_steps=1500)
    mon = NumpyFDRMonitor(cfg)

    rhos, lrs, events = [], [], []
    for t in range(90_000):
        u = a * w + sigma * rng.standard_normal(n)
        st = mon.observe(w, u, eta)
        rhos.append(st.rho)
        lrs.append(eta)
        if st.equilibrated and eta > 2e-3:
            events.append((t, eta, st.rho))
            eta *= 0.5
            mon.reset()                    # equilibrium is defined per eta
        w = w - eta * u

    rhos = np.array(rhos)
    print(f"start: |w| = {40.0 * np.sqrt(n):.1f}, eta = 0.02, N = {n}")
    print()
    print(f"  rho(t), clipped to [0, 3]   {sparkline(np.clip(rhos, 0, 3), 0, 3)}")
    print(f"  {'':<28}{'step 0':<20}{'':<20}{'step 90k':>12}")
    print()
    print("  learning-rate decays fired by the detector:")
    print(f"    {'step':>8} {'eta before':>12} {'eta after':>12} {'rho at trigger':>16}")
    for t, lr, rho in events:
        print(f"    {t:>8} {lr:>12.5f} {lr / 2:>12.5f} {rho:>16.4f}")
    if not events:
        print("    (none)")

    print()
    print(f"  early rho (step 300)      : {rhos[300]:8.3f}   far from equilibrium")
    print(f"  final rho (last 5k mean)  : {rhos[-5000:].mean():8.3f}   back at 1")
    print(f"  |w| at the end            : {np.linalg.norm(w):8.3f}")
    print()
    print("  Note: no validation set was used anywhere above.")


if __name__ == "__main__":
    main()
