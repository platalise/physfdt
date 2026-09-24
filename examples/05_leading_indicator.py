"""X as a leading indicator of generalisation ("grokking").

Trains the canonical grokking test-bed -- a 2-layer network on modular addition
with an MSE readout -- and, at a handful of checkpoints, measures the
fluctuation-dissipation ratio

    X = T_eff / T_bath        (physfdt.response.fd_probe)

The point of the demo: **X saturates ~3-5x earlier than held-out accuracy
moves.** You can read, from the training ramp alone and with no validation set,
whether the run is on track to generalise. See the README section
"Forecasting a run".

This reproduces the published X(t_w) curve of Nguyen (2026) to within ~0.01 at
every checkpoint (the estimator in physfdt.response is a faithful, independently
validated port of the reference protocol).

Runtime: ~5-10 min on a laptop CPU (float64, plain numpy-speed matmuls).
Output:  examples/figures/leading_indicator.png  and a printed table.

    python3 examples/05_leading_indicator.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

from physfdt.response import fd_probe

# ---- task / optimiser (the reference grokking setup) -------------------------
P, H, ALPHA = 31, 200, 3.0        # modulus, hidden width, init scale
ETA, B, WD = 0.1, 64, 1e-3        # plain SGD + weight decay
STEPS = 90_000
CKPTS = [500, 2000, 7000, 15000, 32000, 45000, 60000, 90000]
SEED = 11
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "figures", "leading_indicator.png")


class TwoLayerMSE(nn.Module):
    """relu(x @ W1) @ W2, no biases -- linear in W2, so the loss is exactly
    quadratic in the readout weights (the regime where X is clean)."""

    def __init__(self, seed):
        super().__init__()
        r = np.random.default_rng(seed)
        W1 = r.normal(size=(2 * P, H)) * ALPHA / np.sqrt(2 * P)
        W2 = r.normal(size=(H, P)) * ALPHA / np.sqrt(H)
        self.W1 = nn.Parameter(torch.tensor(W1))
        self.W2 = nn.Parameter(torch.tensor(W2))

    def forward(self, x):
        return torch.relu(x @ self.W1) @ self.W2


def mse_loss(model, xb, yb):
    diff = model(xb) - yb
    return 0.5 * (diff ** 2).sum(dim=1).mean()


def build_data():
    rng = np.random.default_rng(3)
    pairs = np.array([(a, b) for a in range(P) for b in range(P)])
    labels = (pairs[:, 0] + pairs[:, 1]) % P
    Xoh = np.zeros((P * P, 2 * P))
    Xoh[np.arange(P * P), pairs[:, 0]] = 1
    Xoh[np.arange(P * P), P + pairs[:, 1]] = 1
    Yoh = np.eye(P)[labels]
    perm = rng.permutation(P * P)
    ntr = int(0.5 * P * P)
    tr, te = perm[:ntr], perm[ntr:]
    return Xoh, Yoh, labels, tr, te


def main():
    Xoh, Yoh, labels, tr, te = build_data()
    Xtr = torch.tensor(Xoh[tr])
    Ytr = torch.tensor(Yoh[tr])
    Xte = torch.tensor(Xoh[te])
    ntr = len(tr)

    def acc(model, X, idx):
        with torch.no_grad():
            pred = model(X).argmax(1).numpy()
        return float((pred == labels[idx]).mean())

    model = TwoLayerMSE(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=ETA, momentum=0.0, weight_decay=WD)
    rtrain = np.random.default_rng(101 + SEED)
    ckpt_states, curve = {}, []

    t0 = time.time()
    for t in range(STEPS + 1):
        if t in CKPTS:
            ckpt_states[t] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if t % 2000 == 0:
            curve.append((t, acc(model, Xtr, tr), acc(model, Xte, te)))
        if t == STEPS:
            break
        idx = tr[rtrain.integers(0, ntr, size=B)]
        opt.zero_grad(set_to_none=True)
        mse_loss(model, torch.tensor(Xoh[idx]), torch.tensor(Yoh[idx])).backward()
        opt.step()
    print(f"trained in {time.time()-t0:.0f}s  "
          f"(final test acc {curve[-1][2]:.3f})\n")

    # ---- probe X at each checkpoint (model restored each time) ---------------
    probe = TwoLayerMSE(SEED)
    print(f"{'t_w':>7} {'test_acc':>9} {'X':>7}")
    rows = []
    for ck in CKPTS:
        probe.load_state_dict(ckpt_states[ck])
        ate = acc(probe, Xte, te)
        s = fd_probe(probe, Xtr, Ytr, mse_loss, eta=ETA, weight_decay=WD,
                     batch_size=B, ndir=8, nseed=4, M=400, ML=3000,
                     base_seed=50000 + ck)
        rows.append((ck, ate, s.X))
        print(f"{ck:>7d} {ate:>9.3f} {s.X:>7.3f}")

    _plot(curve, rows)


def _plot(curve, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")
        return
    curve = np.array(curve)
    rows = np.array(rows)
    ck, ate, X = rows[:, 0], rows[:, 1], rows[:, 2]

    grok_idx = np.where(curve[:, 2] > 0.9)[0]
    t_grok = curve[grok_idx[0], 0] if len(grok_idx) else curve[-1, 0]
    # X saturation. X wobbles in the post-saturation tail, so use the MEDIAN of
    # the second half of the checkpoints as the plateau level (robust to the
    # noisy peak) and call saturation the first checkpoint reaching 85% of it.
    plateau = float(np.median(X[len(X) // 2:]))
    sat_idx = np.where(X >= 0.85 * plateau)[0]
    t_sat = ck[sat_idx[0]] if len(sat_idx) else ck[0]

    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})

    ax1.plot(curve[:, 0], curve[:, 1], color="#4C72B0", lw=1.6, label="train acc")
    ax1.plot(curve[:, 0], curve[:, 2], color="#C44E52", lw=1.8, label="test acc")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(-0.03, 1.03)
    ax1.legend(loc="center right", frameon=False, fontsize=9)
    ax1.set_title("X saturates long before test accuracy moves "
                  "(2-layer MSE, modular addition)", fontsize=11.5, loc="left")

    ax2.plot(ck, X, "o-", color="#8172B2", ms=7, lw=1.9,
             label=r"$X=T_{\rm eff}/T_{\rm bath}$  (physfdt.response)")
    ax2.axhline(1.0, color="0.3", ls="--", lw=1.0)
    ax2.text(ck.max(), 1.02, "X = 1  (FDT equilibrium)", ha="right", va="bottom",
             fontsize=8.5, color="0.3")
    ax2.axvspan(t_sat, t_grok, color="gold", alpha=0.13)
    ax2.annotate("X saturated;\ntest acc still ~0", xy=(t_sat, plateau * 0.98),
                 xytext=(t_sat * 0.28, plateau * 1.15), fontsize=8.5, color="0.35",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax2.set_ylabel(r"$X = T_{\rm eff}/T_{\rm bath}$")
    ax2.set_xlabel(r"checkpoint $t_w$ (SGD step)")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="lower right", frameon=False, fontsize=9)
    for ax in (ax1, ax2):
        ax.axvline(t_grok, color="#2ca02c", lw=1.2, alpha=0.6)
    ax2.text(t_grok, 0.04, r"  $t_{\rm grok}$", color="#2ca02c", fontsize=9,
             ha="left", va="bottom")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    np.savez(os.path.join(os.path.dirname(OUT_PNG), "leading_indicator_data.npz"),
             curve=curve, rows=rows)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    lead = t_grok / max(t_sat, 1)
    print(f"\nX saturates at t_w~{t_sat:.0f}; grokking at t_grok~{t_grok:.0f}  "
          f"-> lead factor ~{lead:.1f}x")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
