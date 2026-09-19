#!/usr/bin/env python3
"""Monitor one real training run with physfdt.

    # works offline, no download
    python examples/02_torch_quickstart.py

    # real data (downloads MNIST once)
    python examples/02_torch_quickstart.py --dataset mnist

    # let the detector drive the learning rate
    python examples/02_torch_quickstart.py --schedule fdr

Writes a CSV trace with one row per measured step: rho, the two sides of
FDR-1, the learning rate, the loss, and the spectral exponent alpha of the
first hidden layer. Those last two columns are the interesting pair -- rho is a
dynamical signal, alpha is a structural one, and they are computed from
different things.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from physfdt import (
    FDRConfig,
    FDREquilibriumLR,
    FDRMonitor,
    TraceWriter,
    spectral_report_torch,
)


def build_data(name: str, batch_size: int, device):
    if name == "synthetic":
        # A learnable but non-trivial task: random teacher network, 10 classes.
        g = torch.Generator().manual_seed(0)
        n, d = 20_000, 256
        X = torch.randn(n, d, generator=g)
        teacher = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 10))
        for p in teacher.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.5
        with torch.no_grad():
            y = teacher(X).argmax(1)
        ds = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True), d

    if name == "mnist":
        from torchvision import datasets, transforms

        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
             transforms.Lambda(lambda t: t.view(-1))]
        )
        ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True), 784

    raise ValueError(f"unknown dataset {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="0 keeps the exact fluctuation-dissipation reading")
    ap.add_argument("--weight-decay", type=float, default=1e-2,
                    help="REQUIRED for a stationary state. Cross-entropy on "
                         "separable data has none at wd=0: |w| grows without "
                         "bound and rho goes negative. Pass 0 to see that happen.")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--schedule", default="constant", choices=["constant", "fdr", "cosine"])
    ap.add_argument("--every", type=int, default=1, help="measure every N steps")
    ap.add_argument("--alpha-every", type=int, default=100, help="SVD is not cheap")
    ap.add_argument("--half-life", type=int, default=500)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--min-steps", type=int, default=500)
    ap.add_argument("--out", default="fdr_trace.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, in_dim = build_data(args.dataset, args.batch_size, device)

    model = nn.Sequential(
        nn.Linear(in_dim, args.width), nn.ReLU(),
        nn.Linear(args.width, args.width // 2), nn.ReLU(),
        nn.Linear(args.width // 2, 10),
    ).to(device)
    probe_layer = model[0]                       # layer we track alpha on

    opt = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    cfg = FDRConfig(
        half_life=args.half_life, tol=args.tol,
        patience=args.patience, min_steps=args.min_steps,
    )
    monitor = FDRMonitor(opt, cfg, mode="delta", every=args.every)
    fdr_sched = FDREquilibriumLR(opt, monitor, factor=0.5, min_lr=1e-5)
    cos_sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        if args.schedule == "cosine" else None
    )

    trace = TraceWriter(args.out,
                        extra=["loss", "alpha", "r2", "ipr_max", "regime", "rho_std"])
    print(f"device={device}  dataset={args.dataset}  schedule={args.schedule}  "
          f"weight_decay={args.weight_decay:g}")
    if args.weight_decay == 0.0:
        print("\n  !! weight_decay=0 with cross-entropy: once the model separates the\n"
              "     data, max-margin dynamics drives |w| -> inf, there is no stationary\n"
              "     state, and rho goes negative and stays there. That is the tool\n"
              "     working, not failing. Re-run with --weight-decay 1e-2.\n")
    print(f"{'step':>7} {'loss':>9} {'lr':>10} {'rho':>10} {'alpha':>8}  regime / status")

    step, t0, alpha, r2, ipr_max = 0, time.time(), "", "", ""
    done = False
    while not done:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()

            with monitor.measure():
                opt.step()

            state = monitor.last
            step += 1

            if step % args.alpha_every == 0:
                rep = spectral_report_torch(probe_layer.weight, method="mle")
                alpha, r2, ipr_max = rep.alpha, rep.r2, rep.ipr_max

            fired = fdr_sched.step() if args.schedule == "fdr" else False
            if cos_sched is not None:
                cos_sched.step()

            lr_now = opt.param_groups[0]["lr"]
            if state is not None:
                trace.write(state, lr=lr_now, loss=loss.item(),
                            alpha=alpha, r2=r2, ipr_max=ipr_max,
                            regime=state.regime, rho_std=state.rho_std)

            if step % 200 == 0 or fired:
                rho = f"{state.rho:10.4f}" if state else " " * 10
                a = f"{alpha:8.3f}" if alpha != "" else " " * 8
                status = state.regime if state else ""
                if state and state.regime == "norm_growing_fast":
                    status += "   <- no stationary state; add weight decay"
                if fired:
                    status = "LR DECAY"
                print(f"{step:>7} {loss.item():>9.4f} {lr_now:>10.5f} {rho} {a}  {status}")

            if step >= args.steps:
                done = True
                break

    trace.close()
    dt = time.time() - t0
    print(f"\n{step} steps in {dt:.1f}s ({1000 * dt / step:.2f} ms/step)")

    last = monitor.last
    if last is not None:
        print(f"\nfinal rho = {last.rho:.4f}   regime = {last.regime}")
        if last.rho_std == last.rho_std:
            print(f"noise floor of rho (3 sigma) = {3 * last.rho_std:.4f}   "
                  f"tol = {args.tol}")
            if not last.tol_is_achievable:
                print("  !! tol is BELOW the estimator's own noise floor, so "
                      "'equilibrated' would be\n     decided by sampling noise. "
                      f"Raise --half-life (scatter falls as 1/sqrt) or --tol "
                      f"above {3 * last.rho_std:.3f}.")
        if last.regime == "norm_growing_fast":
            print("  !! rho < 0: no stationary state. See --weight-decay.")
    print(f"trace -> {args.out}")
    if fdr_sched.history:
        print("FDR-triggered decays (step, lr_before, rho):")
        for h in fdr_sched.history:
            print(f"  step {h[0]:>6}  lr {h[1]:.5f}  rho {h[2]:.4f}")


if __name__ == "__main__":
    main()
