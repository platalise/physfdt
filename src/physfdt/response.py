"""physfdt.response -- the twin-trajectory fluctuation-dissipation probe.

Where :class:`~physfdt.FDRMonitor` reads the *same-time* stationarity ratio
``rho`` cheaply at every step, this module measures the full
fluctuation-dissipation ratio

    X  =  T_eff / T_bath

at a single frozen checkpoint, by comparing the network's *response* to a small
constant force against its *fluctuations* in the same direction. It is the
discrete-time, mini-batch analogue of the fluctuation-dissipation theorem of
statistical mechanics:

* ``T_bath`` -- the "bath" temperature, the variance of the mini-batch gradient
  projected onto a probe direction (the noise driving the weights).
* ``T_eff``  -- the "effective" temperature read off from the trajectory itself,
  as the slope of the mean-squared displacement ``Delta(tau)`` against twice the
  response ``chi(tau)`` (equilibrium FDT: ``Delta = 2 T chi``).
* ``X = T_eff / T_bath`` -- unity when the FDT holds (equilibrium), and departs
  from unity when it does not.

Unlike ``rho`` this is an *expensive, occasional* measurement: one call runs
tens of thousands of extra SGD-equivalent steps and snapshots/restores the model
(it leaves ongoing training undisturbed). You call it at a handful of
checkpoints, not every step.

Why it is interesting
---------------------
On losses that are locally quadratic in the probed weights (an MSE / linearised
readout), ``X`` rises smoothly and monotonically toward its plateau as the
Hessian spectrum flattens across a learning transition, and it *saturates well
before* held-out accuracy moves. That makes it a **label-free leading indicator**
of delayed generalisation ("grokking"): you can read, from the training ramp
alone, whether a run is on track to generalise -- see
``examples/05_leading_indicator.py`` and the README.

Validation
----------
This estimator is checked two ways:

* against a closed-form multi-mode Ornstein-Uhlenbeck system with a known ``X``
  (``tests/test_response.py``), to within Monte-Carlo error;
* against the reference implementation of Nguyen (2026) on the exact 2-layer MSE
  modular-addition task, reproducing the published ``X(t_w)`` curve to within
  0.01 at every checkpoint.

Scope
-----
``X`` is well-defined and clean in the regime where the probed loss is close to
quadratic. On a strongly non-quadratic, saturating loss (e.g. softmax
cross-entropy once the network is confident, where the gradient noise collapses)
``T_bath`` becomes tiny and ``X`` is numerically fragile and non-monotonic. Read
the README "Scope" section before quoting a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # torch is an optional dependency of physfdt
    import torch
except ImportError:  # pragma: no cover
    torch = None


__all__ = ["ResponseState", "fd_probe"]


@dataclass
class ResponseState:
    """Result of a single :func:`fd_probe` call.

    Attributes
    ----------
    X:
        The fluctuation-dissipation ratio ``T_eff / T_bath``. Unity at
        equilibrium; the order parameter of interest.
    T_eff, T_bath:
        Effective and bath temperatures (see module docstring).
    chi, Delta:
        Response function ``chi(tau)`` and mean-squared displacement
        ``Delta(tau)`` averaged over probe directions, length ``M``. Kept so you
        can inspect the FDT fit yourself.
    T_bath_dirs:
        Per-direction bath temperature, length ``ndir``.
    """

    X: float
    T_eff: float
    T_bath: float
    chi: np.ndarray = field(repr=False)
    Delta: np.ndarray = field(repr=False)
    T_bath_dirs: np.ndarray = field(repr=False)


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def _set_flat_params(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p))
        i += n


def _make_batches(seed, n_train, batch_size, nsteps):
    r = np.random.default_rng(seed)
    return r.integers(0, n_train, size=(nsteps, batch_size))


def _rollout(model, opt_factory, X, y, loss_fn, batches, w0, force=None,
             record_dirs=None):
    """Restore to ``w0``, run ``len(batches)`` SGD steps, optionally with a
    constant force subtracted from the gradient (a loss tilt), optionally
    recording projections onto ``record_dirs`` relative to ``w0`` each step."""
    _set_flat_params(model, w0)
    opt = opt_factory(model.parameters())
    traj = None
    if record_dirs is not None:
        traj = torch.empty(len(batches), record_dirs.shape[0],
                           dtype=w0.dtype, device=w0.device)
    for t, idx in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, X[idx], y[idx])
        loss.backward()
        if force is not None:
            i = 0
            with torch.no_grad():
                for p in model.parameters():
                    n = p.numel()
                    p.grad -= force[i:i + n].view_as(p)
                    i += n
        opt.step()
        if record_dirs is not None:
            traj[t] = record_dirs @ (_flat_params(model) - w0)
    return traj


def fd_probe(model, inputs, targets, loss_fn, *, eta, weight_decay=0.0,
             batch_size, ndir=8, nseed=4, M=400, ML=3000, base_seed=0,
             grad_noise_batches=60, tau_fit_lo_frac=0.3):
    """Measure ``X = T_eff / T_bath`` at the model's current parameters.

    The model is restored to its original parameters before returning, so an
    in-progress training loop is left undisturbed.

    Parameters
    ----------
    model:
        A ``torch.nn.Module``. Trained with plain SGD + weight decay (the FDT
        reading assumes ``u = grad``; momentum/Adam are out of scope).
    inputs, targets:
        The full training tensors. Mini-batches are drawn by random index into
        the first dimension, so the probe reproduces the same noise the model
        actually trained under. Keep them on the same device as the model.
    loss_fn:
        ``loss_fn(model, x_batch, y_batch) -> scalar loss``.
    eta, weight_decay, batch_size:
        Must match the training run: the probe re-runs the *same* SGD update.
    ndir, nseed, M, ML:
        Probe budget. ``ndir`` random unit directions, ``nseed`` shared-noise
        twin pairs each, response window ``M`` steps, one unperturbed run of
        length ``ML`` for the fluctuation side. Defaults reproduce the reference
        paper. Cost is roughly ``ndir*nseed*2*M + ML`` extra SGD steps.
    base_seed:
        Seeds every RNG in the probe; vary it for an independent repeat.
    grad_noise_batches:
        Number of mini-batches used to estimate ``T_bath``.
    tau_fit_lo_frac:
        The FDT slope is fit over ``tau`` in ``[tau_fit_lo_frac*M, M]`` (the
        tail of the window, where the linear-response regime holds).

    Returns
    -------
    ResponseState
    """
    if torch is None:  # pragma: no cover
        raise ImportError("physfdt.response requires torch")

    n_train = inputs.shape[0]
    nparam = sum(p.numel() for p in model.parameters())
    w0 = _flat_params(model)

    def opt_factory(params):
        return torch.optim.SGD(params, lr=eta, weight_decay=weight_decay,
                               momentum=0.0)

    rng = np.random.default_rng(base_seed)
    dirs = torch.tensor(rng.normal(size=(ndir, nparam)),
                        dtype=w0.dtype, device=w0.device)
    dirs /= dirs.norm(dim=1, keepdim=True)

    # --- gradient noise along each direction -> T_bath ---
    gbatches = _make_batches(base_seed + 1, n_train, batch_size, grad_noise_batches)
    gs = torch.empty(grad_noise_batches, nparam, dtype=w0.dtype, device=w0.device)
    _set_flat_params(model, w0)
    opt = opt_factory(model.parameters())
    for t, idx in enumerate(gbatches):
        opt.zero_grad(set_to_none=True)
        loss_fn(model, inputs[idx], targets[idx]).backward()
        parts = []
        for p in model.parameters():
            parts.append(p.grad.detach().reshape(-1) if p.grad is not None
                         else torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
        gs[t] = torch.cat(parts)
    _set_flat_params(model, w0)
    proj = gs @ dirs.T
    s_v = proj.std(dim=0)
    T_bath_dirs = 0.5 * eta * proj.var(dim=0)

    # --- twin runs -> response chi(tau) ---
    chi = torch.zeros(ndir, M, dtype=w0.dtype, device=w0.device)
    for i in range(ndir):
        v = dirs[i]
        f = 2.0 * s_v[i].item()
        for s in range(nseed):
            seed = base_seed + 1000 * i + 7 * s + 13
            batches = _make_batches(seed, n_train, batch_size, M)
            tu = _rollout(model, opt_factory, inputs, targets, loss_fn, batches, w0,
                          force=None, record_dirs=dirs[i:i + 1])
            tp = _rollout(model, opt_factory, inputs, targets, loss_fn, batches, w0,
                          force=f * v, record_dirs=dirs[i:i + 1])
            chi[i] += ((tp - tu) / f).squeeze(-1)
        chi[i] /= nseed
    chi_m = chi.mean(dim=0)

    # --- long unperturbed run -> quadratically-detrended MSD Delta(tau) ---
    lbatches = _make_batches(base_seed + 99991, n_train, batch_size, ML)
    Q = _rollout(model, opt_factory, inputs, targets, loss_fn, lbatches, w0,
                 force=None, record_dirs=dirs)
    _set_flat_params(model, w0)  # restore: probing is done

    Q = Q.cpu().numpy()
    tt = np.arange(ML)
    taus = np.arange(1, M + 1)
    Delta = np.zeros((ndir, M))
    for i in range(ndir):
        q = Q[:, i]
        qd = q - np.polyval(np.polyfit(tt, q, 2), tt)
        for j, tau in enumerate(taus):
            d = qd[tau:] - qd[:-tau]
            Delta[i, j] = np.mean(d ** 2)
    D_m = Delta.mean(axis=0)

    chi_np = chi_m.cpu().numpy()
    k0 = int(tau_fit_lo_frac * M)
    A = 2 * chi_np[k0:]
    T_eff = float(np.sum(A * D_m[k0:]) / np.sum(A * A))
    T_bath = float(T_bath_dirs.mean().item())
    return ResponseState(X=T_eff / T_bath, T_eff=T_eff, T_bath=T_bath,
                         chi=chi_np, Delta=D_m,
                         T_bath_dirs=T_bath_dirs.cpu().numpy())
