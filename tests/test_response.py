"""Validate the twin-trajectory response probe against a closed-form
multi-mode Ornstein-Uhlenbeck system with a known X.

Construction: a diagonal quadratic loss over ``d`` independent modes,
``L_i(w) = 0.5 * sum_a lam_a (w_a - b_{i,a})^2`` with the per-column offsets
``b`` exactly demeaned, so ``w = 0`` is an exact stationary point. The
per-example gradient variance at ``w=0`` is ``lam_a^2 * sigma0_a^2``, giving a
known per-mode temperature ``T_a = eta * lam_a^2 sigma0_a^2 / (2 B)`` -- and the
same per-mode OU formula the reference paper uses for ``X*_th`` then predicts the
value the probe should measure. A random-direction probe averages over modes
exactly as that theory does, so measured X should match the closed form to
within Monte-Carlo error.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physfdt.response import fd_probe  # noqa: E402


class _DiagQuadratic(torch.nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(len(lam), dtype=torch.float64))
        self.register_buffer("lam", torch.as_tensor(lam, dtype=torch.float64))

    def forward(self, x):  # pragma: no cover - loss_fn does the work
        return x


def _loss(model, xb, yb):
    diff = model.w.unsqueeze(0) - xb
    return 0.5 * (model.lam.unsqueeze(0) * diff ** 2).sum(dim=1).mean()


def _X_theory(lam, T_a, eta, M, lo=0.3):
    taus = np.arange(1, M + 1)
    decay = 1 - (1 - eta * lam)[None, :] ** taus[:, None]
    chi = (decay / lam[None, :]).mean(1)
    Dlt = (2 * (T_a / lam)[None, :] * decay).mean(1)
    k0 = int(lo * M)
    A = 2 * chi[k0:]
    T_eff = float(np.sum(A * Dlt[k0:]) / np.sum(A * A))
    return T_eff / float(T_a.mean())


def _make_system(n_modes, lam_lo, lam_hi, T_fn, eta, B, seed, n_examples=20000):
    rng = np.random.default_rng(seed)
    lam = np.exp(np.linspace(np.log(lam_lo), np.log(lam_hi), n_modes))
    T_a = T_fn(lam)
    sigma0 = np.sqrt(2 * B * T_a / eta) / lam
    b = rng.normal(size=(n_examples, n_modes)) * sigma0[None, :]
    b -= b.mean(axis=0, keepdims=True)
    return lam, T_a, b


# Fixed seeds: these system/probe seed pairs give <10% Monte-Carlo error at the
# ndir=8, nseed=4 budget (verified). Do not derive seeds from ``hash`` -- Python
# randomises string hashes per process, which would make the test flaky.
@pytest.mark.parametrize(
    "name, T_fn, sys_seed, probe_seed",
    [
        ("equilibrium", lambda lam: np.full_like(lam, 0.5), 1, 2),   # X == 1
        ("cold-stiff", lambda lam: 0.5 / np.sqrt(lam), 2, 3),        # X > 1
        ("hot-stiff", lambda lam: 0.3 * np.sqrt(lam), 3, 4),         # X < 1
    ],
)
def test_probe_matches_closed_form(name, T_fn, sys_seed, probe_seed):
    eta, B, M = 0.05, 64, 400
    lam, T_a, b = _make_system(40, 0.05, 3.0, T_fn, eta, B, seed=sys_seed)
    Xtr = torch.as_tensor(b, dtype=torch.float64)
    ytr = torch.zeros(b.shape[0], dtype=torch.float64)

    model = _DiagQuadratic(lam)
    out = fd_probe(model, Xtr, ytr, _loss, eta=eta, weight_decay=0.0,
                   batch_size=B, ndir=8, nseed=4, M=M, ML=3000, base_seed=probe_seed)

    # model must be restored to w = 0 after probing
    assert np.max(np.abs(model.w.detach().numpy())) < 1e-10

    X_th = _X_theory(lam, T_a, eta, M)
    assert abs(out.X - X_th) / abs(X_th) < 0.15, (name, out.X, X_th)
