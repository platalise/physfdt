"""physfdt -- fluctuation-dissipation diagnostics for stochastic gradient descent.

Measures whether SGD training has equilibrated at the current learning rate,
using the exact stationarity relation

    2 <w . u> = eta <|u|^2>

with no assumption about the structure of the gradient noise and no held-out
validation set.

Quick start
-----------
::

    from physfdt import FDRMonitor, FDRConfig, FDREquilibriumLR

    monitor = FDRMonitor(optimizer, FDRConfig(half_life=200))
    sched = FDREquilibriumLR(optimizer, monitor, factor=0.5)

    loss.backward()
    with monitor.measure():
        optimizer.step()
    sched.step()

See the README for scope and limitations before using the numbers in a paper.
"""

from .core import FDRAccumulator, FDRConfig, FDRState
from .numpy_backend import NumpyFDRMonitor, fdr_terms
from .spectral import (
    SpectralReport,
    esd,
    ipr,
    powerlaw_alpha,
    spectral_report,
    spectral_report_torch,
)
from .trace import TraceWriter

__version__ = "0.2.0"

__all__ = [
    "FDRAccumulator",
    "FDRConfig",
    "FDRState",
    "NumpyFDRMonitor",
    "fdr_terms",
    "SpectralReport",
    "esd",
    "ipr",
    "powerlaw_alpha",
    "spectral_report",
    "spectral_report_torch",
    "TraceWriter",
    "FDRMonitor",
    "FDREquilibriumLR",
    "__version__",
]


def __getattr__(name):
    # Torch-dependent symbols are imported lazily so that `import physfdt`
    # works in a NumPy-only environment.
    if name in ("FDRMonitor", "FDREquilibriumLR"):
        from . import torch_backend

        return getattr(torch_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
