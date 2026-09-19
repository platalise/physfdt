"""CSV logging of the FDR trace.

One row per measured step. The point of writing a plain CSV rather than
plugging into an experiment tracker is reproducibility: the trace that backs a
figure should be a file you can attach to a paper.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .core import FDRState

__all__ = ["TraceWriter"]

_FIELDS = [
    "step",
    "steps_since_reset",
    "lr",
    "lhs",
    "rhs",
    "rho",
    "residual",
    "in_band",
    "band_run",
    "equilibrated",
]


class TraceWriter:
    """Append FDR states to a CSV file.

    Extra columns are declared once at construction and supplied per row.

    Examples
    --------
    ::

        trace = TraceWriter("run.csv", extra=["loss", "alpha"])
        ...
        state = monitor.post_step()
        if state is not None:
            trace.write(state, lr=current_lr, loss=loss.item(), alpha=a)
        ...
        trace.close()
    """

    def __init__(
        self,
        path: Union[str, Path],
        extra: Optional[list[str]] = None,
        overwrite: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.extra = list(extra or [])
        self.fields = _FIELDS + self.extra
        mode = "w" if overwrite or not self.path.exists() else "a"
        self._fh = self.path.open(mode, newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields)
        if mode == "w":
            self._writer.writeheader()

    def write(self, state: FDRState, lr: float = float("nan"), **extra: Any) -> None:
        row: dict[str, Any] = asdict(state)
        row["lr"] = lr
        for k in self.extra:
            row[k] = extra.get(k, "")
        unknown = set(extra) - set(self.extra)
        if unknown:
            raise KeyError(f"undeclared extra columns: {sorted(unknown)}")
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._fh.flush()

    def write_mapping(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
