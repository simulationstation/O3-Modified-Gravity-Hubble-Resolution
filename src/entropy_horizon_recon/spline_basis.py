from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearSpline1D:
    """Simple 1D piecewise-linear spline with fixed knots.

    This is intentionally tiny and dependency-free. It is used for nuisance models where we want:
    - explicit knot parameters
    - cheap evaluation
    - a basis representation (hat functions) for fast linearized moment calculations.
    """

    x_knots: np.ndarray  # (k,) strictly increasing

    def __post_init__(self) -> None:
        x = np.asarray(self.x_knots, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("x_knots must be 1D with >=2 knots.")
        if np.any(~np.isfinite(x)):
            raise ValueError("x_knots must be finite.")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("x_knots must be strictly increasing.")
        object.__setattr__(self, "x_knots", x)

    @property
    def n_knots(self) -> int:
        return int(self.x_knots.size)

    def basis(self, x: np.ndarray) -> np.ndarray:
        """Return hat-function basis matrix B s.t. spline(x) = B @ y_knots.

        For x outside the knot range, values are clamped to the nearest endpoint knot.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("basis expects 1D x.")
        k = int(self.x_knots.size)
        B = np.zeros((int(x.size), k), dtype=float)
        # Clamp to endpoints.
        xc = np.clip(x, float(self.x_knots[0]), float(self.x_knots[-1]))
        # Bin indices: find i s.t. x in [x_i, x_{i+1}].
        idx = np.searchsorted(self.x_knots, xc, side="right") - 1
        idx = np.clip(idx, 0, k - 2).astype(np.int64, copy=False)
        x0 = self.x_knots[idx]
        x1 = self.x_knots[idx + 1]
        denom = np.clip(x1 - x0, 1e-30, np.inf)
        t = (xc - x0) / denom
        # Linear weights to adjacent knots.
        B[np.arange(x.size), idx] = 1.0 - t
        B[np.arange(x.size), idx + 1] = t
        return B

    def eval(self, x: np.ndarray, y_knots: np.ndarray) -> np.ndarray:
        y = np.asarray(y_knots, dtype=float)
        if y.ndim != 1 or y.size != int(self.x_knots.size):
            raise ValueError("y_knots must be 1D and match x_knots length.")
        B = self.basis(x)
        return B @ y

