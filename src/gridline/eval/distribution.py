"""Scoring a predicted distribution of an integer outcome (a final margin or total).

Every forecast is represented by its CDF on a fixed integer grid, whether it comes from
simulated samples or from a formula, so every forecast is scored the same way:

  CRPS  sum over grid points z of (F(z) - 1{y <= z})^2: the continuous ranked
        probability score, exact for integer-valued forecasts. Lower is better; for a
        point forecast it is the absolute error.
  PIT   F(y - 1) + v * (F(y) - F(y - 1)) with v ~ U(0, 1): the randomized probability
        integral transform. Uniform on (0, 1) exactly when the forecast is calibrated,
        so the share of PITs in (0.25, 0.75) is the coverage of the central 50%
        interval, and so on.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

MARGIN_GRID = np.arange(-100, 101)
TOTAL_GRID = np.arange(0, 161)


def cdf_from_samples(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """F(z) = share of samples <= z, for every z in grid."""
    s = np.sort(np.asarray(samples))
    return np.searchsorted(s, grid, side="right") / s.size


def cdf_normal(mu: float, sigma: float, grid: np.ndarray) -> np.ndarray:
    """A normal forecast rounded to the nearest integer."""
    return norm.cdf((grid + 0.5 - mu) / max(sigma, 1e-6))


def crps(cdf: np.ndarray, grid: np.ndarray, y: float) -> float:
    return float(np.sum((cdf - (grid >= y)) ** 2))


def pit(cdf: np.ndarray, grid: np.ndarray, y: float, v: float) -> float:
    i = int(np.clip(y - grid[0], 0, grid.size - 1))
    below = cdf[i - 1] if i > 0 else 0.0
    return float(below + v * (cdf[i] - below))


def coverage(pits: np.ndarray, level: float) -> float:
    """Share of outcomes inside the central `level` interval."""
    lo = (1 - level) / 2
    p = np.asarray(pits)
    return float(np.mean((p > lo) & (p < 1 - lo)))
