"""Uncertainty from resampling whole games (DESIGN.md §7).

Plays in the same game share one outcome, so they are far from independent: the
effective sample size is closer to the number of games than to the number of plays.
Every interval in gridline therefore resamples games, never plays. A resample is a
vector of row weights (a game drawn twice counts twice), so no rows are copied.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from gridline.eval import metrics

N_BOOT = 1000
SEED = 2013
PAIRED_METRICS = ("calibration_error", "log_loss", "brier")


def game_weights(game_ids, n_boot: int = N_BOOT, seed: int = SEED) -> Iterator[np.ndarray]:
    """Row weights for `n_boot` resamples. Each resample draws as many games as there
    are, with replacement, and every row of a game gets that game's draw count."""
    _, codes = np.unique(np.asarray(game_ids), return_inverse=True)
    n_games = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n_games, n_games), minlength=n_games)[codes].astype(float)


def spread(stat: Callable[[np.ndarray], float], game_ids, n_boot: int = N_BOOT,
           seed: int = SEED) -> dict:
    """How much stat(weights) moves across resamples: SD, mean and 90% interval."""
    draws = np.array([stat(w) for w in game_weights(game_ids, n_boot, seed)])
    return {"sd": float(draws.std(ddof=1)), "mean": float(draws.mean()),
            "p05": float(np.quantile(draws, 0.05)), "p95": float(np.quantile(draws, 0.95))}


def paired(pred_a, pred_b, label, qtr, game_ids, n_boot: int = N_BOOT,
           seed: int = SEED) -> dict[str, dict]:
    """metric(a) - metric(b) on the same plays, for each of PAIRED_METRICS: the difference
    on the full sample and a 90% interval from resampling games (the same games for both
    models in every resample)."""
    full_a, full_b = metrics.summary(pred_a, label, qtr), metrics.summary(pred_b, label, qtr)
    draws: dict[str, list[float]] = {k: [] for k in PAIRED_METRICS}
    for w in game_weights(game_ids, n_boot, seed):
        a, b = metrics.summary(pred_a, label, qtr, w), metrics.summary(pred_b, label, qtr, w)
        for k in PAIRED_METRICS:
            draws[k].append(a[k] - b[k])
    return {k: {"diff": full_a[k] - full_b[k], "p05": float(np.quantile(draws[k], 0.05)),
                "p95": float(np.quantile(draws[k], 0.95))} for k in PAIRED_METRICS}
