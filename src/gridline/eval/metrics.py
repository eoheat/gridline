"""Probability metrics. calibration_error() is nflfastR's definition, verbatim in
logic (Baldwin 2020, Open Source Football, 'nflfastR EP, WP, and CP models'):

    bin      = round(wp / 0.05) * 0.05                     (R rounds half to even; so does numpy)
    per (qtr, bin): n_plays, n_wins, actual = n_wins / n_plays, cal_diff = |bin - actual|
    per qtr:  weighted mean of cal_diff, weights n_plays
    overall:  weighted mean of the per-quarter errors, weights = wins in that quarter

Every metric takes optional row weights. eval/bootstrap.py uses them to resample whole
games without copying rows: a game drawn twice gets weight 2.
"""

from __future__ import annotations

import numpy as np
import polars as pl

N_BINS = 21  # 0.00, 0.05, ..., 1.00


def calibration_table(pred, label, qtr, weight=None) -> pl.DataFrame:
    """One row per (qtr, bin) that has plays: n_plays, n_wins, the actual win rate and
    |bin - actual|. With weights, n_plays and n_wins are weighted counts."""
    p = np.asarray(pred, dtype=float)
    won = np.asarray(label) == 1
    q = np.asarray(qtr).astype(np.int64)
    w = np.ones(p.size) if weight is None else np.asarray(weight, dtype=float)
    cell = q * N_BINS + np.round(p / 0.05).astype(np.int64)
    size = (int(q.max()) + 1) * N_BINS
    n = np.bincount(cell, weights=w, minlength=size)
    wins = np.bincount(cell, weights=w * won, minlength=size)
    used = np.flatnonzero(n > 0)
    return (
        pl.DataFrame({"qtr": used // N_BINS, "bin": (used % N_BINS) * 0.05,
                      "n_plays": n[used], "n_wins": wins[used]})
        .with_columns(actual=pl.col("n_wins") / pl.col("n_plays"))
        .with_columns(cal_diff=(pl.col("bin") - pl.col("actual")).abs())
    )


def calibration_error(pred, label, qtr, weight=None) -> float:
    t = calibration_table(pred, label, qtr, weight)
    q = t["qtr"].to_numpy()
    n, wins, diff = (t[c].to_numpy() for c in ("n_plays", "n_wins", "cal_diff"))
    n_q = np.bincount(q, weights=n)
    err_q = np.divide(np.bincount(q, weights=diff * n), n_q, out=np.zeros_like(n_q), where=n_q > 0)
    wins_q = np.bincount(q, weights=wins)
    return float((err_q * wins_q).sum() / wins_q.sum())


def log_loss(pred, label, weight=None, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(pred, dtype=float), eps, 1 - eps)
    y = np.asarray(label, dtype=float)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=weight))


def brier(pred, label, weight=None) -> float:
    err = np.asarray(pred, dtype=float) - np.asarray(label, dtype=float)
    return float(np.average(err ** 2, weights=weight))


def error_rate(pred, label, weight=None) -> float:
    """Share of plays where the side the model favours (p > 0.5) lost."""
    wrong = (np.asarray(pred) > 0.5) != (np.asarray(label) == 1)
    return float(np.average(wrong, weights=weight))


def summary(pred, label, qtr, weight=None) -> dict:
    return {"calibration_error": calibration_error(pred, label, qtr, weight),
            "log_loss": log_loss(pred, label, weight), "brier": brier(pred, label, weight),
            "error_rate": error_rate(pred, label, weight), "n": int(len(pred))}
