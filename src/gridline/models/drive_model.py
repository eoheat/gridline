"""Phase 2 drive-step model: from any scrimmage snap, the joint distribution of how
the drive ends and how much clock it uses.

One softmax covers every (result, duration bin) pair; end_half is a single class,
because such a drive uses exactly the time left in the half. A single forward pass per
drive step then serves every simulated game at once. The network is a small MLP
evaluated in numpy: the simulator calls it tens of times per state for thousands of
simulated games, which a boosted-tree model is far too slow for (about 50 ms per call
for 4,000 rows against about 5 ms here).

Inputs are from the offense's point of view. The pregame spread and total are inputs,
so the per-game knobs of DESIGN.md decision 3 are simply the values fed in for them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import polars as pl

from gridline import config
from gridline.models.drives import RESULTS

Log = Callable[[str], None]

DURATION_EDGES = np.array([0, 6, 10, 15, 22, 32, 44, 56, 70, 85, 100, 118, 138, 160, 186, 216,
                           250, 300, 380, 900], dtype=float)
N_BINS = len(DURATION_EDGES) - 1
TIMED = tuple(r for r in RESULTS if r != "end_half")
# class k < len(TIMED) * N_BINS is (TIMED[k // N_BINS], bin k % N_BINS); the last is end_half
N_CLASSES = len(TIMED) * N_BINS + 1
END_HALF = N_CLASSES - 1
CLASS_RESULT = np.array([RESULTS.index(TIMED[k // N_BINS]) for k in range(N_CLASSES - 1)]
                        + [RESULTS.index("end_half")])
CLASS_BIN = np.array([k % N_BINS for k in range(N_CLASSES - 1)] + [-1])

DIFF_BUCKETS = (-8.5, -3.5, -0.5, 0.5, 3.5, 8.5)  # <=-9, -8..-4, -3..-1, 0, 1..3, 4..8, >=9
ERA_BASE, ERA_SCALE = 2016, 8.0
TOTAL_BASE = 44.0


def features(yardline, down, ydstogo, half_seconds, second_half, score_diff, timeouts_off,
             timeouts_def, spread_off, total, season, postseason) -> np.ndarray:
    """Model inputs (float32, one row per state). Arguments are equal-length arrays;
    spread_off/total may contain NaN (unknown line)."""
    y = np.asarray(yardline, dtype=np.float32)
    n = y.size
    d = np.clip(np.asarray(down, dtype=np.int64), 1, 4)
    togo = np.clip(np.asarray(ydstogo, dtype=np.float32), 1, 30)
    t = np.clip(np.asarray(half_seconds, dtype=np.float32), 0, 1800)
    diff = np.asarray(score_diff, dtype=np.float32)
    spread = np.nan_to_num(np.asarray(spread_off, dtype=np.float32), nan=0.0)
    tot = np.asarray(total, dtype=np.float32)
    tot = np.where(np.isnan(tot), TOTAL_BASE, tot)
    cols = [
        y / 100,
        *(d == k for k in (1, 2, 3, 4)),
        np.log1p(togo) / np.log(31),
        togo >= y,  # goal to go
        t / 1800,
        np.minimum(t, 300) / 300,
        t <= 120,
        np.asarray(second_half, dtype=np.float32),
        np.clip(diff, -35, 35) / 14,
        *np.eye(len(DIFF_BUCKETS) + 1, dtype=np.float32)[np.searchsorted(DIFF_BUCKETS, diff)].T,
        np.asarray(timeouts_off, dtype=np.float32) / 3,
        np.asarray(timeouts_def, dtype=np.float32) / 3,
        spread / 7,
        (tot - TOTAL_BASE) / 7,
        (np.asarray(season, dtype=np.float32) - ERA_BASE) / ERA_SCALE,
        np.asarray(postseason, dtype=np.float32),
    ]
    return np.column_stack([np.broadcast_to(np.asarray(c, dtype=np.float32), (n,)) for c in cols])


N_FEATURES = features(*[np.zeros(1)] * 12).shape[1]


def frame_features(snaps: pl.DataFrame, season_cap: int | None = None) -> np.ndarray:
    """Features for a snaps table (models/drives.py). Seasons after `season_cap` are fed
    in as that season: the model is never asked to extrapolate the era trend."""
    season = snaps["season"].to_numpy()
    if season_cap is not None:
        season = np.minimum(season, season_cap)
    return features(
        snaps["yardline_100"].to_numpy(), snaps["down"].to_numpy(), snaps["ydstogo"].to_numpy(),
        snaps["half_seconds_remaining"].to_numpy(), (snaps["game_half"] != "Half1").to_numpy(),
        snaps["score_diff"].to_numpy(), snaps["timeouts_off"].to_numpy(),
        snaps["timeouts_def"].to_numpy(), snaps["spread_off"].to_numpy(),
        snaps["pregame_total"].to_numpy(), season, (snaps["season_type"] == "POST").to_numpy(),
    )


def class_labels(snaps: pl.DataFrame) -> np.ndarray:
    result = snaps["result"].replace_strict({r: i for i, r in enumerate(RESULTS)},
                                            return_dtype=pl.Int64).to_numpy()
    b = np.clip(np.searchsorted(DURATION_EDGES, snaps["seconds"].to_numpy(), side="right") - 1,
                0, N_BINS - 1)
    timed = np.array([TIMED.index(RESULTS[r]) if RESULTS[r] != "end_half" else -1 for r in range(len(RESULTS))])
    k = timed[result] * N_BINS + b
    return np.where(result == RESULTS.index("end_half"), END_HALF, k)


def training_snaps(snaps: pl.DataFrame) -> pl.DataFrame:
    """Regulation snaps with a clean label and a clock."""
    return snaps.filter(
        pl.col("result").is_not_null() & (pl.col("game_half") != "Overtime")
        & pl.col("seconds").is_not_null() & pl.col("half_seconds_remaining").is_not_null()
        & pl.col("score_diff").is_not_null() & pl.col("timeouts_off").is_not_null()
        & pl.col("timeouts_def").is_not_null()
    )


N_QUANTILES = 33  # duration draws within a bin follow the bin's own empirical quantiles


def bin_quantiles(snaps: pl.DataFrame) -> np.ndarray:
    """[N_BINS, N_QUANTILES] quantiles of the drive duration within each duration bin.
    Drawing uniformly inside a bin would be badly off in the wide last bin (380-900 s,
    whose real mean is about 445 s)."""
    t = snaps.filter(pl.col("result") != "end_half")["seconds"].to_numpy()
    b = np.clip(np.searchsorted(DURATION_EDGES, t, side="right") - 1, 0, N_BINS - 1)
    grid = np.linspace(0, 1, N_QUANTILES)
    out = np.empty((N_BINS, N_QUANTILES))
    for i in range(N_BINS):
        v = t[b == i]
        out[i] = np.quantile(v, grid) if v.size else np.linspace(DURATION_EDGES[i], DURATION_EDGES[i + 1], N_QUANTILES)
    return out


@dataclass
class StepModel:
    """A ReLU MLP with a softmax over N_CLASSES, evaluated in numpy."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    season_cap: int  # last training season; later seasons are fed in as this one
    quantiles: np.ndarray  # bin_quantiles() of the training data

    def duration(self, bin_: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Seconds for duration bins `bin_` from uniforms `u` (empirical, within the bin)."""
        pos = u * (N_QUANTILES - 1)
        lo = np.minimum(pos.astype(np.int64), N_QUANTILES - 2)
        q = self.quantiles[bin_]
        rows = np.arange(len(bin_))
        return q[rows, lo] + (pos - lo) * (q[rows, lo + 1] - q[rows, lo])

    def probs(self, x: np.ndarray) -> np.ndarray:
        h = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.maximum(h @ w + b, 0)
        z = h @ self.weights[-1] + self.biases[-1]
        z -= z.max(axis=1, keepdims=True)
        np.exp(z, out=z)
        z /= z.sum(axis=1, keepdims=True)
        return z

    def sample(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """A class index per row, by inverse CDF with uniforms u. Rows that are all
        identical (every simulated game at the same state) are evaluated once."""
        same = x.shape[0] > 1 and bool((x == x[0]).all())
        h = x[:1] if same else x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.maximum(h @ w + b, 0)
        z = h @ self.weights[-1] + self.biases[-1]
        z -= z.max(axis=1, keepdims=True)
        np.exp(z, out=z)
        np.cumsum(z, axis=1, out=z)  # unnormalized CDF: compare with u * total
        if same:
            k = np.searchsorted(z[0], u * z[0, -1], side="left")
        else:
            k = (z < (u * z[:, -1])[:, None]).sum(1)
        return np.minimum(k, N_CLASSES - 1)

    def result_probs(self, x: np.ndarray) -> np.ndarray:
        """Marginal over RESULTS (summed over duration bins)."""
        p = self.probs(x)
        out = np.zeros((p.shape[0], len(RESULTS)), dtype=p.dtype)
        np.add.at(out.T, CLASS_RESULT, p.T)
        return out

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)} | {f"b{i}": b for i, b in enumerate(self.biases)}
        np.savez(path, season_cap=self.season_cap, n_layers=len(self.weights), quantiles=self.quantiles,
                 **arrays)
        return path

    @classmethod
    def load(cls, path: Path) -> "StepModel":
        z = np.load(path)
        n = int(z["n_layers"])
        return cls([z[f"w{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)], int(z["season_cap"]),
                   z["quantiles"])


def step_model_path(last_season: int) -> Path:
    return config.DATA_DIR / "models" / f"drive_step_{last_season}.npz"


def season_weights(season: np.ndarray, last: int, half_life: float) -> np.ndarray:
    """Recency weights: a season `half_life` years older counts half as much."""
    return np.power(0.5, (last - season) / half_life)


def train(train: pl.DataFrame, val: pl.DataFrame | None = None, hidden: Sequence[int] = (128, 128),
          epochs: int = 30, batch_size: int = 2048, alpha: float = 1e-4, lr: float = 1e-3,
          half_life: float = 6.0, seed: int = 2013, patience: int = 4, log: Log = print) -> StepModel:
    """Fit on `train` (snaps), keeping the epoch with the best log loss on `val` if given."""
    from sklearn.neural_network import MLPClassifier

    train = training_snaps(train)
    cap = int(train["season"].max())
    quantiles = bin_quantiles(train.filter(pl.col("season") > cap - half_life * 2))
    x, y = frame_features(train, cap), class_labels(train)
    w = season_weights(train["season"].to_numpy(), cap, half_life)
    if val is not None:
        val = training_snaps(val)
        xv, yv = frame_features(val, cap), class_labels(val)
    mlp = MLPClassifier(hidden_layer_sizes=tuple(hidden), alpha=alpha, batch_size=batch_size,
                        learning_rate_init=lr, random_state=seed, max_iter=1, shuffle=True)
    classes = np.arange(N_CLASSES)
    rng = np.random.default_rng(seed)
    best, best_loss, stale = None, np.inf, 0
    for epoch in range(epochs):
        t0 = time.time()
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size * 64):
            idx = order[start:start + batch_size * 64]
            mlp.partial_fit(x[idx], y[idx], classes=classes, sample_weight=w[idx])
        model = StepModel([c.astype(np.float32) for c in mlp.coefs_],
                          [b.astype(np.float32) for b in mlp.intercepts_], cap, quantiles)
        if val is None:
            best = model
            log(f"  epoch {epoch + 1}: {time.time() - t0:.0f}s")
            continue
        p = model.probs(xv)
        loss = float(-np.mean(np.log(np.maximum(p[np.arange(len(yv)), yv], 1e-12))))
        log(f"  epoch {epoch + 1}: validation log loss {loss:.4f} ({time.time() - t0:.0f}s)")
        if loss < best_loss - 1e-4:
            best, best_loss, stale = model, loss, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best
