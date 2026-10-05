"""Direct win-probability model (Phase 1): nflfastR's XGBoost models, retrained here.

Two variants, with nflfastR's published hyperparameters (Baldwin 2020):
  wp         no betting line.   Published LOSO calibration error 0.0055, error rate 27%
  wp_spread  with the spread.   Published LOSO calibration error 0.0066, error rate 23%,
             log loss 0.44 (vs 0.52 without the spread)

Training rows follow nflfastR's filters: quarters 1-4, no tied games, and a row
must have a possession team, a play type, an EP value, a yard line and both
timeout counts. The CV protocol is theirs too: leave one season out, 2000-2019.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import polars as pl

from gridline import config
from gridline.data import nflverse
from gridline.eval import bootstrap, metrics
from gridline.models.features import WP_FEATURES, WP_SPREAD_FEATURES, wp_features

Log = Callable[[str], None]

MODELS = {
    "wp": {
        "features": WP_FEATURES,
        "nrounds": 65,
        "params": {"eta": 0.2, "gamma": 0.0, "subsample": 0.8, "colsample_bytree": 0.8,
                   "max_depth": 4, "min_child_weight": 1},
        "published": {"calibration_error": 0.0055, "error_rate": 0.27, "log_loss": 0.52},
        "nflverse": "wp",  # the pbp column nflfastR's shipped version of this model writes
    },
    "wp_spread": {
        "features": WP_SPREAD_FEATURES,
        "nrounds": 534,
        "params": {"eta": 0.05, "gamma": 0.79012017, "subsample": 0.9224245,
                   "colsample_bytree": 5 / 12, "max_depth": 5, "min_child_weight": 7,
                   "monotone_constraints": "(0,0,0,0,0,1,1,-1,-1,-1,1,-1)"},
        "published": {"calibration_error": 0.0066, "error_rate": 0.23, "log_loss": 0.44},
        "nflverse": "vegas_wp",
    },
}
BASE_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
               "seed": 2013}
CV_SEASONS = range(2000, 2020)
# The data has been revised since 2020 and the tree method differs (hist vs R's
# default), so an exact match isn't expected: the gate allows a small tolerance.
GATE_TOLERANCE = 0.001


def model_dir() -> Path:
    return config.DATA_DIR / "models"


def training_frame(seasons: Iterable[int]) -> pl.DataFrame:
    """Features + label for `seasons`, with nflfastR's row filters applied."""
    from gridline.inventory import load_events

    frames = []
    for s in seasons:
        ev = load_events(s)
        # nflfastR also drops rows with no play type or no EP value
        extra = nflverse.load_pbp([s], columns=["game_id", "play_id", "play_type", "ep"])
        keep = ev.select("game_id", "play_id", "seq").join(extra, on=["game_id", "play_id"])
        feats = wp_features(ev).join(keep.select("game_id", "seq", "play_type", "ep"),
                                     on=["game_id", "seq"])
        frames.append(feats)
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.filter(
        (pl.col("qtr") <= 4) & pl.col("label").is_not_null() & pl.col("play_type").is_not_null()
        & pl.col("ep").is_not_null() & pl.col("yardline_100").is_not_null()
        & pl.col("posteam_timeouts_remaining").is_not_null()
        & pl.col("defteam_timeouts_remaining").is_not_null()
    ).sort("season", "game_id", "seq")  # a fixed row order keeps XGBoost's row sampling reproducible


def _dmatrix(df: pl.DataFrame, features: Sequence[str], with_label: bool = True):
    import xgboost as xgb

    x = df.select(features).cast(pl.Float64).to_numpy()
    y = df["label"].to_numpy().astype(float) if with_label else None
    return xgb.DMatrix(x, label=y, feature_names=list(features), missing=np.nan)


def fit(df: pl.DataFrame, kind: str = "wp_spread", nthread: int = 0):
    import xgboost as xgb

    spec = MODELS[kind]
    params = BASE_PARAMS | spec["params"] | ({"nthread": nthread} if nthread else {})
    return xgb.train(params, _dmatrix(df, spec["features"]), num_boost_round=spec["nrounds"])


def predict(booster, df: pl.DataFrame, kind: str = "wp_spread") -> np.ndarray:
    return booster.predict(_dmatrix(df, MODELS[kind]["features"], with_label=False))


def loso_cv(df: pl.DataFrame, kind: str = "wp_spread", seasons: Iterable[int] = CV_SEASONS,
            log: Log = print) -> pl.DataFrame:
    """Leave-one-season-out: for each season, train on the others (within `seasons`)
    and predict the held-out one. Returns the held-out rows with a `wp` column."""
    seasons = list(seasons)
    data = df.filter(pl.col("season").is_in(seasons))
    out = []
    for s in seasons:
        t0 = time.time()
        test = data.filter(pl.col("season") == s)
        booster = fit(data.filter(pl.col("season") != s), kind)
        out.append(test.with_columns(wp=pl.Series(predict(booster, test, kind))))
        log(f"  {kind} holdout {s}: {test.height:,} plays, {time.time() - t0:.0f}s")
    return pl.concat(out)


def with_nflverse(frame: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Attach, as column `nflverse`, the prediction nflfastR's shipped version of this
    model wrote into nflverse's play-by-play (`wp` or `vegas_wp`). Rows it left blank
    are dropped."""
    from gridline.inventory import load_events

    col = MODELS[kind]["nflverse"]
    seasons = sorted(frame["season"].unique().to_list())
    ids = pl.concat([load_events(s).select("game_id", "seq", "play_id") for s in seasons])
    pbp = nflverse.load_pbp(seasons, columns=["game_id", "play_id", col]).rename({col: "nflverse"})
    return (frame.join(ids, on=["game_id", "seq"])
            .join(pbp, on=["game_id", "play_id"], how="left")
            .filter(pl.col("nflverse").is_not_null()))


def report(cv: pl.DataFrame, kind: str, log: Log = print) -> dict:
    """Score held-out predictions against nflfastR's published numbers, with three
    yardsticks for the calibration error: how much it moves when whole games are
    resampled, what one season alone scores, and what nflfastR's shipped model scores
    on the same plays."""
    p, y, q = cv["wp"].to_numpy(), cv["label"].to_numpy(), cv["qtr"].to_numpy()
    m = metrics.summary(p, y, q)
    noise = bootstrap.spread(lambda w: metrics.calibration_error(p, y, q, w), cv["game_id"].to_numpy())
    by_season = [metrics.calibration_error(g["wp"], g["label"], g["qtr"])
                 for _, g in cv.group_by("season")]
    shipped = with_nflverse(cv, kind)
    theirs = metrics.summary(shipped["nflverse"], shipped["label"], shipped["qtr"])
    m |= {"games": cv["game_id"].n_unique(), "calibration_error_sd": noise["sd"],
          "season_min": min(by_season), "season_max": max(by_season), "nflverse": theirs}
    pub = MODELS[kind]["published"]
    log(f"{kind}: calibration error {m['calibration_error']:.4f} (nflfastR {pub['calibration_error']:.4f}) · "
        f"error rate {m['error_rate']:.3f} ({pub['error_rate']:.2f}) · "
        f"log loss {m['log_loss']:.3f} ({pub['log_loss']:.2f}) · brier {m['brier']:.4f} · "
        f"{m['n']:,} plays, {m['games']:,} games")
    log(f"  resampling games moves the calibration error by ±{noise['sd']:.4f} (SD); one held-out "
        f"season on its own scores {m['season_min']:.3f}-{m['season_max']:.3f}")
    log(f"  nflverse's shipped {MODELS[kind]['nflverse']} on {theirs['n']:,} of these plays (in-sample: "
        f"it was fit on these seasons): calibration error {theirs['calibration_error']:.4f} · "
        f"error rate {theirs['error_rate']:.3f} · log loss {theirs['log_loss']:.3f} · "
        f"brier {theirs['brier']:.4f}")
    return m


def cv_path(kind: str) -> Path:
    return config.DERIVED_DIR / "wp_cv" / f"{kind}.parquet"


def save_cv(cv: pl.DataFrame, kind: str) -> Path:
    """Keep the held-out predictions (scripts/make_figures.py draws the calibration chart)."""
    path = cv_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.select("game_id", "seq", "season", "qtr", "label", "wp").write_parquet(path)
    return path


def save(booster, kind: str, seasons: Sequence[int]) -> Path:
    path = model_dir() / f"{kind}_{min(seasons)}_{max(seasons)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(path)
    return path


def forward_test(train_seasons: Sequence[int], test_season: int, kind: str = "wp_spread",
                 log: Log = print) -> dict:
    """Train on `train_seasons`, predict `test_season`, and score our model and nflverse's
    published one on the same plays, with a paired interval on the difference."""
    train = training_frame(train_seasons)
    test = training_frame([test_season])
    booster = fit(train, kind)
    path = save(booster, kind, train_seasons)
    # one frame carries both predictions, the labels and the quarters, so rows stay aligned
    scored = with_nflverse(test.with_columns(gridline=pl.Series(predict(booster, test, kind))), kind)
    y, q, games = scored["label"].to_numpy(), scored["qtr"].to_numpy(), scored["game_id"].to_numpy()
    ours, theirs = scored["gridline"].to_numpy(), scored["nflverse"].to_numpy()
    name = f"nflverse {MODELS[kind]['nflverse']}"
    res = {"gridline": metrics.summary(ours, y, q), name: metrics.summary(theirs, y, q),
           "difference": bootstrap.paired(ours, theirs, y, q, games)}
    log(f"{kind}: trained on {min(train_seasons)}-{max(train_seasons)} ({train.height:,} plays), "
        f"tested on {test_season} ({scored.height:,} plays, {scored['game_id'].n_unique()} games); "
        f"model saved to {path}")
    for label in ("gridline", name):
        m = res[label]
        log(f"  {label:<18} calibration error {m['calibration_error']:.4f} · log loss "
            f"{m['log_loss']:.4f} · brier {m['brier']:.4f} · error rate {m['error_rate']:.3f}")
    log("  gridline minus nflverse, with 90% intervals from resampling games: " + " · ".join(
        f"{k.replace('_', ' ')} {d['diff']:+.4f} [{d['p05']:+.4f}, {d['p95']:+.4f}]"
        for k, d in res["difference"].items()))
    return res
