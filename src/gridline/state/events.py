"""Play events: nflverse play-by-play -> one cleaned, timed row per pbp row.

Two facts about nflverse drive this module (verified on 2025 data, docs/DATA.md):

1. Scores. total_home_score / total_away_score are POST-play: they already include
   the row's own points. posteam_score / score_differential are PRE-play. Treating
   total_*_score as "the current score" leaks the play's result into its own
   features. Here home_score_post = the row's own total and home_score_pre = the
   previous row's post score (0 before the first row).
   Out-of-order timeout rows carry the score as of their true (earlier) time, so a
   plain lag would briefly "un-score" a touchdown (~250 plays a season, 2022-25).
   Scores never go down, so post scores are repaired with a running max; that
   agrees with nflverse's own pre-play scores on 99.96% of play rows since 1999.

2. Time. time_of_day is the wall-clock snap time (UTC; ~97% of rows since 2020)
   and end_clock_time is the wall-clock end of the play (~83%; median play 4 s).
   Timeout rows are logged out of play_id order (their timestamps are right,
   their position isn't), a few timestamps are hours off, and administrative
   rows (game start, quarter ends, two-minute warning) have none.

Cleaning, per game in play_id order. It is conservative on purpose: a fact may
become known late, never early.
  a. drop wild snaps: more than OUTLIER_S from the median of nearby valid snaps
  b. flag out-of-order snaps (earlier than some previous row's snap)
  c. impute a missing snap as the previous row's end
  d. make snaps monotone (running max): an out-of-order row is placed no earlier
     than the rows before it
  e. a missing or implausible end (before its snap, or > MAX_PLAY_S after it)
     becomes snap + the median duration of that play type
  f. make ends monotone too, never before their own snap

The publication delay is not baked in: with_known_times(events, delta_s) adds
t_known = running max of (t_end + delta_s), the moment a play's result is
treated as public. Evaluation sweeps delta_s rather than pretending to know it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

OUTLIER_S = 45 * 60   # halftime is <= ~30 min, so a real snap never sits this far off
MAX_PLAY_S = 120      # no single play lasts 2 minutes of wall-clock
NEIGHBOURS = 4        # valid snaps on each side used for the outlier median
FALLBACK_DURATION_S = 5.0
ADMIN_DURATION_S = 0.0  # rows with no play_type (quarter end, timeout notes...)

PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "home_team", "away_team",
    "play_type", "desc", "time_of_day", "end_clock_time",
    "qtr", "game_seconds_remaining", "half_seconds_remaining", "game_half",
    "posteam", "down", "ydstogo", "yardline_100", "goal_to_go",
    "home_timeouts_remaining", "away_timeouts_remaining",
    "total_home_score", "total_away_score",
    "spread_line", "total_line", "home_opening_kickoff", "result", "total",
]


_ISO_PREFIX = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"


def _epoch_seconds(col: str) -> pl.Expr:
    """Wall-clock ISO timestamps ('2025-09-07T17:02:38.787Z', with or without
    milliseconds) -> float epoch seconds. Anything else is null: seasons before
    2005 store a bare 'HH:MM:SS' with no date or zone, and their end_clock_time is
    the *game* clock ('MM:SS'), not wall-clock time."""
    c = pl.col(col)
    return (
        pl.when(c.str.contains(_ISO_PREFIX))
        .then(c.str.strip_suffix("Z"))
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f", strict=False, time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.epoch("ms")
        .cast(pl.Float64)
        / 1000.0
    )


def clean_game_times(
    snap: np.ndarray, end: np.ndarray, duration: np.ndarray
) -> dict[str, np.ndarray]:
    """Clean one game's timestamps (float epoch seconds, NaN = missing), rows in
    play_id order. `duration` is the per-row fallback play length in seconds.
    Returns t_snap, t_end and flags; see the module docstring for the rules."""
    n = len(snap)
    snap = np.asarray(snap, dtype=float).copy()
    end = np.asarray(end, dtype=float)

    # a. wild snaps
    wild = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(~np.isnan(snap))
    if len(idx) >= 3:
        vals = snap[idx]
        for k in range(len(idx)):
            lo, hi = max(0, k - NEIGHBOURS), min(len(idx), k + NEIGHBOURS + 1)
            neighbours = np.delete(vals[lo:hi], k - lo)
            if abs(vals[k] - np.median(neighbours)) > OUTLIER_S:
                wild[idx[k]] = True
    snap[wild] = np.nan
    missing = np.isnan(snap)

    # b. out of order, judged on the surviving raw snaps
    running = np.fmax.accumulate(np.where(missing, -np.inf, snap))
    prev_max = np.concatenate([[-np.inf], running[:-1]])
    out_of_order = ~missing & (snap < prev_max)

    # e. which raw ends to distrust (judged against the raw snap)
    with np.errstate(invalid="ignore"):
        bad_end = np.isnan(end) | missing | (end < snap) | (end - snap > MAX_PLAY_S)

    # the first valid snap at or after each row, for rows before any valid snap
    next_valid = snap.copy()
    for i in range(n - 2, -1, -1):
        if np.isnan(next_valid[i]):
            next_valid[i] = next_valid[i + 1]

    t_snap, t_end = np.full(n, np.nan), np.full(n, np.nan)
    last_snap = last_end = -np.inf
    for i in range(n):
        s = snap[i]
        if np.isnan(s):  # c. missing: happened after the previous row ended
            s = last_end if np.isfinite(last_end) else next_valid[i]
        if np.isnan(s):  # no valid time anywhere in this game
            continue
        s = max(s, last_snap)  # d.
        e = s + duration[i] if bad_end[i] else end[i]  # e.
        e = max(e, s, last_end)  # f.
        t_snap[i], t_end[i] = s, e
        last_snap, last_end = s, e

    return {
        "t_snap": t_snap,
        "t_end": t_end,
        "snap_imputed": missing,
        "snap_wild": wild,
        "snap_out_of_order": out_of_order,
        "end_imputed": bad_end,
    }


def _durations(df: pl.DataFrame) -> np.ndarray:
    """Per-row fallback duration: the median observed play length for the row's
    play type, estimated from `df` itself; admin rows (no play_type) get 0."""
    obs = (
        df.filter(
            pl.col("snap_raw").is_not_null() & pl.col("end_raw").is_not_null()
            & (pl.col("end_raw") >= pl.col("snap_raw"))
            & (pl.col("end_raw") - pl.col("snap_raw") <= MAX_PLAY_S)
        )
        .group_by("play_type")
        .agg((pl.col("end_raw") - pl.col("snap_raw")).median().alias("dur"))
    )
    table = dict(zip(obs["play_type"].to_list(), obs["dur"].to_list()))
    return np.array([
        ADMIN_DURATION_S if pt is None else table.get(pt, FALLBACK_DURATION_S)
        for pt in df["play_type"].to_list()
    ])


def build_play_events(pbp: pl.DataFrame) -> pl.DataFrame:
    """nflverse pbp (any number of games) -> cleaned play events, one row per pbp row."""
    df = (
        pbp.select(PBP_COLUMNS)
        .sort("game_id", "play_id")
        .with_columns(
            snap_raw=_epoch_seconds("time_of_day"),
            end_raw=_epoch_seconds("end_clock_time"),
        )
    )
    duration = _durations(df)
    snap = df["snap_raw"].to_numpy(allow_copy=True).astype(float)
    end = df["end_raw"].to_numpy(allow_copy=True).astype(float)

    cols = {k: np.empty(df.height, dtype=float if k.startswith("t_") else bool)
            for k in ("t_snap", "t_end", "snap_imputed", "snap_wild",
                      "snap_out_of_order", "end_imputed")}
    game_ids = df["game_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, game_ids[1:] != game_ids[:-1]])
    bounds = np.r_[starts, df.height]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        res = clean_game_times(snap[lo:hi], end[lo:hi], duration[lo:hi])
        for k, v in res.items():
            cols[k][lo:hi] = v

    def to_ts(a: np.ndarray) -> pl.Series:
        ms = pl.Series(np.round(a * 1000.0)).cast(pl.Int64, strict=False)  # NaN -> null
        return ms.cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")

    g = "game_id"
    return (
        df.with_columns(
            t_snap=to_ts(cols["t_snap"]),
            t_end=to_ts(cols["t_end"]),
            **{k: pl.Series(cols[k]) for k in
               ("snap_imputed", "snap_wild", "snap_out_of_order", "end_imputed")},
        )
        .with_columns(
            seq=pl.int_range(pl.len()).over(g),
            home_score_post=pl.col("total_home_score").forward_fill().fill_null(0)
            .cum_max().over(g),
            away_score_post=pl.col("total_away_score").forward_fill().fill_null(0)
            .cum_max().over(g),
        )
        .with_columns(
            home_score_pre=pl.col("home_score_post").shift(1).over(g).fill_null(0),
            away_score_pre=pl.col("away_score_post").shift(1).over(g).fill_null(0),
            posteam_is_home=pl.col("posteam") == pl.col("home_team"),
        )
        .select(
            "game_id", "season", "week", "season_type", "home_team", "away_team",
            "play_id", "seq", "play_type", "desc",
            "t_snap", "t_end", "snap_imputed", "snap_wild", "snap_out_of_order", "end_imputed",
            "qtr", "game_seconds_remaining", "half_seconds_remaining", "game_half",
            "posteam", "posteam_is_home", "down", "ydstogo", "yardline_100", "goal_to_go",
            pl.col("home_timeouts_remaining").alias("home_timeouts"),
            pl.col("away_timeouts_remaining").alias("away_timeouts"),
            "home_score_pre", "away_score_pre", "home_score_post", "away_score_post",
            pl.col("spread_line").alias("pregame_spread"),
            pl.col("total_line").alias("pregame_total"),
            "home_opening_kickoff",
            # outcomes: labels only, never features (named so a leak is obvious)
            pl.col("result").alias("final_margin_home"),
            pl.col("total").alias("final_total"),
        )
    )


def with_known_times(events: pl.DataFrame, delta_s: float = 0.0) -> pl.DataFrame:
    """Add t_known: when each row's result is treated as public (play end + delta),
    kept monotone within a game."""
    return events.with_columns(
        t_known=(pl.col("t_end") + pl.duration(milliseconds=round(delta_s * 1000)))
        .cum_max()
        .over("game_id")
    )


def timing_summary(events: pl.DataFrame) -> pl.DataFrame:
    """Per-season timestamp quality over real plays (rows with a play_type)."""
    plays = events.filter(pl.col("play_type").is_not_null())
    return (
        plays.group_by("season")
        .agg(
            games=pl.col("game_id").n_unique(),
            plays=pl.len(),
            snap_raw=(~pl.col("snap_imputed")).mean(),
            end_raw=(~pl.col("end_imputed")).mean(),
            out_of_order=pl.col("snap_out_of_order").mean(),
            wild=pl.col("snap_wild").sum(),
            untimed=pl.col("t_snap").is_null().mean(),
        )
        .sort("season")
    )
