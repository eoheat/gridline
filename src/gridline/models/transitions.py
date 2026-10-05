"""What happens between two drives, sampled from recent history.

After a drive ends the simulator needs: who has the ball at the next snap and where,
how much clock ran in between, and any points scored in between (a kick returned for a
touchdown). These are drawn from pools of real transitions, bucketed by what drives them:

- Kicks, seen from the kicking team: after a touchdown or field goal and to start a
  half ("normal"), when the kicker trails late in the game ("onside"), and free kicks
  after a safety.
- Changes of possession (punt, turnover, turnover on downs, missed field goal), seen
  from the offense: bucketed by the result, the snap's yard line (5-yard bins) and
  whether it was 3rd or 4th down. A thin bucket borrows from coarser ones.

Kickoff rules change almost every year (touchback spot 20, then 25, 30, 35; the 2024
redesign), so normal kicks come from the most recent season only. Everything else pools
several seasons. Rare events (onside recoveries, muffed punts, return touchdowns) come
along at their historical rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from gridline import config

KICK_NORMAL, KICK_ONSIDE, KICK_FREE = 0, 1, 2
POSSESSION_RESULTS = ("punt", "turnover", "downs", "fg_miss")
N_YBINS = 20
MIN_BUCKET = 40
ONSIDE_SECONDS = 300  # "late": the last 5 minutes of the game


@dataclass
class Pool:
    """Samples in buckets: bucket b owns rows start[b] .. start[b] + count[b]."""

    start: np.ndarray
    count: np.ndarray
    same: np.ndarray  # next snap belongs to the kicker (kicks) / the offense (possession)
    yardline: np.ndarray  # next snap's yardline_100, from whoever has the ball
    seconds: np.ndarray  # clock between the drive's end and the next snap
    points_a: np.ndarray  # points in between for the kicker / the offense
    points_b: np.ndarray  # ... for the receiver / the defence

    def draw(self, bucket: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Row indices for buckets `bucket` from uniforms `u`."""
        return self.start[bucket] + np.minimum((u * self.count[bucket]).astype(np.int64),
                                               self.count[bucket] - 1)

    @classmethod
    def from_frame(cls, df: pl.DataFrame, n_buckets: int) -> "Pool":
        df = df.sort("bucket")
        count = np.bincount(df["bucket"].to_numpy(), minlength=n_buckets)
        start = np.concatenate([[0], np.cumsum(count)[:-1]])
        return cls(start, count, df["same"].to_numpy().astype(bool),
                   df["yardline"].to_numpy().astype(np.float64),
                   df["seconds"].to_numpy().astype(np.float64),
                   df["points_a"].to_numpy().astype(np.float64),
                   df["points_b"].to_numpy().astype(np.float64))

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}_{k}": getattr(self, k) for k in
                ("start", "count", "same", "yardline", "seconds", "points_a", "points_b")}

    @classmethod
    def from_arrays(cls, z, prefix: str) -> "Pool":
        return cls(*(z[f"{prefix}_{k}"] for k in
                     ("start", "count", "same", "yardline", "seconds", "points_a", "points_b")))


def possession_bucket(result_idx: np.ndarray, yardline: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Bucket id for (result in POSSESSION_RESULTS, 5-yard bin, 3rd/4th down)."""
    ybin = np.clip((np.asarray(yardline) - 1) // 5, 0, N_YBINS - 1).astype(np.int64)
    late = (np.asarray(down) >= 3).astype(np.int64)
    return (np.asarray(result_idx) * N_YBINS + ybin) * 2 + late


N_POSSESSION_BUCKETS = len(POSSESSION_RESULTS) * N_YBINS * 2

# Timeouts used between one drive's first snap and the next one's, by time left in the
# half, the offence's lead and the half. Late in a half they are used on most drives.
TIMEOUT_CLOCK = (120, 300, 600)  # seconds left in the half: <=120, <=300, <=600, more
TIMEOUT_LEAD = (-8.5, -0.5, 0.5, 8.5)  # offence trails by 9+, 1-8, tied, leads 1-8, 9+
N_TIMEOUT_BUCKETS = (len(TIMEOUT_CLOCK) + 1) * (len(TIMEOUT_LEAD) + 1) * 2


def timeout_bucket(half_seconds: np.ndarray, lead: np.ndarray, second_half: np.ndarray) -> np.ndarray:
    c = np.searchsorted(TIMEOUT_CLOCK, np.asarray(half_seconds), side="left")
    lead_b = np.searchsorted(TIMEOUT_LEAD, np.asarray(lead))
    return (c * (len(TIMEOUT_LEAD) + 1) + lead_b) * 2 + np.asarray(second_half).astype(np.int64)


@dataclass
class Transitions:
    kicks: Pool
    possession: Pool
    opp_td_conversion: np.ndarray  # P(defence's conversion adds 0, 1, 2)
    pat: np.ndarray  # P(0, 1, 2 extra points) by conversion context (see pat_context)
    timeouts: Pool  # points_a / points_b hold the timeouts the offence / defence used
    seasons: tuple[int, int] = field(default=(0, 0))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.kicks.arrays("kick"), **self.possession.arrays("pos"),
                 **self.timeouts.arrays("to"), opp_td_conversion=self.opp_td_conversion, pat=self.pat,
                 seasons=np.array(self.seasons))
        return path

    @classmethod
    def load(cls, path: Path) -> "Transitions":
        z = np.load(path)
        return cls(Pool.from_arrays(z, "kick"), Pool.from_arrays(z, "pos"), z["opp_td_conversion"],
                   z["pat"], Pool.from_arrays(z, "to"), tuple(int(s) for s in z["seasons"]))


def transitions_path(last_season: int) -> Path:
    return config.DATA_DIR / "models" / f"transitions_{last_season}.npz"


# A pending conversion's odds depend on the scorer's lead after the six points and on
# whether it is late in the game (two-point decisions follow the scoreboard then).
PAT_DIFFS = np.arange(-16, 17)


def pat_context(lead_after_six: np.ndarray, late: np.ndarray) -> np.ndarray:
    d = np.clip(np.asarray(lead_after_six), PAT_DIFFS[0], PAT_DIFFS[-1]) - PAT_DIFFS[0]
    return d * 2 + np.asarray(late).astype(np.int64)


def _kick_frame(d: pl.DataFrame) -> pl.DataFrame:
    """Kicks from the drives table, seen from the kicking team."""
    home_off = pl.col("offense") == pl.col("home_team")
    lead_off = pl.when(home_off).then(pl.col("home_post") - pl.col("away_post")).otherwise(
        pl.col("away_post") - pl.col("home_post"))
    late = (pl.col("half") == "Half2") & (pl.col("end_gsr") <= ONSIDE_SECONDS)
    ok = (pl.col("next_kind") == "same_half") & pl.col("gap_points_off").ge(0) & pl.col("gap_points_def").ge(0)
    off_kicks = d.filter(ok & pl.col("result").is_in(["td0", "td1", "td2", "fg"])).select(
        bucket=pl.when(late & (lead_off < 0)).then(KICK_ONSIDE).otherwise(KICK_NORMAL),
        same=pl.col("next_same_offense"), yardline=pl.col("next_yardline"),
        seconds=pl.col("gap_seconds"), points_a=pl.col("gap_points_off"),
        points_b=pl.col("gap_points_def"), season=pl.col("season"))
    def_kicks = d.filter(ok & (pl.col("result") == "opp_td")).select(
        bucket=pl.when(late & (lead_off > 0)).then(KICK_ONSIDE).otherwise(KICK_NORMAL),
        same=~pl.col("next_same_offense"), yardline=pl.col("next_yardline"),
        seconds=pl.col("gap_seconds"), points_a=pl.col("gap_points_def"),
        points_b=pl.col("gap_points_off"), season=pl.col("season"))
    free = d.filter(ok & (pl.col("result") == "safety")).select(
        bucket=pl.lit(KICK_FREE), same=pl.col("next_same_offense"), yardline=pl.col("next_yardline"),
        seconds=pl.col("gap_seconds"), points_a=pl.col("gap_points_off"),
        points_b=pl.col("gap_points_def"), season=pl.col("season"))
    return pl.concat([off_kicks, def_kicks, free], how="diagonal_relaxed").with_columns(
        pl.col("bucket").cast(pl.Int64))


def build(drives: pl.DataFrame, snaps: pl.DataFrame, last_season: int,
          kick_seasons: int = 1, onside_seasons: int = 4, free_kick_seasons: int = 10,
          possession_seasons: int = 8) -> Transitions:
    """Pools from seasons up to `last_season` (inclusive), each looking back its own
    number of seasons."""
    d = drives.filter(pl.col("result").is_not_null() & (pl.col("season") <= last_season))
    kicks = _kick_frame(d)
    since = {KICK_NORMAL: kick_seasons, KICK_ONSIDE: onside_seasons, KICK_FREE: free_kick_seasons}
    kicks = pl.concat([kicks.filter((pl.col("bucket") == b) & (pl.col("season") > last_season - n))
                       for b, n in since.items()])

    s = snaps.filter(pl.col("result").is_in(POSSESSION_RESULTS) & (pl.col("next_kind") == "same_half")
                     & (pl.col("season") <= last_season) & (pl.col("season") > last_season - possession_seasons)
                     & pl.col("gap_points_off").ge(0) & pl.col("gap_points_def").ge(0))
    ridx = s["result"].replace_strict({r: i for i, r in enumerate(POSSESSION_RESULTS)},
                                      return_dtype=pl.Int64).to_numpy()
    pos = pl.DataFrame({
        "bucket": possession_bucket(ridx, s["yardline_100"].to_numpy(), s["down"].to_numpy()),
        "same": s["next_same_offense"], "yardline": s["next_yardline"], "seconds": s["gap_seconds"],
        "points_a": s["gap_points_off"], "points_b": s["gap_points_def"],
        "result": ridx, "ybin": np.clip((s["yardline_100"].to_numpy() - 1) // 5, 0, N_YBINS - 1),
    })
    pos = _fill_thin_buckets(pos)

    opp = d.filter((pl.col("result") == "opp_td") & (pl.col("season") > last_season - possession_seasons))
    conv = np.bincount(np.clip(opp["def_points"].to_numpy().astype(int) - 6, 0, 2), minlength=3)
    return Transitions(Pool.from_frame(kicks, 3), Pool.from_frame(pos, N_POSSESSION_BUCKETS),
                       conv / conv.sum(), _pat_table(d, last_season),
                       _timeout_pool(drives, last_season, possession_seasons),
                       (last_season - possession_seasons + 1, last_season))


def _timeout_pool(drives: pl.DataFrame, last_season: int, seasons: int) -> Pool:
    """Timeouts each side used from one drive's first snap to the next drive's."""
    d = drives.filter((pl.col("season") <= last_season) & (pl.col("season") > last_season - seasons)
                      & (pl.col("n_snaps") > 0)).sort("game_id", "first_seq")
    home_off = pl.col("offense") == pl.col("home_team")
    nxt_h = pl.col("snap_home_timeouts").shift(-1).over("game_id")
    nxt_a = pl.col("snap_away_timeouts").shift(-1).over("game_id")
    same_half = pl.col("snap_half").shift(-1).over("game_id") == pl.col("snap_half")
    d = d.with_columns(
        used_h=pl.col("snap_home_timeouts") - nxt_h, used_a=pl.col("snap_away_timeouts") - nxt_a,
        lead=pl.when(home_off).then(pl.col("snap_home_pre") - pl.col("snap_away_pre"))
        .otherwise(pl.col("snap_away_pre") - pl.col("snap_home_pre")),
        half_seconds=pl.when(pl.col("snap_half") == "Half1").then(pl.col("snap_gsr") - 1800)
        .otherwise(pl.col("snap_gsr")),
    ).filter(same_half & pl.col("used_h").is_not_null() & pl.col("used_a").is_not_null()
             & (pl.col("snap_half") != "Overtime") & pl.col("lead").is_not_null()
             & pl.col("half_seconds").is_not_null())
    used_off = d.select(pl.when(home_off).then(pl.col("used_h")).otherwise(pl.col("used_a"))).to_series()
    used_def = d.select(pl.when(home_off).then(pl.col("used_a")).otherwise(pl.col("used_h"))).to_series()
    frame = pl.DataFrame({
        "bucket": timeout_bucket(d["half_seconds"].to_numpy(), d["lead"].to_numpy(),
                                 (d["snap_half"] == "Half2").to_numpy()),
        "same": np.zeros(d.height, bool), "yardline": np.zeros(d.height), "seconds": np.zeros(d.height),
        "points_a": used_off.clip(0, 3).to_numpy(), "points_b": used_def.clip(0, 3).to_numpy(),
    })
    return Pool.from_frame(frame, N_TIMEOUT_BUCKETS)


def _fill_thin_buckets(pos: pl.DataFrame) -> pl.DataFrame:
    """Give every (result, yard bin, down) bucket at least MIN_BUCKET rows: a thin bucket
    takes the rows of the same result and yard bin at any down, then of neighbouring yard
    bins, then of the whole result."""
    out = []
    counts = pos.group_by("bucket").len()
    have = dict(zip(counts["bucket"].to_list(), counts["len"].to_list()))
    for r in range(len(POSSESSION_RESULTS)):
        pr = pos.filter(pl.col("result") == r)
        for ybin in range(N_YBINS):
            for late in (0, 1):
                b = (r * N_YBINS + ybin) * 2 + late
                if have.get(b, 0) >= MIN_BUCKET:
                    rows = pr.filter(pl.col("bucket") == b)
                else:
                    rows = pr.filter(pl.col("ybin") == ybin)
                    width = 1
                    while rows.height < MIN_BUCKET and width < N_YBINS:
                        rows = pr.filter((pl.col("ybin") - ybin).abs() <= width)
                        width += 1
                out.append(rows.with_columns(bucket=pl.lit(b, dtype=pl.Int64)))
    return pl.concat(out)


def _pat_table(d: pl.DataFrame, last_season: int, seasons: int = 6) -> np.ndarray:
    """P(0, 1, 2 points after a touchdown) by pat_context, from recent touchdown drives;
    contexts with little data borrow the overall rates."""
    home_off = pl.col("offense") == pl.col("home_team")
    td = d.filter(pl.col("result").is_in(["td0", "td1", "td2"]) & (pl.col("season") > last_season - seasons))
    td = td.filter(pl.col("home_pre").is_not_null() & pl.col("away_pre").is_not_null())
    lead_after_six = td.select(pl.when(home_off).then(pl.col("home_pre") - pl.col("away_pre")).otherwise(
        pl.col("away_pre") - pl.col("home_pre")) + 6).to_series().to_numpy().astype(np.int64)
    late = td.select(((pl.col("half") == "Half2") & (pl.col("end_gsr") <= 900)).fill_null(False)
                     ).to_series().to_numpy().astype(bool)
    k = td["result"].replace_strict({"td0": 0, "td1": 1, "td2": 2}, return_dtype=pl.Int64).to_numpy()
    ctx = pat_context(lead_after_six, late)
    n_ctx = len(PAT_DIFFS) * 2
    counts = np.zeros((n_ctx, 3))
    np.add.at(counts, (ctx, k), 1)
    overall = counts.sum(0) / counts.sum()
    prior = 20.0  # pseudo-observations of the overall rates
    return (counts + prior * overall) / (counts.sum(1, keepdims=True) + prior)
