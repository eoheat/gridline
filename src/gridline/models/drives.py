"""Drive-level training data for the Phase 2 simulator.

nflverse numbers drives (`fixed_drive`) and names their outcomes
(`fixed_drive_result`). This module turns that into, for every scrimmage snap,
what the rest of its drive did and what happened between the drive's end and the
next scrimmage snap. Labels are from the offense's point of view.

A scrimmage snap is a row with a possession team, a down and a yard line: runs,
passes, punts, field goals, kneels, spikes and pre-snap penalties. Kickoffs,
extra points and two-point tries are not snaps. They are part of a result
(conversions) or of the gap between two drives (kickoffs, and the kick returns for
touchdowns that nflverse books as drives of their own).

Per drive with at least one snap:
  result          one of RESULTS; the conversion is folded into touchdowns (td1 = TD
                  plus a made extra point, td2 = plus a two-point conversion)
  end_gsr         game clock when the drive ended
  next_*          the next drive with a snap: same offense or not, its first yard line
                  and clock, and the points scored in between (gap_*)
Per snap:
  seconds         game clock from the snap to the end of its drive
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import polars as pl

from gridline import config
from gridline.data import nflverse

Log = Callable[[str], None]

RESULTS = ("td0", "td1", "td2", "fg", "fg_miss", "punt", "turnover", "downs", "safety",
           "opp_td", "end_half")
OFFENSE_POINTS = {"td0": 6, "td1": 7, "td2": 8, "fg": 3}
DEFENSE_POINTS = {"safety": 2}  # opp_td: 6-8, the defence's own conversion (def_points)
SCORES = frozenset({"td0", "td1", "td2", "fg", "safety", "opp_td"})

SNAP = (pl.col("posteam").is_not_null() & pl.col("down").is_not_null()
        & pl.col("yardline_100").is_not_null())
PBP_COLUMNS = ["game_id", "play_id", "fixed_drive", "fixed_drive_result"]


def snaps_path(season: int) -> Path:
    return config.DERIVED_DIR / "drives" / f"snaps_{season}.parquet"


def drives_path(season: int) -> Path:
    return config.DERIVED_DIR / "drives" / f"drives_{season}.parquet"


def _half_end(half: pl.Expr) -> pl.Expr:
    return pl.when(half == "Half1").then(1800.0).otherwise(0.0)


def drive_table(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, drive) from events rows joined with nflverse's drive columns,
    sorted by game and play order. Drives without a snap are kept: their points belong
    to the gap before the next drive that has one."""
    rows = rows.filter(pl.col("fixed_drive").is_not_null())
    d = (
        rows.group_by("game_id", "fixed_drive", maintain_order=True)
        .agg(
            season=pl.col("season").first(), season_type=pl.col("season_type").first(),
            home_team=pl.col("home_team").first(), away_team=pl.col("away_team").first(),
            first_seq=pl.col("seq").min(),
            n_snaps=SNAP.sum(),
            n_offenses=pl.col("posteam").filter(SNAP).n_unique(),
            offense=pl.col("posteam").filter(SNAP).first(),
            half=pl.col("game_half").first(),
            snap_half=pl.col("game_half").filter(SNAP).first(),
            gsr_start=pl.col("game_seconds_remaining").first(),
            snap_gsr=pl.col("game_seconds_remaining").filter(SNAP).first(),
            snap_yardline=pl.col("yardline_100").filter(SNAP).first(),
            snap_home_pre=pl.col("home_score_pre").filter(SNAP).first(),
            snap_away_pre=pl.col("away_score_pre").filter(SNAP).first(),
            snap_home_timeouts=pl.col("home_timeouts").filter(SNAP).first(),
            snap_away_timeouts=pl.col("away_timeouts").filter(SNAP).first(),
            result_raw=pl.col("fixed_drive_result").first(),
            home_pre=pl.col("home_score_pre").first(), away_pre=pl.col("away_score_pre").first(),
            home_post=pl.col("home_score_post").last(), away_post=pl.col("away_score_post").last(),
            final_home=(pl.col("final_margin_home").first() + pl.col("final_total").first()) / 2,
            final_away=(pl.col("final_total").first() - pl.col("final_margin_home").first()) / 2,
        )
        .sort("game_id", "first_seq")
    )
    # the clock stops between drives, so a drive ends when the next one starts,
    # unless the half ends first
    nxt = pl.col("half").shift(-1).over("game_id")
    d = d.with_columns(
        end_gsr=pl.when(nxt == pl.col("half")).then(pl.col("gsr_start").shift(-1).over("game_id"))
        .otherwise(_half_end(pl.col("half")))
    )
    home_off = pl.col("offense") == pl.col("home_team")
    off_pts = pl.when(home_off).then(pl.col("home_post") - pl.col("home_pre")).otherwise(
        pl.col("away_post") - pl.col("away_pre"))
    def_pts = pl.when(home_off).then(pl.col("away_post") - pl.col("away_pre")).otherwise(
        pl.col("home_post") - pl.col("home_pre"))
    return d.with_columns(off_points=off_pts, def_points=def_pts)


def label_results(d: pl.DataFrame) -> pl.DataFrame:
    """Map nflverse's drive outcome and the points actually scored to RESULTS. A drive
    whose points don't fit its outcome gets result None (dropped from training)."""
    raw, op, dp = pl.col("result_raw"), pl.col("off_points"), pl.col("def_points")
    clean_def = dp == 0
    result = (
        pl.when((raw == "Touchdown") & op.is_in([6, 7, 8]) & clean_def)
        .then(pl.concat_str(pl.lit("td"), (op - 6).cast(pl.Int32).cast(pl.Utf8)))
        .when((raw == "Field goal") & (op == 3) & clean_def).then(pl.lit("fg"))
        .when((raw == "Missed field goal") & (op == 0) & clean_def).then(pl.lit("fg_miss"))
        .when((raw == "Punt") & (op == 0) & clean_def).then(pl.lit("punt"))
        .when((raw == "Turnover") & (op == 0) & clean_def).then(pl.lit("turnover"))
        .when((raw == "Turnover on downs") & (op == 0) & clean_def).then(pl.lit("downs"))
        .when((raw == "Safety") & (op == 0) & (dp == 2)).then(pl.lit("safety"))
        .when((raw == "Opp touchdown") & (op == 0) & dp.is_in([6, 7, 8])).then(pl.lit("opp_td"))
        .when((raw == "End of half") & (op == 0) & clean_def).then(pl.lit("end_half"))
        .otherwise(None)
    )
    return d.with_columns(result=result)


def add_transitions(d: pl.DataFrame) -> pl.DataFrame:
    """For each drive with a snap: the next drive with a snap in the same game, and the
    points scored in between (from the drive's end to that drive's first snap)."""
    s = d.filter((pl.col("n_snaps") > 0) & (pl.col("n_offenses") == 1)).sort("game_id", "first_seq")

    def nxt(c: str) -> pl.Expr:
        return pl.col(c).shift(-1).over("game_id")

    home_off = pl.col("offense") == pl.col("home_team")
    last = nxt("offense").is_null()
    # after the last drive the "gap" runs to the final score
    gap_home = pl.when(last).then(pl.col("final_home")).otherwise(nxt("snap_home_pre")) - pl.col("home_post")
    gap_away = pl.when(last).then(pl.col("final_away")).otherwise(nxt("snap_away_pre")) - pl.col("away_post")
    return s.with_columns(
        next_kind=pl.when(last).then(pl.lit("game_end"))
        .when(nxt("snap_half") == pl.col("snap_half")).then(pl.lit("same_half"))
        .otherwise(pl.lit("next_half")),
        next_same_offense=nxt("offense") == pl.col("offense"),
        next_yardline=nxt("snap_yardline"),
        next_gsr=nxt("snap_gsr"),
        gap_seconds=pl.when(nxt("snap_half") == pl.col("snap_half"))
        .then(pl.col("end_gsr") - nxt("snap_gsr")).otherwise(None),
        gap_points_off=pl.when(home_off).then(gap_home).otherwise(gap_away),
        gap_points_def=pl.when(home_off).then(gap_away).otherwise(gap_home),
    )


def season_rows(season: int) -> pl.DataFrame:
    from gridline.inventory import load_events

    pbp = nflverse.load_pbp([season], columns=PBP_COLUMNS)
    return load_events(season).join(pbp, on=["game_id", "play_id"], how="left").sort("game_id", "seq")


def build(rows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(drives, snaps) for one season's rows."""
    drives = add_transitions(label_results(drive_table(rows)))
    keep = drives.select(
        "game_id", "fixed_drive", "result", "end_gsr", "def_points", "next_kind",
        "next_same_offense", "next_yardline", "next_gsr", "gap_seconds", "gap_points_off",
        "gap_points_def", pl.col("snap_yardline").alias("drive_start_yardline"),
        pl.col("snap_gsr").alias("drive_start_gsr"),
    )
    snaps = (
        rows.filter(SNAP & pl.col("fixed_drive").is_not_null())
        .join(keep, on=["game_id", "fixed_drive"], how="inner")
        .with_columns(
            is_home=pl.col("posteam") == pl.col("home_team"),
            seconds=(pl.col("game_seconds_remaining") - pl.col("end_gsr")).clip(0, None),
        )
        .with_columns(
            score_diff=pl.when(pl.col("is_home")).then(pl.col("home_score_pre") - pl.col("away_score_pre"))
            .otherwise(pl.col("away_score_pre") - pl.col("home_score_pre")),
            timeouts_off=pl.when(pl.col("is_home")).then(pl.col("home_timeouts")).otherwise(pl.col("away_timeouts")),
            timeouts_def=pl.when(pl.col("is_home")).then(pl.col("away_timeouts")).otherwise(pl.col("home_timeouts")),
            spread_off=pl.when(pl.col("is_home")).then(pl.col("pregame_spread")).otherwise(-pl.col("pregame_spread")),
            drive_start=pl.col("seq") == pl.col("seq").min().over("game_id", "fixed_drive"),
        )
        .select(
            "game_id", "season", "season_type", "week", "seq", "play_id", "fixed_drive", "drive_start",
            "qtr", "game_half", "game_seconds_remaining", "half_seconds_remaining", "posteam",
            "is_home", "down", "ydstogo", "yardline_100", "goal_to_go", "score_diff", "timeouts_off",
            "timeouts_def", "spread_off", "pregame_total", "result", "seconds", "end_gsr",
            "def_points", "next_kind", "next_same_offense", "next_yardline", "next_gsr",
            "gap_seconds", "gap_points_off", "gap_points_def", "drive_start_yardline",
            "drive_start_gsr",
        )
    )
    return drives, snaps


def check_scores(drives: pl.DataFrame) -> pl.DataFrame:
    """Rebuild every final score from the labels alone: points before the first snap, then
    per drive the points its result implies plus the gap after it. Returns the games
    where that does not reproduce the final score (for drives with a clean label)."""
    first = drives.filter(pl.col("n_snaps") > 0).group_by("game_id").agg(
        pl.col("snap_home_pre").sort_by("first_seq").first(),
        pl.col("snap_away_pre").sort_by("first_seq").first(),
    )
    t = add_transitions(label_results(drives)) if "gap_points_off" not in drives.columns else drives
    implied_off = pl.col("result").replace_strict(OFFENSE_POINTS, default=0, return_dtype=pl.Float64)
    implied_def = pl.when(pl.col("result") == "opp_td").then(pl.col("def_points")).otherwise(
        pl.col("result").replace_strict(DEFENSE_POINTS, default=0, return_dtype=pl.Float64))
    home_off = pl.col("offense") == pl.col("home_team")
    per = t.with_columns(
        home=pl.when(home_off).then(implied_off + pl.col("gap_points_off")).otherwise(implied_def + pl.col("gap_points_def")),
        away=pl.when(home_off).then(implied_def + pl.col("gap_points_def")).otherwise(implied_off + pl.col("gap_points_off")),
    ).group_by("game_id").agg(
        pl.col("home").sum(), pl.col("away").sum(), pl.col("final_home").first(),
        pl.col("final_away").first(), bad=pl.col("result").is_null().sum(),
    )
    out = per.join(first, on="game_id").with_columns(
        home=pl.col("home") + pl.col("snap_home_pre"), away=pl.col("away") + pl.col("snap_away_pre"))
    return out.filter((pl.col("home") != pl.col("final_home")) | (pl.col("away") != pl.col("final_away")))


def build_seasons(seasons: Iterable[int], log: Log = print) -> None:
    for s in seasons:
        drives, snaps = build(season_rows(s))
        for df, path in ((drives, drives_path(s)), (snaps, snaps_path(s))):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
        bad = check_scores(drives)
        log(f"drives {s}: {drives.height:,} drives, {snaps.height:,} snaps, "
            f"{drives['result'].null_count()} drives unlabelled, "
            f"{bad.height} games whose labels don't rebuild the final score")


def load_snaps(seasons: Iterable[int]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(snaps_path(s)) for s in seasons], how="diagonal_relaxed")


def load_drives(seasons: Iterable[int]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(drives_path(s)) for s in seasons], how="diagonal_relaxed")
