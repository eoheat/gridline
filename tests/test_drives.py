"""Drive labels on a hand-built game: a touchdown, a punt, a field goal answered by a
kickoff returned for a touchdown (a drive with no snap), and two drives that run out
the halves."""

import polars as pl
import pytest

from gridline.models import drives as dr

# (fixed_drive, posteam, down, yardline, half, clock, home_pre, away_pre, home_post, away_post, result)
PLAYS = [
    (1, "H", None, 35, "Half1", 3600, 0, 0, 0, 0, "Touchdown"),   # kickoff (not a snap)
    (1, "H", 1, 75, "Half1", 3600, 0, 0, 0, 0, "Touchdown"),
    (1, "H", 2, 70, "Half1", 3560, 0, 0, 0, 0, "Touchdown"),
    (1, "H", 1, 30, "Half1", 3520, 0, 0, 6, 0, "Touchdown"),       # the touchdown
    (1, "H", None, 15, "Half1", 3500, 6, 0, 7, 0, "Touchdown"),    # extra point
    (2, "A", None, 35, "Half1", 3500, 7, 0, 7, 0, "Punt"),
    (2, "A", 1, 75, "Half1", 3495, 7, 0, 7, 0, "Punt"),
    (2, "A", 4, 71, "Half1", 3450, 7, 0, 7, 0, "Punt"),
    (3, "H", 1, 60, "Half1", 3440, 7, 0, 7, 0, "Field goal"),
    (3, "H", 4, 20, "Half1", 3300, 7, 0, 10, 0, "Field goal"),
    (4, "A", None, 35, "Half1", 3295, 10, 0, 10, 6, "Touchdown"),  # kick return TD, no snap
    (4, "A", None, 15, "Half1", 3290, 10, 6, 10, 7, "Touchdown"),
    (5, "H", None, 35, "Half1", 3290, 10, 7, 10, 7, "End of half"),
    (5, "H", 1, 75, "Half1", 3285, 10, 7, 10, 7, "End of half"),
    (5, "H", 1, 75, "Half1", 1805, 10, 7, 10, 7, "End of half"),
    (6, "A", None, 35, "Half2", 1800, 10, 7, 10, 7, "End of half"),
    (6, "A", 1, 75, "Half2", 1795, 10, 7, 10, 7, "End of half"),
]


def rows() -> pl.DataFrame:
    df = pl.DataFrame(PLAYS, orient="row", schema=[
        "fixed_drive", "posteam", "down", "yardline_100", "game_half", "game_seconds_remaining",
        "home_score_pre", "away_score_pre", "home_score_post", "away_score_post", "fixed_drive_result"])
    n = df.height
    return df.with_columns(
        game_id=pl.lit("G"), season=pl.lit(2024), season_type=pl.lit("REG"), week=pl.lit(1),
        home_team=pl.lit("H"), away_team=pl.lit("A"), seq=pl.int_range(n), play_id=pl.int_range(n) * 10.0,
        qtr=pl.when(pl.col("game_half") == "Half1").then(1).otherwise(3),
        half_seconds_remaining=pl.when(pl.col("game_half") == "Half1")
        .then(pl.col("game_seconds_remaining") - 1800).otherwise(pl.col("game_seconds_remaining")),
        ydstogo=pl.lit(10), goal_to_go=pl.lit(False), home_timeouts=pl.lit(3), away_timeouts=pl.lit(3),
        pregame_spread=pl.lit(2.5), pregame_total=pl.lit(44.5),
        final_margin_home=pl.lit(3), final_total=pl.lit(17),
    ).with_columns(pl.col("game_seconds_remaining").cast(pl.Float64))


def test_results_clock_and_transitions():
    drives, snaps = dr.build(rows())
    by = {r["fixed_drive"]: r for r in drives.iter_rows(named=True)}
    assert [by[i]["result"] for i in (1, 2, 3, 5, 6)] == ["td1", "punt", "fg", "end_half", "end_half"]
    assert 4 not in by  # the return touchdown has no snap: it is part of a gap
    # the field goal is answered by the return TD, then the same offence (H) has the ball
    fg = by[3]
    assert fg["next_same_offense"] and fg["gap_points_def"] == 7 and fg["gap_points_off"] == 0
    assert fg["gap_seconds"] == 3295 - 3285 and fg["next_yardline"] == 75
    assert by[2]["next_yardline"] == 60 and not by[2]["next_same_offense"]
    assert by[5]["next_kind"] == "next_half" and by[5]["end_gsr"] == 1800
    assert by[6]["next_kind"] == "game_end"
    # seconds from each snap to its drive's end
    td = snaps.filter(pl.col("fixed_drive") == 1).sort("seq")
    assert td["seconds"].to_list() == [100, 60, 20] and td["drive_start"].to_list() == [True, False, False]
    assert snaps.filter(pl.col("fixed_drive") == 5)["seconds"].to_list() == [1485, 5]
    a = snaps.filter(pl.col("posteam") == "A").row(0, named=True)
    assert a["spread_off"] == pytest.approx(-2.5) and a["score_diff"] == -7


def test_labels_rebuild_the_final_score():
    drives, _ = dr.build(rows())
    assert dr.check_scores(drives).height == 0
    broken = rows().with_columns(  # a field goal booked as a punt no longer adds up
        fixed_drive_result=pl.when(pl.col("fixed_drive") == 3).then(pl.lit("Punt"))
        .otherwise(pl.col("fixed_drive_result")))
    drives, _ = dr.build(broken)
    assert drives.filter(pl.col("fixed_drive") == 3)["result"][0] is None
    assert dr.check_scores(drives).height == 1
