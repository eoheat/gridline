import datetime as dt

import numpy as np
import polars as pl
import pytest

from gridline import inventory
from gridline.state.events import (
    PBP_COLUMNS, build_play_events, clean_game_times, with_known_times,
)
from gridline.state.game_state import GameState

UTC = dt.timezone.utc
T0 = dt.datetime(2025, 9, 7, 17, 0, 0, tzinfo=UTC)


def iso(sec: float | None, ms: bool = True) -> str | None:
    if sec is None:
        return None
    t = T0 + dt.timedelta(seconds=sec)
    return t.strftime("%Y-%m-%dT%H:%M:%S") + (f".{t.microsecond // 1000:03d}Z" if ms else "Z")


def row(play_id, play_type, snap, end, home_total=0, away_total=0, posteam="WAS", **kw):
    base = {c: None for c in PBP_COLUMNS}
    base.update(
        game_id="2025_01_NYG_WAS", play_id=float(play_id), season=2025, week=1,
        season_type="REG", home_team="WAS", away_team="NYG", play_type=play_type,
        desc=play_type or "admin", time_of_day=snap, end_clock_time=end,
        qtr=1.0, game_seconds_remaining=3600.0, half_seconds_remaining=1800.0,
        game_half="Half1", posteam=posteam, down=1.0, ydstogo=10.0, yardline_100=75.0,
        goal_to_go=0.0, home_timeouts_remaining=3.0, away_timeouts_remaining=3.0,
        total_home_score=float(home_total), total_away_score=float(away_total),
        spread_line=6.5, total_line=45.5, home_opening_kickoff=1.0, result=15.0, total=27.0,
    )
    base.update(kw)
    return base


@pytest.fixture
def game() -> pl.DataFrame:
    rows = [
        row(1, None, None, None, posteam=None),         # GAME admin row: no time
        row(40, "kickoff", iso(100), iso(106)),
        row(63, "pass", iso(150), iso(155)),
        row(88, "run", iso(190), None),                  # end missing -> imputed
        row(99, "no_play", iso(170, ms=False), None),    # timeout logged out of order
        row(110, "pass", iso(230 + 3 * 3600), iso(236 + 3 * 3600)),  # wild: 3 h off
        row(130, "pass", iso(270), iso(275), home_total=6),          # TD
        # timeout from *before* the TD, logged after it, still showing the old score
        row(135, "no_play", iso(265, ms=False), None, home_total=0),
        row(140, "extra_point", iso(300), iso(302), home_total=7),
        row(150, "kickoff", iso(330), iso(336), home_total=7, posteam="NYG"),
        row(160, None, None, None, home_total=7, posteam=None),      # END QUARTER
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def secs(ts: dt.datetime | None) -> float | None:
    return None if ts is None else (ts - T0).total_seconds()


def test_timestamps_are_cleaned_conservatively(game):
    ev = build_play_events(game)
    by = {int(r["play_id"]): r for r in ev.iter_rows(named=True)}
    snap = {k: secs(r["t_snap"]) for k, r in by.items()}
    end = {k: secs(r["t_end"]) for k, r in by.items()}
    assert snap[1] == 100                   # leading admin row: first valid snap
    assert by[99]["snap_out_of_order"]      # the timeout...
    assert snap[99] == snap[88] == 190      # ...is placed no earlier than the row before it
    assert by[110]["snap_wild"] and by[110]["snap_imputed"]
    assert snap[110] == end[99]             # wild snap replaced by previous row's end
    assert by[88]["end_imputed"] and end[88] > snap[88]
    assert snap[160] == end[150]            # trailing admin row: right after the last play
    s, e = list(snap.values()), list(end.values())
    assert all(np.diff(s) >= 0) and all(np.diff(e) >= 0)
    assert all(b >= a for a, b in zip(s, e))


def test_scores_pre_and_post_never_leak(game):
    ev = build_play_events(game)
    td = ev.filter(pl.col("play_id") == 130).row(0, named=True)
    xp = ev.filter(pl.col("play_id") == 140).row(0, named=True)
    assert (td["home_score_pre"], td["home_score_post"]) == (0, 6)
    assert (xp["home_score_pre"], xp["home_score_post"]) == (6, 7)
    stale = ev.filter(pl.col("play_id") == 135).row(0, named=True)
    assert stale["home_score_post"] == 6  # a stale timeout row can't un-score the TD
    # the raw post-play columns must not survive into events under their old names
    assert "total_home_score" not in ev.columns and "result" not in ev.columns
    assert ev["final_margin_home"][0] == 15


def test_known_times_add_delay_and_stay_monotone(game):
    ev = with_known_times(build_play_events(game), delta_s=10)
    k = [secs(t) for t in ev["t_known"]]
    e = [secs(t) for t in ev["t_end"]]
    assert k[1] == e[1] + 10
    assert all(np.diff(k) >= 0)


def test_clean_game_times_with_no_timestamps_at_all():
    n = 3
    out = clean_game_times(np.full(n, np.nan), np.full(n, np.nan), np.full(n, 5.0))
    assert np.isnan(out["t_snap"]).all() and out["snap_imputed"].all()


def test_game_state_from_event(game):
    ev = build_play_events(game)
    s = GameState.pre_snap(ev.filter(pl.col("play_id") == 140).row(0, named=True))
    assert (s.home_score, s.away_score, s.score_diff_home) == (6, 0, 6)
    assert s.home_receives_2h_kickoff is False  # home took the opening kickoff
    assert s.posteam_is_home and not s.overtime
    assert s.pregame_spread == 6.5


# --------------------------------------------------------------- real data (if cached)

@pytest.fixture(scope="module")
def real_2025():
    try:
        return inventory.load_events(2025)
    except FileNotFoundError:
        pytest.skip("no cached 2025 events (run `gridline events build --seasons 2025`)")


def test_real_events_are_monotone_and_consistent(real_2025):
    ev = real_2025
    bad = ev.group_by("game_id").agg(
        (pl.col("t_snap").diff() < pl.duration(seconds=0)).any().alias("snap_backwards"),
        (pl.col("t_end") < pl.col("t_snap")).any().alias("end_before_snap"),
        (pl.col("home_score_post").diff() < 0).any().alias("home_score_down"),
        (pl.col("away_score_post").diff() < 0).any().alias("away_score_down"),
    ).filter(pl.any_horizontal("snap_backwards", "end_before_snap",
                               "home_score_down", "away_score_down"))
    assert bad.height == 0, bad
    # the final post score equals the official result in every game
    last = ev.group_by("game_id").agg(pl.all().sort_by("seq").last())
    assert (last["home_score_post"] - last["away_score_post"] == last["final_margin_home"]).all()


def test_real_pre_scores_match_nflverse_pre_play_scores(real_2025):
    """Independent check: nflverse's own pre-play posteam/defteam scores."""
    from gridline.data import nflverse

    pbp = nflverse.load_pbp([2025], columns=["game_id", "play_id", "posteam", "home_team",
                                             "play_type", "posteam_score", "defteam_score"])
    j = real_2025.join(pbp, on=["game_id", "play_id"]).filter(
        pl.col("posteam").is_not_null() & pl.col("play_type").is_not_null()
    )
    home_pre = pl.when(pl.col("posteam_is_home")).then("posteam_score").otherwise("defteam_score")
    away_pre = pl.when(pl.col("posteam_is_home")).then("defteam_score").otherwise("posteam_score")
    mismatches = j.filter((pl.col("home_score_pre") != home_pre)
                          | (pl.col("away_score_pre") != away_pre))
    assert j.height > 40_000 and mismatches.height == 0, mismatches.head()
