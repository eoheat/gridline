"""WP features: nflfastR's definitions, and a leakage guard that scrambles every
column the features are not allowed to read."""

import math

import numpy as np
import polars as pl
import pytest

from gridline.models.features import (
    FEATURE_INPUTS, LABEL_INPUTS, WP_FEATURES, WP_SPREAD_FEATURES, wp_features,
)


def events(**overrides) -> pl.DataFrame:
    """A 4-row game: BUF (home) receives the opening kickoff, NO answers."""
    base = {
        "game_id": ["G"] * 4, "season": [2025] * 4, "seq": [0, 1, 2, 3], "play_id": [1., 2., 3., 4.],
        "qtr": [1, 1, 3, 4], "game_seconds_remaining": [3600., 3000., 1500., 60.],
        "half_seconds_remaining": [1800., 1200., 1500., 60.],
        "posteam": [None, "BUF", "NO", "BUF"], "home_team": ["BUF"] * 4, "away_team": ["NO"] * 4,
        "down": [None, 1., 3., 2.], "ydstogo": [None, 10., 4., 7.], "yardline_100": [None, 75., 40., 20.],
        "home_timeouts": [3, 3, 2, 1], "away_timeouts": [3, 3, 3, 0],
        "home_score_pre": [0, 0, 7, 17], "away_score_pre": [0, 0, 3, 20],
        "pregame_spread": [6.5] * 4, "final_margin_home": [4] * 4,
        # post-play and timing columns the features must never read
        "home_score_post": [0, 7, 7, 23], "away_score_post": [0, 0, 3, 20],
        "t_snap": [0., 1., 2., 3.], "t_end": [1., 2., 3., 4.], "play_type": ["kickoff", "pass", "run", "pass"],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_column_order_matches_nflfastR():
    assert WP_SPREAD_FEATURES[:1] + WP_SPREAD_FEATURES[2:] == WP_FEATURES
    assert WP_SPREAD_FEATURES[1] == "spread_time"


def test_feature_values():
    f = wp_features(events())
    assert f.height == 3  # the row with no possession team is dropped
    buf1, no3, buf4 = f.rows(named=True)
    # BUF took the opening kickoff, so NO kicked and NO receives the 2nd-half kickoff
    assert buf1["receive_2h_ko"] == 0 and buf1["home"] == 1
    assert no3["receive_2h_ko"] == 0  # 3rd quarter: always 0
    assert no3["home"] == 0 and no3["score_differential"] == 3 - 7
    assert no3["posteam_timeouts_remaining"] == 3 and no3["defteam_timeouts_remaining"] == 2
    share = (3600 - 1500) / 3600
    assert no3["spread_time"] == pytest.approx(-6.5 * math.exp(-4 * share))
    assert no3["Diff_Time_Ratio"] == pytest.approx(-4 / math.exp(-4 * share))
    assert buf4["score_differential"] == -3
    # labels from the posteam's side: BUF won by 4
    assert (buf1["label"], no3["label"], buf4["label"]) == (1, 0, 1)


def test_first_half_receiver_flag():
    e = events(posteam=[None, "NO", "BUF", "BUF"], qtr=[1, 1, 2, 3])
    f = wp_features(e)
    # NO received the opening kickoff -> BUF (the kicker) gets the 2nd-half kickoff
    assert f["receive_2h_ko"].to_list() == [0, 1, 0]


def test_ties_have_no_label_and_missing_spread_defaults_like_nflfastR():
    f = wp_features(events(final_margin_home=[0] * 4, pregame_spread=[None] * 4))
    assert f["label"].null_count() == f.height
    first = f.row(0, named=True)  # BUF is home: spread defaults to +1.5 for the home team
    assert first["spread_time"] == pytest.approx(1.5 * math.exp(-4 * (600 / 3600)))


def test_leakage_guard_features_ignore_everything_else():
    e = events()
    rng = np.random.default_rng(0)
    scrambled = e.with_columns(
        pl.Series(c, rng.permutation(e[c].to_numpy())) for c in e.columns
        if c not in FEATURE_INPUTS | LABEL_INPUTS
    )
    a, b = wp_features(e), wp_features(scrambled)
    assert a.select(WP_SPREAD_FEATURES).equals(b.select(WP_SPREAD_FEATURES))


def test_label_is_the_only_thing_that_reads_the_outcome():
    a = wp_features(events())
    b = wp_features(events(final_margin_home=[-10] * 4))
    assert a.select(WP_SPREAD_FEATURES).equals(b.select(WP_SPREAD_FEATURES))
    assert a["label"].to_list() != b["label"].to_list()
