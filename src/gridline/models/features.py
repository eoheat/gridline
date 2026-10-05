"""Features for the direct win-probability model (Phase 1).

Mirrors nflfastR's prepare_wp_data() exactly: same feature names, same order,
same definitions, computed from our play events (state/events.py) instead of
nflfastR's pbp. Everything is from the possession team's point of view:

  receive_2h_ko        1 in the first half if posteam kicked off to start the game
                       (so it receives the 2nd-half kickoff)
  spread_time          posteam's pregame spread * exp(-4 * elapsed_share)
  home                 1 if posteam is the home team
  half_seconds_remaining, game_seconds_remaining
  Diff_Time_Ratio      score_differential / exp(-4 * elapsed_share)
  score_differential   posteam score - defteam score, BEFORE the snap
  down, ydstogo, yardline_100
  posteam_timeouts_remaining, defteam_timeouts_remaining

Leakage guard: features may only read FEATURE_INPUTS (pre-snap columns). The
label reads final_margin_home and nothing else. tests/test_features.py scrambles
every other column and checks the features don't move.
"""

from __future__ import annotations

import polars as pl

# nflfastR's column order (R/helper_add_ep_wp.R: wp_model_select / wp_spread_model_select)
WP_FEATURES = [
    "receive_2h_ko", "home", "half_seconds_remaining", "game_seconds_remaining",
    "Diff_Time_Ratio", "score_differential", "down", "ydstogo", "yardline_100",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
]
WP_SPREAD_FEATURES = [
    "receive_2h_ko", "spread_time", "home", "half_seconds_remaining", "game_seconds_remaining",
    "Diff_Time_Ratio", "score_differential", "down", "ydstogo", "yardline_100",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
]

# the only event columns feature code may read (all known at the snap)
FEATURE_INPUTS = frozenset({
    "game_id", "season", "seq", "qtr", "game_seconds_remaining", "half_seconds_remaining",
    "posteam", "home_team", "away_team", "down", "ydstogo", "yardline_100",
    "home_timeouts", "away_timeouts", "home_score_pre", "away_score_pre", "pregame_spread",
})
LABEL_INPUTS = frozenset({"final_margin_home"})

MISSING_SPREAD = 1.5  # what nflfastR substitutes (for the home team) when spread_line is missing


def wp_features(events: pl.DataFrame) -> pl.DataFrame:
    """Rows with a possession team -> keys + nflfastR WP features + `label`
    (1 if posteam went on to win, 0 if it lost, null for a tie)."""
    e = events.select(sorted(FEATURE_INPUTS | LABEL_INPUTS)).sort("game_id", "seq")
    g = "game_id"
    # nflfastR: posteam == first(na.omit(defteam)) -> the team that kicked off to start
    first_pos = pl.col("posteam").drop_nulls().first().over(g)
    first_def = (pl.when(first_pos == pl.col("home_team"))
                 .then(pl.col("away_team")).otherwise(pl.col("home_team")))
    is_home = pl.col("posteam") == pl.col("home_team")
    elapsed_share = (3600 - pl.col("game_seconds_remaining")) / 3600
    decay = (-4 * elapsed_share).exp()
    spread = pl.col("pregame_spread").fill_null(MISSING_SPREAD)
    return (
        e.with_columns(receive_2h_ko=((pl.col("qtr") <= 2) & (pl.col("posteam") == first_def))
                       .cast(pl.Int8))
        .filter(pl.col("posteam").is_not_null())
        .with_columns(
            home=is_home.cast(pl.Int8),
            score_differential=pl.when(is_home)
            .then(pl.col("home_score_pre") - pl.col("away_score_pre"))
            .otherwise(pl.col("away_score_pre") - pl.col("home_score_pre")),
            posteam_timeouts_remaining=pl.when(is_home).then("home_timeouts").otherwise("away_timeouts"),
            defteam_timeouts_remaining=pl.when(is_home).then("away_timeouts").otherwise("home_timeouts"),
            posteam_spread=pl.when(is_home).then(spread).otherwise(-spread),
            decay=decay,
        )
        .with_columns(
            spread_time=pl.col("posteam_spread") * pl.col("decay"),
            Diff_Time_Ratio=pl.col("score_differential") / pl.col("decay"),
            label=pl.when(pl.col("final_margin_home") == 0).then(None)
            .when(is_home).then((pl.col("final_margin_home") > 0).cast(pl.Int8))
            .otherwise((pl.col("final_margin_home") < 0).cast(pl.Int8)),
        )
        .select("game_id", "season", "seq", "qtr", "posteam", "home_team",
                *WP_SPREAD_FEATURES, "label")
    )
