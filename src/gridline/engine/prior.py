"""The prior: where model and market both start (DESIGN.md decision 3).

The headline prior is Kalshi's own pregame price, read from the last one-minute candle
that closed at or before kickoff:

  win    the home team's winner contract, as the average of its mid and one minus the
         away contract's mid (the two are separate order books; ties settle 50/50 in
         both, so either mid is a price with ties counting half)
  total  the median total read off the total ladder: the strike where P(over) crosses
         50%, interpolated between the two rungs that straddle it. Only rungs with a
         two-sided quote no wider than MAX_TOTAL_SPREAD, closed within MAX_STALE_S of
         kickoff, are used. Without two such rungs straddling 50%, the sportsbook
         closing total from nflverse is used instead (the two agree to within half a
         point in 2025, SD).

The engine sees no other market price: everything after kickoff is joined only at
evaluation (eval/market.py).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from gridline.data.kalshi_pull import consolidated_path
from gridline.models.simulator import ladder_median

MAX_STALE_S = 30 * 60
MAX_TOTAL_SPREAD = 0.10


@dataclass(frozen=True)
class Prior:
    game_id: str
    home_win: float  # fair YES price of the home winner contract (ties count half)
    total: float  # the total with a 50% chance of being exceeded
    win_source: str  # "kalshi" or "line" (no usable winner quote: the sportsbook spread)
    total_source: str  # "kalshi" or "line"
    quoted_at: dt.datetime | None  # close of the winner candle used
    home_spread: float | None = None  # win_source "line": the spread to match instead


def _last_quotes(candles: pl.DataFrame, kickoffs: pl.DataFrame) -> pl.DataFrame:
    """Each ticker's last candle with a two-sided quote closed at or before kickoff."""
    return (candles.join(kickoffs, on="game_id")
            .filter((pl.col("end_ts") <= pl.col("kickoff")) & pl.col("mid_close").is_not_null()
                    & (pl.col("end_ts") >= pl.col("kickoff") - pl.duration(seconds=MAX_STALE_S)))
            .sort("end_ts").group_by("ticker").last())


def kalshi_priors(season: int, kickoffs: dict[str, dt.datetime],
                  lines: dict[str, tuple[float | None, float | None]]) -> dict[str, Prior]:
    """Priors for every game in `kickoffs` (game_id -> kickoff time). `lines` holds the
    sportsbook (spread, total) per game, the fallback."""
    ko = pl.DataFrame({"game_id": list(kickoffs), "kickoff": list(kickoffs.values())},
                      schema={"game_id": pl.String, "kickoff": pl.Datetime("us", "UTC")})
    markets = pl.read_parquet(consolidated_path("markets", season)).select("ticker", "kind", "side", "strike")
    candles = (pl.scan_parquet(consolidated_path("candles", season))
               .filter(pl.col("game_id").is_in(ko["game_id"].implode()))
               .select("game_id", "ticker", "end_ts", "mid_close", "spread_close").collect())
    last = _last_quotes(candles, ko).join(markets, on="ticker")
    out = {}
    for gid in kickoffs:
        spread_line, total_line = lines.get(gid, (None, None))
        g = last.filter(pl.col("game_id") == gid)
        w = {r["side"]: r for r in g.filter(pl.col("kind") == "winner").iter_rows(named=True)}
        if "home" in w and "away" in w:
            home_win = (w["home"]["mid_close"] + 1 - w["away"]["mid_close"]) / 2
            win_source, quoted = "kalshi", min(w["home"]["end_ts"], w["away"]["end_ts"])
        else:
            home_win, win_source, quoted = float("nan"), "line", None
        t = g.filter((pl.col("kind") == "total") & (pl.col("spread_close") <= MAX_TOTAL_SPREAD + 1e-9))
        total = ladder_median(t["strike"].to_numpy(), t["mid_close"].to_numpy()) if t.height >= 2 else np.nan
        total_source = "kalshi"
        if not np.isfinite(total):
            total, total_source = (float(total_line) if total_line is not None else 44.0), "line"
        out[gid] = Prior(gid, float(home_win), float(total), win_source, total_source, quoted,
                         home_spread=spread_line)
    return out
