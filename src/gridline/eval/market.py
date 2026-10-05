"""Kalshi's prices, shaped for the benchmark (DESIGN.md decision 5: no lookahead).

Two views of the market:

  candles  one per contract per minute. A candle is known only when it closes (end_ts),
           so its close is the market's state at end_ts: the mid of the best bid and
           ask (mid_close) and their distance (spread_close).
  trades   every trade, timestamped to the microsecond. Used where seconds matter (the
           lead-lag event study), at the price of the bid-ask bounce.

Everything is put on the home team's side. The winner market is two contracts, one per
team, each with its own order book. A tie settles both at 50 cents, so each is a price
with ties counting half, and "home wins" is either the home contract or one minus the
away contract. The home price is the average of the two mids when both books are
quoted, else whichever is.
"""

from __future__ import annotations

import polars as pl

from gridline.data.kalshi_pull import consolidated_path


def markets(season: int) -> pl.DataFrame:
    """One row per contract: game, ticker, kind (winner/spread/total), side, strike, and
    how it settled (y: 1 yes, 0 no, 0.5 a tie on the winner contracts)."""
    return (pl.read_parquet(consolidated_path("markets", season))
            .select("game_id", "ticker", "kind", "side", "strike", "result")
            .with_columns(y=pl.when(pl.col("result") == "yes").then(1.0)
                          .when(pl.col("result") == "no").then(0.0)
                          .when(pl.col("result") == "scalar").then(0.5)))


def winner_minutes(season: int) -> pl.DataFrame:
    """The home win price at every candle close: game_id, t, market (the home price),
    home_mid, away_mid and spread (the wider of the two books' bid-ask spreads)."""
    m = markets(season).filter(pl.col("kind") == "winner").select("ticker", "side")
    c = (pl.scan_parquet(consolidated_path("candles", season))
         .select("game_id", "ticker", "end_ts", "mid_close", "spread_close").collect()
         .join(m, on="ticker"))
    wide = (c.filter(pl.col("mid_close").is_not_null())
            .pivot(on="side", index=["game_id", "end_ts"], values=["mid_close", "spread_close"])
            .rename({"end_ts": "t"}))
    for col in ("mid_close_home", "mid_close_away", "spread_close_home", "spread_close_away"):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    return (wide.with_columns(
                home_mid=pl.col("mid_close_home"), away_mid=pl.col("mid_close_away"),
                market=pl.mean_horizontal(pl.col("mid_close_home"), 1 - pl.col("mid_close_away")),
                spread=pl.max_horizontal("spread_close_home", "spread_close_away"))
            .select("game_id", "t", "market", "home_mid", "away_mid", "spread")
            .sort("game_id", "t"))


def winner_trades(season: int) -> pl.DataFrame:
    """Every winner-contract trade as a home price: game_id, ts, price, count, side (the
    contract that traded)."""
    m = markets(season).filter(pl.col("kind") == "winner").select("ticker", "side")
    tr = (pl.scan_parquet(consolidated_path("trades", season))
          .select("game_id", "ticker", "ts", "yes_price", "count")
          .join(m.lazy(), on="ticker")
          .with_columns(price=pl.when(pl.col("side") == "home").then(pl.col("yes_price"))
                        .otherwise(1 - pl.col("yes_price")))
          .select("game_id", "ts", "price", "count", "side").collect())
    return tr.sort("game_id", "ts")


def rung_minutes(season: int, max_spread: float = 0.05) -> pl.DataFrame:
    """Spread and total rungs at every candle close where the rung's book is two-sided
    and no wider than max_spread: game_id, t, ticker, kind, side, strike, market (the
    mid), spread, y (how it settled)."""
    m = markets(season).filter(pl.col("kind").is_in(["spread", "total"])).drop("game_id", "result")
    return (pl.scan_parquet(consolidated_path("candles", season))
            .select("game_id", "ticker", "end_ts", "mid_close", "spread_close")
            .filter(pl.col("mid_close").is_not_null() & (pl.col("spread_close") <= max_spread + 1e-9))
            .collect()
            .join(m, on="ticker")
            .rename({"end_ts": "t", "mid_close": "market", "spread_close": "spread"})
            .sort("game_id", "t"))
