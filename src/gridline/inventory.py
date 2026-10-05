"""Phase 0 inventory: what data do we actually have, and is it enough?

Exit gate for Phase 0 (per season): at least 95% of played games have
  - play events whose snaps are mostly raw timestamps (>= 90% of plays), and
  - a Kalshi pull with both winner markets and a quote in at least 80% of the
    in-game minutes. A quote on either side counts: in a blowout the book is
    one-sided at the extremes (a 99c bid with no ask, or a 1c ask with no bid),
    which says "nearly certain", not "no data". Two-sided coverage is reported
    alongside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import polars as pl

from gridline import config
from gridline.data import kalshi_pull, nflverse
from gridline.state.events import build_play_events, timing_summary

Log = Callable[[str], None]

GATE_GAME_SHARE = 0.95
GATE_SNAP_SHARE = 0.90
GATE_QUOTE_SHARE = 0.80


def events_path(season: int) -> Path:
    return config.DERIVED_DIR / "events" / f"events_{season}.parquet"


def build_events(seasons: Sequence[int], log: Log = print) -> None:
    for season in seasons:
        ev = build_play_events(nflverse.load_pbp([season]))
        path = events_path(season)
        path.parent.mkdir(parents=True, exist_ok=True)
        ev.write_parquet(path)
        log(f"events {season}: {ev.height:,} rows, {ev['game_id'].n_unique()} games -> {path}")


def load_events(season: int) -> pl.DataFrame:
    if not events_path(season).exists():
        raise FileNotFoundError(f"no events for {season}; run `gridline events build`")
    return pl.read_parquet(events_path(season))


def per_game_timing(events: pl.DataFrame) -> pl.DataFrame:
    plays = events.filter(pl.col("play_type").is_not_null())
    return plays.group_by("game_id").agg(
        snap_raw=(~pl.col("snap_imputed")).mean(),
        first_snap=pl.col("t_snap").min(),
        last_end=pl.col("t_end").max(),
    )


def kalshi_coverage(season: int, timing: pl.DataFrame) -> pl.DataFrame:
    """One row per pulled game: market counts and in-game quote coverage."""
    rows = []
    games_root = kalshi_pull.season_dir(season) / "games"
    tmap = {r["game_id"]: r for r in timing.iter_rows(named=True)}
    for gdir in sorted(games_root.glob("*")) if games_root.exists() else []:
        if not (gdir / "DONE").exists():
            continue
        gid = gdir.name
        mk = pl.read_parquet(gdir / "markets.parquet")
        cd = pl.read_parquet(gdir / "candles.parquet")
        n_trades = (pl.scan_parquet(gdir / "trades.parquet").select(pl.len()).collect().item()
                    if (gdir / "trades.parquet").exists() else 0)
        home_ml = mk.filter((pl.col("kind") == "winner") & (pl.col("side") == "home"))
        quote_share = two_sided_share = None
        t = tmap.get(gid)
        if home_ml.height == 1 and t and t["first_snap"] is not None:
            live = cd.filter(
                (pl.col("ticker") == home_ml["ticker"][0])
                & (pl.col("end_ts") > t["first_snap"]) & (pl.col("end_ts") <= t["last_end"])
            )
            minutes = max(1.0, (t["last_end"] - t["first_snap"]).total_seconds() / 60)
            quoted = live.filter((pl.col("bid_close") > 0) | (pl.col("ask_close") < 1)).height
            quote_share = min(1.0, quoted / minutes)
            two_sided_share = min(1.0, live["mid_close"].is_not_null().sum() / minutes)
        rows.append({
            "game_id": gid,
            "winner_markets": mk.filter(pl.col("kind") == "winner").height,
            "spread_rungs": mk.filter(pl.col("kind") == "spread").height,
            "total_rungs": mk.filter(pl.col("kind") == "total").height,
            "traded_rungs": mk.filter((pl.col("kind") != "winner") & (pl.col("volume") > 0)).height,
            "in_game_quote_share": quote_share,
            "two_sided_share": two_sided_share,
            "trades": n_trades,
        })
    schema = {"game_id": pl.Utf8, "winner_markets": pl.Int64, "spread_rungs": pl.Int64,
              "total_rungs": pl.Int64, "traded_rungs": pl.Int64,
              "in_game_quote_share": pl.Float64, "two_sided_share": pl.Float64,
              "trades": pl.Int64}
    return pl.DataFrame(rows, schema=schema)


def phase0_report(season: int, log: Log = print) -> bool:
    sched = nflverse.load_schedules([season]).filter(pl.col("result").is_not_null())
    played = sched["game_id"].to_list()
    events = load_events(season)
    timing = per_game_timing(events)

    log(f"== Phase 0 inventory, season {season}: {len(played)} games played ==")
    log(str(timing_summary(events)))

    timed_ok = timing.filter(pl.col("snap_raw") >= GATE_SNAP_SHARE)["game_id"].to_list()
    share_timed = len(set(timed_ok) & set(played)) / max(1, len(played))
    log(f"nflverse: {share_timed:.1%} of played games have >= {GATE_SNAP_SHARE:.0%} "
        f"raw snap timestamps")

    cov = kalshi_coverage(season, timing)
    if cov.height == 0:
        log("kalshi: nothing pulled yet for this season (run `gridline kalshi pull` on your Mac)")
        share_quoted = 0.0
    else:
        ok = cov.filter((pl.col("winner_markets") == 2)
                        & (pl.col("in_game_quote_share") >= GATE_QUOTE_SHARE))
        share_quoted = len(set(ok["game_id"].to_list()) & set(played)) / max(1, len(played))
        log(f"kalshi: {cov.height} games pulled; {share_quoted:.1%} of played games have both "
            f"winner markets and a quote in >= {GATE_QUOTE_SHARE:.0%} of in-game minutes")
        log(str(cov.select(
            pl.col("in_game_quote_share").median().alias("median_quote_share"),
            pl.col("two_sided_share").median().alias("median_two_sided_share"),
            pl.col("traded_rungs").median().alias("median_traded_rungs"),
            pl.col("trades").median().alias("median_trades"),
        )))
        thin = cov.filter(pl.col("in_game_quote_share").fill_null(0) < GATE_QUOTE_SHARE)
        if thin.height:
            log(f"  under {GATE_QUOTE_SHARE:.0%} quoted: {thin['game_id'].to_list()}")
        missing = sorted(set(played) - set(cov["game_id"].to_list()))
        if missing:
            log(f"  not pulled: {len(missing)} games: {missing}")
            errors = [(g, kalshi_pull.game_dir(season, g) / "ERROR.txt") for g in missing]
            for g, path in errors:
                if path.exists():
                    log(f"    {g}: {path.read_text().strip()}")

    passed = share_timed >= GATE_GAME_SHARE and share_quoted >= GATE_GAME_SHARE
    log(f"Phase 0 gate for {season}: {'PASS' if passed else 'NOT YET'}")
    return passed
