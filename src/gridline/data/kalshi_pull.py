"""Pull Kalshi NFL market data for one season into data/raw/kalshi/<season>/.

Must run on a machine that can reach api.elections.kalshi.com (your Mac; the
cloud sandbox is blocked). Resumable: a game folder with a DONE marker is
skipped, so an interrupted pull just continues.

Layout
  data/raw/kalshi/<season>/catalog.parquet             Kalshi event <-> nflverse game_id
  data/raw/kalshi/<season>/games/<game_id>/
      markets.parquet   the two winner markets + every spread/total rung, with kind/side/strike
                        (winner = KXNFLGAME, or the champion market for title games)
      candles.parquet   1-minute candles (winner markets + traded ladder rungs)
      trades.parquet    every trade on the two winner markets
      DONE              or ERROR.txt with the reason; failed games are retried next run
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable

import polars as pl

from gridline import config
from gridline.data import catalog, nflverse
from gridline.data.kalshi import KalshiError, KalshiPublic

Log = Callable[[str], None]

_UTC = pl.Datetime("us", "UTC")
CANDLE_SCHEMA = {
    "game_id": pl.Utf8, "ticker": pl.Utf8, "end_ts": _UTC,
    **{f"{s}_{f}": pl.Float64 for s in ("bid", "ask", "trade")
       for f in ("open", "high", "low", "close")},
    "trade_mean": pl.Float64, "volume": pl.Float64, "open_interest": pl.Float64,
    "mid_close": pl.Float64, "spread_close": pl.Float64,
}
TRADE_SCHEMA = {
    "game_id": pl.Utf8, "trade_id": pl.Utf8, "ticker": pl.Utf8, "ts": _UTC,
    "yes_price": pl.Float64, "count": pl.Float64, "taker_side": pl.Utf8,
    "is_block_trade": pl.Boolean,
}
MARKET_SCHEMA = {
    "game_id": pl.Utf8, "ticker": pl.Utf8, "event_ticker": pl.Utf8, "series_ticker": pl.Utf8,
    "kind": pl.Utf8, "side": pl.Utf8, "strike": pl.Float64,
    "title": pl.Utf8, "yes_sub_title": pl.Utf8, "status": pl.Utf8, "result": pl.Utf8,
    "open_time": _UTC, "close_time": _UTC, "settlement_ts": _UTC, "settlement_value": pl.Float64,
    "strike_type": pl.Utf8, "floor_strike": pl.Float64, "cap_strike": pl.Float64,
    "volume": pl.Float64, "open_interest": pl.Float64, "last_price": pl.Float64,
    "rules_primary": pl.Utf8, "rules_secondary": pl.Utf8,
}


def season_dir(season: int) -> Path:
    return config.KALSHI_DIR / str(season)


def game_dir(season: int, game_id: str) -> Path:
    return season_dir(season) / "games" / game_id


def _frame(rows: list[dict], schema: dict) -> pl.DataFrame:
    return pl.DataFrame([{k: r.get(k) for k in schema} for r in rows], schema=schema)


def build_catalog(season: int, client: KalshiPublic, log: Log = print) -> pl.DataFrame:
    """Match every KXNFLGAME event to this season's nflverse games, and add the
    conference championships and Super Bowl, whose winner trades as a champion
    market instead (catalog.championship_events)."""
    sched = nflverse.load_schedules([season])
    cols = ["game_id", "game_type", "week", "gameday", "gametime", "away_team", "home_team", "result"]
    events = client.events(config.GAME_SERIES)
    cat = catalog.match_events((e["event_ticker"] for e in events), sched)
    matched = cat.filter(pl.col("status") == "matched").join(sched.select(cols), on="game_id",
                                                              how="inner")
    dupes = matched.filter(pl.col("game_id").is_duplicated())
    if dupes.height:
        log(f"WARNING: games matched by more than one event: {dupes['game_id'].unique().to_list()}")
    champ = (sched.filter(pl.col("game_type").is_in(list(catalog.CHAMPIONSHIP_GAME_TYPES)))
             .join(matched.select("game_id"), on="game_id", how="anti")
             .select(cols).with_columns(status=pl.lit("championship")))
    full = pl.concat([matched, champ], how="diagonal_relaxed")
    played = sched.filter(pl.col("result").is_not_null())
    missing = played.join(full, on="game_id", how="anti")
    log(
        f"catalog {season}: {len(events)} KXNFLGAME events listed, {matched.height} matched to "
        f"{season} games, {champ.height} championship games added; {played.height} games played, "
        f"{missing.height} with no Kalshi market"
    )
    if missing.height:
        log("  no market for: " + ", ".join(missing["game_id"].to_list()[:20]))
    path = season_dir(season) / "catalog.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    full.write_parquet(path)
    return full


def _markets(client: KalshiPublic, event_ticker: str) -> list[dict]:
    try:
        return client.markets(event_ticker)
    except KalshiError:
        return []  # an event can be missing for a game; recorded as 0 markets


def _winner_markets(client: KalshiPublic, game: dict, season: int) -> list[dict]:
    home, away = game["home_team"], game["away_team"]
    if game["game_type"] in catalog.CHAMPIONSHIP_GAME_TYPES:
        # champion markets list every team; keep the two playing
        for et in catalog.championship_events(game["game_type"], season):
            ms = [m for m in _markets(client, et) if catalog.market_team(m) in (home, away)]
            if len(ms) == 2:
                return ms
        return []
    return _markets(client, f"{config.GAME_SERIES}-{game['suffix']}")


def _ladder_markets(client: KalshiPublic, game: dict) -> list[dict]:
    suffixes = ([game["suffix"]] if game.get("suffix") else
                catalog.candidate_suffixes(game["gameday"], game["away_team"], game["home_team"]))
    for s in suffixes:
        found = (_markets(client, f"{config.SPREAD_SERIES}-{s}")
                 + _markets(client, f"{config.TOTAL_SERIES}-{s}"))
        if found:
            return found
    return []


def pull_game(
    season: int,
    game: dict,
    client: KalshiPublic,
    trades: bool = True,
    pre_kickoff: dt.timedelta = dt.timedelta(hours=3),
    after_kickoff: dt.timedelta = dt.timedelta(hours=8),
    rung_min_volume: float = 1.0,
) -> dict:
    """Pull one game's markets, candles and (winner-market) trades. Returns counts.

    The data window is fixed relative to kickoff. It is deliberately NOT cut at
    the market's close_time: Kalshi can rewrite that field (the 2025 GB-DAL tie
    shows a close_time 8 hours before kickoff), and nothing trades after close."""
    gid, home, away = game["game_id"], game["home_team"], game["away_team"]
    kick = nflverse.kickoff_utc(game["gameday"], game["gametime"])

    markets: list[dict] = []
    for m in _winner_markets(client, game, season) + _ladder_markets(client, game):
        m.update(catalog.classify_market(m, home, away), game_id=gid)
        markets.append(m)
    winners = [m for m in markets if m["kind"] == "winner"]
    if len(winners) != 2:
        raise KalshiError(f"{gid}: expected 2 winner markets, got {len(winners)}")

    start, end = kick - pre_kickoff, kick + after_kickoff
    wanted = winners + [
        m for m in markets if m["kind"] != "winner" and (m["volume"] or 0) >= rung_min_volume
    ]
    candles = [c | {"game_id": gid} for m in wanted for c in client.candles(m, start, end)]
    trade_rows = []
    if trades:
        for m in winners:
            trade_rows += [t | {"game_id": gid}
                           for t in client.trades(m["ticker"], kick - dt.timedelta(hours=1), end)]

    out = game_dir(season, gid)
    out.mkdir(parents=True, exist_ok=True)
    _frame(markets, MARKET_SCHEMA).write_parquet(out / "markets.parquet")
    _frame(candles, CANDLE_SCHEMA).write_parquet(out / "candles.parquet")
    if trades:
        _frame(trade_rows, TRADE_SCHEMA).write_parquet(out / "trades.parquet")
    (out / "DONE").write_text(dt.datetime.now(dt.timezone.utc).isoformat())
    return {"markets": len(markets), "candle_markets": len(wanted),
            "candles": len(candles), "trades": len(trade_rows)}


def pull_season(
    season: int,
    client: KalshiPublic | None = None,
    trades: bool = True,
    max_games: int | None = None,
    redo: set[str] | frozenset[str] = frozenset(),
    log: Log = print,
) -> list[tuple[str, str]]:
    """Pull every played game not yet DONE (plus any game id in `redo`).
    A failed game gets an ERROR.txt with the reason and is retried next run.
    Returns the (game_id, error) pairs that failed."""
    client = client or KalshiPublic()
    cat = build_catalog(season, client, log)
    todo = cat.filter(pl.col("result").is_not_null()).sort("gameday", "game_id")
    unknown = set(redo) - set(todo["game_id"].to_list())
    if unknown:
        log(f"WARNING: --redo games not in the {season} catalog: {sorted(unknown)}")
    done = skipped = 0
    failures: list[tuple[str, str]] = []
    for game in todo.iter_rows(named=True):
        if max_games is not None and done >= max_games:
            break
        gid = game["game_id"]
        gdir = game_dir(season, gid)
        if (gdir / "DONE").exists() and gid not in redo:
            skipped += 1
            continue
        try:
            n = pull_game(season, game, client, trades=trades)
        except Exception as e:  # one bad game must not stop the season
            msg = f"{type(e).__name__}: {e}"
            failures.append((gid, msg))
            gdir.mkdir(parents=True, exist_ok=True)
            (gdir / "DONE").unlink(missing_ok=True)
            (gdir / "ERROR.txt").write_text(msg + "\n")
            log(f"  FAILED {gid}: {msg}")
            continue
        (gdir / "ERROR.txt").unlink(missing_ok=True)
        done += 1
        log(f"  {gid}: {n['markets']} markets, candles for {n['candle_markets']} "
            f"({n['candles']} rows), {n['trades']} trades  [{client.n_requests} requests]")
    log(f"season {season}: pulled {done}, already had {skipped}, failed {len(failures)}")
    for gid, msg in failures:
        log(f"  failed: {gid}: {msg}")
    return failures


# ------------------------------------------------------------------ one file per table

TABLES = ("markets", "candles", "trades")


def consolidated_path(table: str, season: int) -> Path:
    return config.DERIVED_DIR / "kalshi" / f"{table}_{season}.parquet"


def consolidate(season: int, log: Log = print) -> dict[str, Path]:
    """Write each table of a season's pull as one file (data/derived/kalshi/<table>_<season>.parquet).
    The per-game layout is what gets pulled and re-pulled; these are what analysis reads.
    Streams game by game, so memory stays small."""
    games = sorted(p for p in (season_dir(season) / "games").glob("*") if (p / "DONE").exists())
    drop = {"markets": ["rules_primary", "rules_secondary"], "candles": [], "trades": ["trade_id"]}
    out = {}
    for table in TABLES:
        files = [g / f"{table}.parquet" for g in games if (g / f"{table}.parquet").exists()]
        path = consolidated_path(table, season)
        path.parent.mkdir(parents=True, exist_ok=True)
        scans = [pl.scan_parquet(f).drop(drop[table], strict=False) for f in files]
        pl.concat(scans, how="diagonal_relaxed").sink_parquet(path)
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        log(f"{table}: {len(files)} games, {rows:,} rows -> {path}")
        out[table] = path
    return out
