"""End-to-end pull into a temp data dir with a fake Kalshi client built from the
real response shapes: catalog -> markets -> candles -> trades -> files -> coverage.
This is the code path that runs on the Mac, where it can't be debugged from here."""

import datetime as dt

import polars as pl
import pytest

from gridline import config, inventory
from gridline.data import kalshi_pull
from gridline.data.kalshi import KalshiError, normalize_candle, normalize_market, normalize_trade

UTC = dt.timezone.utc
KICK = dt.datetime(2025, 9, 7, 17, 0, tzinfo=UTC)  # 13:00 ET


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    for name, sub in [("DATA_DIR", ""), ("RAW_DIR", "raw"), ("DERIVED_DIR", "derived"),
                      ("NFLVERSE_DIR", "raw/nflverse"), ("KALSHI_DIR", "raw/kalshi")]:
        monkeypatch.setattr(config, name, tmp_path / sub)
    sched = pl.DataFrame({
        "game_id": ["2025_01_NYG_WAS", "2025_18_NYG_WAS"], "season": [2025, 2025],
        "game_type": ["REG", "REG"], "week": [1, 18],
        "gameday": ["2025-09-07", "2026-01-04"], "gametime": ["13:00", "13:00"],
        "away_team": ["NYG", "NYG"], "home_team": ["WAS", "WAS"],
        "result": [15, None],  # week 18 not played yet -> not pulled
    })
    (tmp_path / "raw/nflverse").mkdir(parents=True)
    sched.write_parquet(tmp_path / "raw/nflverse/schedules.parquet")
    return tmp_path


class FakeKalshi:
    def __init__(self, shapes):
        self.s, self.n_requests, self.calls = shapes, 0, []

    def events(self, series):
        return [{"event_ticker": "KXNFLGAME-25SEP07NYGWAS"},
                {"event_ticker": "KXNFLGAME-24SEP15NYGWAS"}]  # a different season

    def markets(self, event_ticker):
        self.calls.append(("markets", event_ticker))
        s = self.s
        if event_ticker.startswith("KXNFLGAME-"):
            # Kalshi rewrote close_time on the 2025 GB-DAL tie to before kickoff; the pull
            # must not use it to cut the data window
            was = dict(s["winner_market"], close_time="2025-09-07T09:00:00Z")
            nyg = dict(was, ticker="KXNFLGAME-25SEP07NYGWAS-NYG", yes_sub_title="New York G",
                       result="no", settlement_value_dollars="0.0000")
            return [normalize_market(was), normalize_market(nyg)]
        if event_ticker.startswith("KXNFLSPREAD-"):
            traded = dict(s["spread_market"], ticker="KXNFLSPREAD-25SEP07NYGWAS-WAS3",
                          floor_strike=3.5, volume_fp="1200.00")
            return [normalize_market(s["spread_market"]), normalize_market(traded)]
        return [normalize_market(s["total_market"])]

    def candles(self, market, start, end):
        self.calls.append(("candles", market["ticker"]))
        assert start == KICK - dt.timedelta(hours=3)
        assert end == KICK + dt.timedelta(hours=8)  # fixed window, not close_time
        out = []
        for k in (1, 2, 3):
            raw = dict(self.s["historical_candle"],
                       end_period_ts=int((KICK + dt.timedelta(minutes=k)).timestamp()))
            out.append(normalize_candle(raw, market["ticker"]))
        return out

    def trades(self, ticker, start, end):
        self.calls.append(("trades", ticker))
        return [normalize_trade(dict(self.s["historical_trade"], ticker=ticker))]


def test_pull_season_end_to_end(data_dir, shapes):
    fake = FakeKalshi(shapes)
    kalshi_pull.pull_season(2025, client=fake, log=lambda m: None)

    gdir = kalshi_pull.game_dir(2025, "2025_01_NYG_WAS")
    assert (gdir / "DONE").exists()
    assert not kalshi_pull.game_dir(2025, "2025_18_NYG_WAS").exists()  # unplayed

    cat = pl.read_parquet(kalshi_pull.season_dir(2025) / "catalog.parquet")
    assert cat["game_id"].to_list() == ["2025_01_NYG_WAS"]  # the 2024 event didn't match

    mk = pl.read_parquet(gdir / "markets.parquet")
    kinds = {(r["ticker"], r["kind"], r["side"], r["strike"]) for r in mk.iter_rows(named=True)}
    assert ("KXNFLGAME-25SEP07NYGWAS-WAS", "winner", "home", None) in kinds
    assert ("KXNFLGAME-25SEP07NYGWAS-NYG", "winner", "away", None) in kinds
    assert ("KXNFLSPREAD-25SEP07NYGWAS-WAS3", "spread", "home", 3.5) in kinds
    assert ("KXNFLTOTAL-25SEP07NYGWAS-45", "total", None, 45.5) in kinds

    candle_tickers = {t for kind, t in fake.calls if kind == "candles"}
    assert "KXNFLSPREAD-25SEP07NYGWAS-WAS49" not in candle_tickers  # never traded: skipped
    assert "KXNFLSPREAD-25SEP07NYGWAS-WAS3" in candle_tickers
    cd = pl.read_parquet(gdir / "candles.parquet")
    assert cd.schema["end_ts"] == pl.Datetime("us", "UTC")
    assert cd.height == 3 * len(candle_tickers)
    tr = pl.read_parquet(gdir / "trades.parquet")
    assert set(tr["ticker"]) == {"KXNFLGAME-25SEP07NYGWAS-WAS", "KXNFLGAME-25SEP07NYGWAS-NYG"}

    # resumable: a second run makes no market/candle/trade calls
    fake.calls.clear()
    kalshi_pull.pull_season(2025, client=fake, log=lambda m: None)
    assert fake.calls == []

    # consolidated: one file per table, the same rows, minus the columns analysis never reads
    paths = kalshi_pull.consolidate(2025, log=lambda m: None)
    for table, per_game in (("markets", mk), ("candles", cd), ("trades", tr)):
        whole = pl.read_parquet(paths[table])
        assert whole.height == per_game.height and "game_id" in whole.columns
    assert "trade_id" not in pl.read_parquet(paths["trades"]).columns

    # coverage: quotes in 3 of the 4 in-game minutes
    timing = pl.DataFrame({"game_id": ["2025_01_NYG_WAS"], "snap_raw": [1.0],
                           "first_snap": [KICK], "last_end": [KICK + dt.timedelta(minutes=4)]})
    cov = inventory.kalshi_coverage(2025, timing).row(0, named=True)
    assert cov["winner_markets"] == 2 and cov["traded_rungs"] == 2
    assert cov["in_game_quote_share"] == pytest.approx(0.75)
    assert cov["trades"] == 2


def test_one_sided_minutes_count_as_quoted(data_dir, shapes):
    """A 99c bid with no ask means 'nearly certain', not missing data."""
    fake = FakeKalshi(shapes)
    kalshi_pull.pull_season(2025, client=fake, log=lambda m: None)
    gdir = kalshi_pull.game_dir(2025, "2025_01_NYG_WAS")
    cd = pl.read_parquet(gdir / "candles.parquet")
    one_sided = cd.with_columns(
        bid_close=pl.when(pl.col("end_ts") == KICK + dt.timedelta(minutes=3)).then(0.99)
        .otherwise(pl.col("bid_close")),
        ask_close=pl.when(pl.col("end_ts") == KICK + dt.timedelta(minutes=3)).then(1.0)
        .otherwise(pl.col("ask_close")),
        mid_close=pl.when(pl.col("end_ts") == KICK + dt.timedelta(minutes=3)).then(None)
        .otherwise(pl.col("mid_close")),
    )
    one_sided.write_parquet(gdir / "candles.parquet")
    timing = pl.DataFrame({"game_id": ["2025_01_NYG_WAS"], "snap_raw": [1.0],
                           "first_snap": [KICK], "last_end": [KICK + dt.timedelta(minutes=4)]})
    cov = inventory.kalshi_coverage(2025, timing).row(0, named=True)
    assert cov["in_game_quote_share"] == pytest.approx(0.75)
    assert cov["two_sided_share"] == pytest.approx(0.5)


class FlakyKalshi(FakeKalshi):
    def __init__(self, shapes, fail=True):
        super().__init__(shapes)
        self.fail = fail

    def trades(self, ticker, start, end):
        if self.fail:
            raise KalshiError("GET /historical/trades -> HTTP 500: boom")
        return super().trades(ticker, start, end)


def test_failures_leave_an_error_file_and_retry(data_dir, shapes):
    gdir = kalshi_pull.game_dir(2025, "2025_01_NYG_WAS")
    failures = kalshi_pull.pull_season(2025, client=FlakyKalshi(shapes), log=lambda m: None)
    assert failures and failures[0][0] == "2025_01_NYG_WAS"
    assert "HTTP 500" in (gdir / "ERROR.txt").read_text() and not (gdir / "DONE").exists()
    assert kalshi_pull.pull_season(2025, client=FlakyKalshi(shapes, fail=False),
                                   log=lambda m: None) == []
    assert (gdir / "DONE").exists() and not (gdir / "ERROR.txt").exists()
    # --redo re-pulls a finished game
    fake = FakeKalshi(shapes)
    kalshi_pull.pull_season(2025, client=fake, redo={"2025_01_NYG_WAS"}, log=lambda m: None)
    assert any(kind == "candles" for kind, _ in fake.calls)


SB_KICK = dt.datetime(2026, 2, 8, 23, 30, tzinfo=UTC)  # 18:30 ET


class SuperBowlKalshi:
    """No KXNFLGAME event; the winner trades as KXSB-26-<team>; the ladders exist
    under the suffix with the teams in the other order."""

    def __init__(self, shapes):
        self.s, self.n_requests, self.calls = shapes, 0, []

    def events(self, series):
        return []

    def markets(self, event_ticker):
        self.calls.append(("markets", event_ticker))
        base = self.s["winner_market"]
        if event_ticker == "KXSB-26":
            return [normalize_market(dict(base, ticker=f"KXSB-26-{t}", yes_sub_title=t))
                    for t in ("SEA", "NE", "KC")]
        if event_ticker == "KXNFLSPREAD-26FEB08NESEA":
            return [normalize_market(dict(self.s["spread_market"],
                                          ticker="KXNFLSPREAD-26FEB08NESEA-SEA3",
                                          floor_strike=3.5, volume_fp="500.00"))]
        return []

    def candles(self, market, start, end):
        assert (start, end) == (SB_KICK - dt.timedelta(hours=3), SB_KICK + dt.timedelta(hours=8))
        raw = dict(self.s["historical_candle"], end_period_ts=int(SB_KICK.timestamp()) + 60)
        return [normalize_candle(raw, market["ticker"])]

    def trades(self, ticker, start, end):
        return []


def test_super_bowl_uses_the_champion_market(data_dir, shapes):
    pl.DataFrame({
        "game_id": ["2025_22_SEA_NE"], "season": [2025], "game_type": ["SB"], "week": [22],
        "gameday": ["2026-02-08"], "gametime": ["18:30"], "away_team": ["SEA"],
        "home_team": ["NE"], "result": [-16],
    }).write_parquet(data_dir / "raw/nflverse/schedules.parquet")
    fake = SuperBowlKalshi(shapes)
    assert kalshi_pull.pull_season(2025, client=fake, log=lambda m: None) == []
    mk = pl.read_parquet(kalshi_pull.game_dir(2025, "2025_22_SEA_NE") / "markets.parquet")
    winners = mk.filter(pl.col("kind") == "winner")
    assert dict(zip(winners["ticker"], winners["side"])) == {"KXSB-26-SEA": "away", "KXSB-26-NE": "home"}
    assert mk.filter(pl.col("kind") == "spread")["ticker"].to_list() == ["KXNFLSPREAD-26FEB08NESEA-SEA3"]
