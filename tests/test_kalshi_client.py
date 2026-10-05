"""The client against a fake HTTP session: tier routing, pagination, retries."""

import datetime as dt

import pytest

from gridline.data.kalshi import KalshiError, KalshiPublic, normalize_market

UTC = dt.timezone.utc


class FakeResponse:
    def __init__(self, status: int, body: dict | str, headers: dict | None = None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        return self._body

    @property
    def text(self):
        return self._body if isinstance(self._body, str) else str(self._body)


class FakeSession:
    """Routes GETs to a handler(path, params) -> FakeResponse and records calls."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, timeout=None):
        path = url.split("/trade-api/v2", 1)[1]
        self.calls.append((path, dict(params or {})))
        return self.handler(path, dict(params or {}))


def make_client(handler) -> tuple[KalshiPublic, FakeSession, list]:
    sleeps: list[float] = []
    s = FakeSession(handler)
    c = KalshiPublic(session=s, min_interval_s=0.0, sleep=sleeps.append)
    return c, s, sleeps


def with_cutoff(handler, shapes):
    def h(path, params):
        if path == "/historical/cutoff":
            return FakeResponse(200, shapes["cutoff"])
        return handler(path, params)
    return h


def test_candles_route_by_settlement_vs_cutoff(shapes):
    def h(path, params):
        return FakeResponse(200, {"candlesticks": [shapes["historical_candle"]]})

    c, s, _ = make_client(with_cutoff(h, shapes))
    old = normalize_market(shapes["winner_market"])  # settled 2025-09-07
    new = dict(old, ticker="KXNFLGAME-26SEP24ATLGB-GB",
               settlement_ts=dt.datetime(2026, 9, 25, 4, tzinfo=UTC))
    t0 = dt.datetime(2025, 9, 7, 16, tzinfo=UTC)
    c.candles(old, t0, t0 + dt.timedelta(hours=1))
    c.candles(new, t0, t0 + dt.timedelta(hours=1))
    paths = [p for p, _ in s.calls if "candlesticks" in p]
    assert paths == [
        "/historical/markets/KXNFLGAME-25SEP07NYGWAS-WAS/candlesticks",
        "/series/KXNFLGAME/markets/KXNFLGAME-26SEP24ATLGB-GB/candlesticks",
    ]


def test_candles_chunk_long_windows_and_dedupe(shapes):
    def h(path, params):
        return FakeResponse(200, {"candlesticks": [shapes["historical_candle"]]})

    c, s, _ = make_client(with_cutoff(h, shapes))
    m = normalize_market(shapes["winner_market"])
    t0 = dt.datetime(2025, 9, 7, 12, tzinfo=UTC)
    out = c.candles(m, t0, t0 + dt.timedelta(minutes=250), chunk_periods=100)
    calls = [p for p in s.calls if "candlesticks" in p[0]]
    assert len(calls) == 3  # 250 one-minute periods in chunks of 100
    assert calls[0][1]["end_ts"] - calls[0][1]["start_ts"] == 100 * 60
    assert len(out) == 1  # the same candle returned by every chunk is kept once


def test_trades_split_across_cutoff_and_dedupe(shapes):
    trade = shapes["historical_trade"]

    def h(path, params):
        return FakeResponse(200, {"trades": [trade], "cursor": ""})

    c, s, _ = make_client(with_cutoff(h, shapes))
    cut = dt.datetime(2026, 7, 28, tzinfo=UTC)
    out = c.trades("X", cut - dt.timedelta(hours=1), cut + dt.timedelta(hours=1))
    paths = [p for p, _ in s.calls if "trades" in p]
    assert paths == ["/historical/trades", "/markets/trades"]
    assert len(out) == 1  # same trade_id from both tiers


def test_pagination_follows_cursor():
    pages = {None: (["a", "b"], "c1"), "c1": (["c"], "c2"), "c2": ([], None)}

    def h(path, params):
        items, nxt = pages[params.get("cursor")]
        return FakeResponse(200, {"events": [{"event_ticker": i} for i in items], "cursor": nxt})

    c, s, _ = make_client(h)
    assert [e["event_ticker"] for e in c.events("KXNFLGAME")] == ["a", "b", "c"]


def test_markets_fall_back_to_historical(shapes):
    def h(path, params):
        if path == "/markets":
            return FakeResponse(200, {"markets": [], "cursor": ""})
        return FakeResponse(200, {"markets": [shapes["winner_market"]], "cursor": ""})

    c, s, _ = make_client(h)
    ms = c.markets("KXNFLGAME-25SEP07NYGWAS")
    assert [m["ticker"] for m in ms] == ["KXNFLGAME-25SEP07NYGWAS-WAS"]
    assert [p for p, _ in s.calls] == ["/markets", "/historical/markets"]


def test_retries_429_honouring_retry_after():
    responses = [FakeResponse(429, "slow down", {"Retry-After": "2"}),
                 FakeResponse(503, "busy"), FakeResponse(200, {"ok": 1})]
    c, s, sleeps = make_client(lambda p, q: responses.pop(0))
    assert c.get("/x") == {"ok": 1}
    assert sleeps == [2.0, 2.0]  # Retry-After, then exponential backoff (2**1)


def test_client_errors_carry_the_body():
    c, s, _ = make_client(lambda p, q: FakeResponse(404, '{"error":"market not found"}'))
    with pytest.raises(KalshiError, match="market not found"):
        c.get("/markets/NOPE")
