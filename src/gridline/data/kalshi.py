"""Read-only client for Kalshi's public market-data API, plus normalizers.

This project never trades: no credentials, no order endpoints. Series, events,
markets, candlesticks and trades are public GETs on Kalshi's v2 REST API.

Two storage tiers, verified 2026-09-26 (details in docs/DATA.md):

  live tier        GET /markets?event_ticker=...
                   GET /series/{series}/markets/{ticker}/candlesticks
                   GET /markets/trades?ticker=...
  historical tier  GET /historical/markets?event_ticker=...
                   GET /historical/markets/{ticker}/candlesticks
                   GET /historical/trades?ticker=...

  GET /historical/cutoff says where the boundary is: markets settled before
  `market_settled_ts` and trades created before `trades_created_ts` live in the
  historical tier. The two tiers spell the same numbers differently
  (historical: price.close="0.7200", volume="3493.00";
   live:       price.close_dollars="0.6900", volume_fp="12354.44"),
  so every normalizer below accepts both spellings.

Conventions of the normalized output: prices are YES prices in dollars
(0.00-1.00), quantities are contracts, timestamps are tz-aware UTC datetimes.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any, Callable, Iterator

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiError(RuntimeError):
    """A non-retryable API error. The message always includes the response body:
    Kalshi puts the useful part of an error there, not in the status code."""


# ----------------------------------------------------------------------------- parsing


def price(x: Any) -> float | None:
    """A YES price in dollars. Strings are dollar amounts ("0.7200"); bare ints are
    legacy integer cents (72). Empty -> None."""
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        raise TypeError("bool is not a price")
    if isinstance(x, int):
        return x / 100.0
    return float(x)


def qty(x: Any) -> float | None:
    """A contract count (volume, open interest, trade size). Strings ("12354.44",
    fractional contracts exist) and numbers both map to float. Empty -> None."""
    if x is None or x == "":
        return None
    return float(x)


def ts(x: Any) -> dt.datetime | None:
    """ISO-8601 (with 'Z') or unix seconds -> tz-aware UTC datetime."""
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return dt.datetime.fromtimestamp(x, tz=dt.timezone.utc)
    return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def _pick(d: dict | None, name: str) -> Any:
    """Read `name` from a nested OHLC dict in either tier's spelling."""
    if not d:
        return None
    return d.get(f"{name}_dollars", d.get(name))


def valid_quote(bid: float | None, ask: float | None) -> bool:
    """An empty side shows as bid 0.00 / ask 1.00: that is 'no quote', not a price."""
    return bid is not None and ask is not None and 0.0 < bid <= ask < 1.0


def normalize_candle(raw: dict, ticker: str) -> dict:
    yb, ya, pr = raw.get("yes_bid"), raw.get("yes_ask"), raw.get("price")
    bid_close, ask_close = price(_pick(yb, "close")), price(_pick(ya, "close"))
    ok = valid_quote(bid_close, ask_close)
    return {
        "ticker": ticker,
        # a candle covers (end - period, end]; it is only *known* at end_ts
        "end_ts": ts(int(raw["end_period_ts"])),
        "bid_open": price(_pick(yb, "open")),
        "bid_high": price(_pick(yb, "high")),
        "bid_low": price(_pick(yb, "low")),
        "bid_close": bid_close,
        "ask_open": price(_pick(ya, "open")),
        "ask_high": price(_pick(ya, "high")),
        "ask_low": price(_pick(ya, "low")),
        "ask_close": ask_close,
        "trade_open": price(_pick(pr, "open")),
        "trade_high": price(_pick(pr, "high")),
        "trade_low": price(_pick(pr, "low")),
        "trade_close": price(_pick(pr, "close")),
        "trade_mean": price(_pick(pr, "mean")),
        "volume": qty(raw.get("volume_fp", raw.get("volume"))),
        "open_interest": qty(raw.get("open_interest_fp", raw.get("open_interest"))),
        "mid_close": (bid_close + ask_close) / 2 if ok else None,
        "spread_close": (ask_close - bid_close) if ok else None,
    }


def normalize_trade(raw: dict) -> dict:
    return {
        "trade_id": raw.get("trade_id"),
        "ticker": raw.get("ticker"),
        "ts": ts(raw.get("created_time")),
        "yes_price": price(raw.get("yes_price_dollars", raw.get("yes_price"))),
        "count": qty(raw.get("count_fp", raw.get("count"))),
        "taker_side": raw.get("taker_side"),
        "is_block_trade": bool(raw.get("is_block_trade", False)),
    }


def normalize_market(raw: dict) -> dict:
    ticker = raw["ticker"]
    return {
        "ticker": ticker,
        "event_ticker": raw.get("event_ticker") or ticker.rsplit("-", 1)[0],
        "series_ticker": ticker.split("-", 1)[0],
        "title": raw.get("title"),
        "yes_sub_title": raw.get("yes_sub_title"),
        "status": raw.get("status"),
        "result": raw.get("result"),
        "open_time": ts(raw.get("open_time")),
        "close_time": ts(raw.get("close_time")),
        "settlement_ts": ts(raw.get("settlement_ts")),
        "settlement_value": price(
            raw.get("settlement_value_dollars", raw.get("settlement_value"))
        ),
        "strike_type": raw.get("strike_type"),
        "floor_strike": qty(raw.get("floor_strike")),
        "cap_strike": qty(raw.get("cap_strike")),
        "volume": qty(raw.get("volume_fp", raw.get("volume"))),
        "open_interest": qty(raw.get("open_interest_fp", raw.get("open_interest"))),
        "last_price": price(raw.get("last_price_dollars", raw.get("last_price"))),
        "rules_primary": raw.get("rules_primary"),
        "rules_secondary": raw.get("rules_secondary"),
    }


# ----------------------------------------------------------------------------- client


class KalshiPublic:
    """Throttled, retrying GETs against the public API. `session` is injectable
    for tests; `sleep`/`clock` too, so tests never wait."""

    RETRY_STATUS = (429, 500, 502, 503, 504)

    def __init__(
        self,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        min_interval_s: float = 0.1,
        timeout_s: float = 30.0,
        max_retries: int = 8,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sleep, self._clock = sleep, clock
        self._last = -1e18
        self._cutoff: dict[str, dt.datetime] | None = None
        self.n_requests = 0

    # -- transport

    def _throttle(self) -> None:
        wait = self.min_interval_s - (self._clock() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = self._clock()

    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.n_requests += 1
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_s)
            except requests.RequestException as e:  # network blip: retry
                if attempt == self.max_retries:
                    raise KalshiError(f"GET {path} {params}: {e}") from e
                self._sleep(min(2**attempt, 60))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in self.RETRY_STATUS and attempt < self.max_retries:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2**attempt, 60)
                self._sleep(delay)
                continue
            raise KalshiError(f"GET {path} {params} -> HTTP {r.status_code}: {r.text[:500]}")
        raise AssertionError("unreachable")

    def paginate(self, path: str, key: str, params: dict | None = None) -> Iterator[dict]:
        params = dict(params or {})
        while True:
            data = self.get(path, params)
            items = data.get(key) or []
            yield from items
            cursor = data.get("cursor")
            if not cursor or not items:
                return
            params["cursor"] = cursor

    # -- tier routing

    def cutoff(self) -> dict[str, dt.datetime]:
        if self._cutoff is None:
            raw = self.get("/historical/cutoff")
            self._cutoff = {k: ts(v) for k, v in raw.items() if v}
        return self._cutoff

    def market_is_historical(self, market: dict) -> bool:
        """`market` is a normalized market dict."""
        settled = market.get("settlement_ts") or market.get("close_time")
        return settled is not None and settled < self.cutoff()["market_settled_ts"]

    # -- endpoints

    def events(self, series_ticker: str, status: str | None = None) -> list[dict]:
        params = {"series_ticker": series_ticker, "limit": 200}
        if status:
            params["status"] = status
        return list(self.paginate("/events", "events", params))

    def markets(self, event_ticker: str) -> list[dict]:
        """All markets of an event, normalized. Tries the live tier first and
        falls back to the historical tier (the caller doesn't need to know)."""
        params = {"event_ticker": event_ticker, "limit": 1000}
        raw = list(self.paginate("/markets", "markets", params))
        if not raw:
            raw = list(self.paginate("/historical/markets", "markets", params))
        return [normalize_market(m) for m in raw]

    def candles(
        self,
        market: dict,
        start: dt.datetime,
        end: dt.datetime,
        period_min: int = 1,
        chunk_periods: int = 1000,
    ) -> list[dict]:
        """Candlesticks for one normalized market over [start, end], normalized."""
        ticker, series = market["ticker"], market["series_ticker"]
        path = (
            f"/historical/markets/{ticker}/candlesticks"
            if self.market_is_historical(market)
            else f"/series/{series}/markets/{ticker}/candlesticks"
        )
        out, step = [], period_min * 60 * chunk_periods
        lo, hi = int(start.timestamp()), int(end.timestamp())
        while lo < hi:
            params = {"start_ts": lo, "end_ts": min(lo + step, hi), "period_interval": period_min}
            for c in self.get(path, params).get("candlesticks") or []:
                out.append(normalize_candle(c, ticker))
            lo += step
        # chunk edges can repeat a candle; keep one per end_ts
        return list({c["end_ts"]: c for c in out}.values())

    def trades(self, ticker: str, start: dt.datetime, end: dt.datetime) -> list[dict]:
        """All trades in [start, end], normalized, routed across tiers by the
        trades cutoff (a window spanning the cutoff queries both)."""
        cut = self.cutoff()["trades_created_ts"]
        windows = []
        if start < cut:
            windows.append(("/historical/trades", start, min(end, cut)))
        if end > cut:
            windows.append(("/markets/trades", max(start, cut), end))
        out: dict[str, dict] = {}
        for path, lo, hi in windows:
            params = {
                "ticker": ticker,
                "min_ts": int(lo.timestamp()),
                "max_ts": int(hi.timestamp()),
                "limit": 1000,
            }
            for t in self.paginate(path, "trades", params):
                n = normalize_trade(t)
                out[n["trade_id"]] = n  # a trade at the cutoff edge can come back twice
        return sorted(out.values(), key=lambda t: t["ts"])
