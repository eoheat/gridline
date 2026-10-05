import datetime as dt

import pytest

from gridline.data.kalshi import (
    normalize_candle, normalize_market, normalize_trade, price, qty, ts, valid_quote,
)

UTC = dt.timezone.utc


def test_both_tiers_normalize_to_the_same_fields(shapes):
    h = normalize_candle(shapes["historical_candle"], "T")
    live = normalize_candle(shapes["live_candle"], "T")
    assert h.keys() == live.keys()
    assert (h["bid_close"], h["ask_close"], h["mid_close"]) == (0.71, 0.72, pytest.approx(0.715))
    assert (live["bid_close"], live["ask_close"], live["mid_close"]) == (0.68, 0.69, pytest.approx(0.685))
    assert h["volume"] == 3493.0 and live["volume"] == pytest.approx(12354.44)
    assert live["open_interest"] == pytest.approx(3157869.07)
    assert live["trade_mean"] == pytest.approx(0.6891)
    assert h["end_ts"] == dt.datetime(2025, 9, 7, 17, 0, tzinfo=UTC)
    assert h["spread_close"] == pytest.approx(0.01)


def test_empty_book_side_is_no_quote_not_a_price(shapes):
    c = normalize_candle(shapes["historical_candle_empty_bid"], "T")
    assert c["bid_close"] == 0.0 and c["ask_close"] == 1.0
    assert c["mid_close"] is None and c["spread_close"] is None
    assert c["trade_close"] == 0.99  # last trade still reported


@pytest.mark.parametrize("bid,ask,ok", [
    (0.71, 0.72, True), (0.0, 0.72, False), (0.71, 1.0, False),
    (None, 0.72, False), (0.73, 0.72, False), (0.5, 0.5, True),
])
def test_valid_quote(bid, ask, ok):
    assert valid_quote(bid, ask) is ok


def test_price_units():
    assert price("0.7200") == 0.72
    assert price(72) == 0.72          # legacy integer cents
    assert price(None) is None and price("") is None
    assert qty(3493) == 3493.0        # counts are never divided
    with pytest.raises(TypeError):
        price(True)


def test_trade(shapes):
    t = normalize_trade(shapes["historical_trade"])
    assert t["yes_price"] == 0.92 and t["count"] == 1.0 and t["taker_side"] == "yes"
    assert t["ts"] == dt.datetime(2025, 9, 7, 18, 43, 7, 623065, tzinfo=UTC)
    # Kalshi trims trailing zeros from microseconds; still parses
    t5 = normalize_trade(shapes["historical_trade_5_digit_fraction"])
    assert t5["ts"] == dt.datetime(2025, 9, 7, 18, 43, 1, 200970, tzinfo=UTC)


def test_market(shapes):
    m = normalize_market(shapes["winner_market"])
    assert m["event_ticker"] == "KXNFLGAME-25SEP07NYGWAS"  # derived when absent
    assert m["series_ticker"] == "KXNFLGAME"
    assert m["settlement_value"] == 1.0 and m["result"] == "yes"
    assert m["volume"] == 2752401.0
    assert m["settlement_ts"] > m["close_time"]
    assert "tie" in m["rules_secondary"]  # the 50/50 tie rule the pricer must honour
    s = normalize_market(shapes["spread_market"])
    assert s["floor_strike"] == 49.5 and s["series_ticker"] == "KXNFLSPREAD"


def test_ts_accepts_epoch_and_iso():
    assert ts(0) == dt.datetime(1970, 1, 1, tzinfo=UTC)
    assert ts("2026-07-28T00:00:00Z") == dt.datetime(2026, 7, 28, tzinfo=UTC)
