import datetime as dt

import polars as pl
import pytest

from gridline.data.catalog import (
    TEAM_NAMES, NFLVERSE_TEAMS, candidate_suffixes, championship_events, classify_market,
    market_team, match_events, parse_event_ticker, split_matchup, team_from_text,
    team_from_token, to_nflverse,
)


@pytest.mark.parametrize("code,expected", [
    ("NYGWAS", ("NYG", "WAS")),
    ("TBLA", ("TB", "LA")),        # Rams are LA on Kalshi too
    ("JACARI", ("JAX", "ARI")),    # Kalshi's JAC is nflverse's JAX
    ("LACLA", ("LAC", "LA")),
    ("NENYJ", ("NE", "NYJ")),
    ("ATLGB", ("ATL", "GB")),
    ("SFLAR", ("SF", "LA")),       # 2026 tickers spell the Rams LAR
    ("LACLAR", ("LAC", "LA")),
    ("XXXYYY", None),
])
def test_split_matchup(code, expected):
    assert split_matchup(code) == expected


def test_aliases():
    assert to_nflverse("JAC") == "JAX" and to_nflverse("KC") == "KC"
    assert to_nflverse("ZZZ") is None


def test_parse_event_ticker():
    k = parse_event_ticker("KXNFLGAME-26SEP24ATLGB")
    assert (k.series, k.date, k.away, k.home) == ("KXNFLGAME", dt.date(2026, 9, 24), "ATL", "GB")
    assert k.ticker("KXNFLSPREAD") == "KXNFLSPREAD-26SEP24ATLGB"
    assert parse_event_ticker("KXNFLGAME-26XYZ24ATLGB") is None
    assert parse_event_ticker("garbage") is None


SCHED = pl.DataFrame({
    "game_id": ["2025_01_NYG_WAS", "2025_22_SEA_NE", "2025_12_JAX_ARI"],
    "season": [2025] * 3, "game_type": ["REG", "SB", "REG"], "week": [1, 22, 12],
    "gameday": ["2025-09-07", "2026-02-08", "2025-11-23"],
    "gametime": ["13:00", "18:30", "16:05"],
    "away_team": ["NYG", "SEA", "JAX"], "home_team": ["WAS", "NE", "ARI"],
})


def test_match_events():
    out = match_events([
        "KXNFLGAME-25SEP07NYGWAS",   # exact
        "KXNFLGAME-26FEB08NESEA",    # neutral site, teams in the other order
        "KXNFLGAME-25NOV24JACARI",   # a day off (late kickoff / date convention)
        "KXNFLGAME-25OCT05NYGWAS",   # same teams, no game that week
        "KXNFLGAME-25NOV23ZZZARI",   # unknown team
    ], SCHED)
    status = dict(zip(out["event_ticker"], out["status"]))
    game = dict(zip(out["event_ticker"], out["game_id"]))
    assert game["KXNFLGAME-25SEP07NYGWAS"] == "2025_01_NYG_WAS"
    assert game["KXNFLGAME-26FEB08NESEA"] == "2025_22_SEA_NE"
    assert game["KXNFLGAME-25NOV24JACARI"] == "2025_12_JAX_ARI"
    assert status["KXNFLGAME-25OCT05NYGWAS"] == "no_game"
    assert status["KXNFLGAME-25NOV23ZZZARI"] == "unparsed"


def test_classify_market(shapes):
    from gridline.data.kalshi import normalize_market

    w = classify_market(normalize_market(shapes["winner_market"]), home="WAS", away="NYG")
    assert w == {"kind": "winner", "side": "home", "strike": None}
    s = classify_market(normalize_market(shapes["spread_market"]), home="WAS", away="NYG")
    assert s == {"kind": "spread", "side": "home", "strike": 49.5}
    t = classify_market(normalize_market(shapes["total_market"]), home="WAS", away="NYG")
    assert t == {"kind": "total", "side": None, "strike": 45.5}
    # strike falls back to the ticker when floor_strike is missing
    bare = {"ticker": "KXNFLSPREAD-25SEP07NYGWAS-NYG3", "series_ticker": "KXNFLSPREAD",
            "floor_strike": None}
    assert classify_market(bare, home="WAS", away="NYG")["strike"] == 3.5
    with pytest.raises(ValueError):
        classify_market({"ticker": "KXNFLGAME-X-DAL", "series_ticker": "KXNFLGAME"},
                        home="WAS", away="NYG")


def test_championship_events():
    # verified tickers: KXNFLAFCCHAMP-25 / KXNFLNFCCHAMP-25 (season year), KXSB-26 (game year)
    assert championship_events("CON", 2025) == ["KXNFLAFCCHAMP-25", "KXNFLNFCCHAMP-25"]
    assert championship_events("SB", 2025) == ["KXSB-26"]
    assert championship_events("REG", 2025) == []


def test_candidate_suffixes_cover_orders_and_spellings():
    s = candidate_suffixes("2026-02-08", "SEA", "NE")
    assert s == ["26FEB08SEANE", "26FEB08NESEA"]
    s = candidate_suffixes("2026-01-25", "LA", "SEA")
    assert {"26JAN25LASEA", "26JAN25SEALA", "26JAN25LARSEA", "26JAN25SEALAR"} == set(s)
    assert "26JAN11BUFJAC" in candidate_suffixes("2026-01-11", "BUF", "JAX")


def test_champion_markets_classify_as_winners():
    sea = {"ticker": "KXSB-26-SEA", "series_ticker": "KXSB", "floor_strike": None}
    ne = {"ticker": "KXNFLAFCCHAMP-25-NE", "series_ticker": "KXNFLAFCCHAMP", "floor_strike": None}
    assert market_team(sea) == "SEA"
    assert classify_market(sea, home="NE", away="SEA") == {"kind": "winner", "side": "away", "strike": None}
    assert classify_market(ne, home="DEN", away="NE")["side"] == "away"


def test_team_names_cover_every_team():
    assert set(TEAM_NAMES) == set(NFLVERSE_TEAMS)


@pytest.mark.parametrize("token,code", [
    ("BUF", "BUF"), ("JAC", "JAX"), ("LAR", "LA"),
    ("BUFFALO", "BUF"), ("PHILADELPHIA", "PHI"), ("NEWORLEANS", "NO"),
    ("NEWYORKG", "NYG"), ("LOSANGELESC", "LAC"), ("SEAHAWKS", "SEA"), ("NOPE", None),
])
def test_team_from_token(token, code):
    assert team_from_token(token) == code


def test_team_from_text():
    assert team_from_text("Buffalo wins by over 22.5 points", ("BUF", "NO")) == "BUF"
    assert team_from_text("New York G wins by over 3.5 points", ("NYG", "WAS")) == "NYG"
    assert team_from_text("New York wins", ("NYJ", "MIA")) == "NYJ"  # only one NY team
    assert team_from_text("New York wins", ("NYJ", "NYG")) is None   # ambiguous
    assert team_from_text("New York G wins by over 2.5", ("NYJ", "NYG")) == "NYG"
    assert team_from_text("Los Angeles C wins", ("LA", "LAC")) == "LAC"
    assert team_from_text("Over 45.5 points scored", ("BUF", "NO")) is None


def test_spread_tickers_with_team_names():
    # real tickers from the 2025 pull that failed on the code-only parser
    buf = {"ticker": "KXNFLSPREAD-25SEP28NOBUF-BUFFALO22", "series_ticker": "KXNFLSPREAD",
           "floor_strike": 22.5, "yes_sub_title": "Buffalo wins by over 22.5 points"}
    phi = {"ticker": "KXNFLSPREAD-25NOV16DETPHI-PHILADELPHIA17", "series_ticker": "KXNFLSPREAD",
           "floor_strike": 17.5}
    assert classify_market(buf, home="BUF", away="NO") == {"kind": "spread", "side": "home", "strike": 22.5}
    assert classify_market(phi, home="PHI", away="DET")["side"] == "home"
    # an unreadable token still resolves from the market's own words
    odd = {"ticker": "KXNFLSPREAD-X-ZZ3", "series_ticker": "KXNFLSPREAD", "floor_strike": 3.5,
           "yes_sub_title": "New Orleans wins by over 3.5 points"}
    assert classify_market(odd, home="BUF", away="NO")["side"] == "away"
