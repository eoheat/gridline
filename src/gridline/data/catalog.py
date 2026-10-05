"""Map nflverse games to Kalshi event and market tickers.

Ticker formats (verified 2026-09-26, see docs/DATA.md):

  event   KXNFLGAME-25SEP07NYGWAS         <series>-<YYMONDD><AWAY><HOME>, Eastern date
  winner  KXNFLGAME-25SEP07NYGWAS-WAS     one contract per team: "<team> wins?"
  spread  KXNFLSPREAD-25SEP07NYGWAS-WAS9  "<team> wins by over 9.5 points?" (floor_strike 9.5)
  total   KXNFLTOTAL-25SEP07NYGWAS-45     "over 45.5 points scored?"        (floor_strike 45.5)

The three series share the <YYMONDD><AWAY><HOME> suffix, so one matched
KXNFLGAME event gives the spread and total events too.

Team codes are nflverse's except Jacksonville (Kalshi JAC, nflverse JAX) and, from
2026, the Rams (Kalshi LAR; LA in 2025 tickers and in nflverse). Matching uses the
*unordered* team pair within +-1 day, so neutral-site games (where "home" is only a
designation) still match, and the nflverse schedule decides which side is home.

Conference championships and the Super Bowl are the exception (verified 2026-09-27):
their winner trades as a "champion" market, not KXNFLGAME. For the 2025 season that
is KXNFLAFCCHAMP-25 / KXNFLNFCCHAMP-25 (one market per conference team, named for the
season) and KXSB-26 (one market per team, named for the calendar year of the game).
The NFC title game also had a KXNFLGAME event, with zero volume.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Iterable

import polars as pl

NFLVERSE_TEAMS = frozenset(
    "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC "
    "LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split()
)
KALSHI_TO_NFLVERSE = {"JAC": "JAX", "LAR": "LA"}  # both seen in real tickers
_DEFENSIVE_ALIASES = {"WSH": "WAS", "LVR": "LV"}  # not seen yet; harmless
# nflverse code -> Kalshi spellings to try when a ticker has to be built, not parsed
_KALSHI_SPELLINGS = {"JAX": ("JAC", "JAX"), "LA": ("LA", "LAR")}

# Kalshi sometimes spells a team out instead of using its code, e.g. the spread
# ticker KXNFLSPREAD-25SEP28NOBUF-BUFFALO22 (2025 NO @ BUF). Names are matched after
# upper-casing and dropping spaces and punctuation: "New York G" -> "NEWYORKG".
TEAM_NAMES = {
    "ARI": ("Arizona", "Cardinals"), "ATL": ("Atlanta", "Falcons"),
    "BAL": ("Baltimore", "Ravens"), "BUF": ("Buffalo", "Bills"),
    "CAR": ("Carolina", "Panthers"), "CHI": ("Chicago", "Bears"),
    "CIN": ("Cincinnati", "Bengals"), "CLE": ("Cleveland", "Browns"),
    "DAL": ("Dallas", "Cowboys"), "DEN": ("Denver", "Broncos"),
    "DET": ("Detroit", "Lions"), "GB": ("Green Bay", "Packers"),
    "HOU": ("Houston", "Texans"), "IND": ("Indianapolis", "Colts"),
    "JAX": ("Jacksonville", "Jaguars"), "KC": ("Kansas City", "Chiefs"),
    "LA": ("Los Angeles R", "Rams"), "LAC": ("Los Angeles C", "Chargers"),
    "LV": ("Las Vegas", "Raiders"), "MIA": ("Miami", "Dolphins"),
    "MIN": ("Minnesota", "Vikings"), "NE": ("New England", "Patriots"),
    "NO": ("New Orleans", "Saints"), "NYG": ("New York G", "Giants"),
    "NYJ": ("New York J", "Jets"), "PHI": ("Philadelphia", "Eagles"),
    "PIT": ("Pittsburgh", "Steelers"), "SEA": ("Seattle", "Seahawks"),
    "SF": ("San Francisco", "49ers"), "TB": ("Tampa Bay", "Buccaneers"),
    "TEN": ("Tennessee", "Titans"), "WAS": ("Washington", "Commanders"),
}


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


_NAME_TO_CODE = {_norm(n): code for code, names in TEAM_NAMES.items() for n in names}

WINNER_SERIES = frozenset({"KXNFLGAME", "KXNFLAFCCHAMP", "KXNFLNFCCHAMP", "KXSB"})
CHAMPIONSHIP_GAME_TYPES = frozenset({"CON", "SB"})

_MONTHS = {
    m: i
    for i, m in enumerate(
        "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), start=1
    )
}
_EVENT_RE = re.compile(
    r"^(?P<series>[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<teams>[A-Z]+)$"
)


def to_nflverse(code: str) -> str | None:
    c = KALSHI_TO_NFLVERSE.get(code, _DEFENSIVE_ALIASES.get(code, code))
    return c if c in NFLVERSE_TEAMS else None


def team_from_token(token: str) -> str | None:
    """A ticker's team token -> nflverse code: 'BUF', 'JAC', 'LAR', 'BUFFALO', 'NEWYORKG'."""
    return to_nflverse(token) or _NAME_TO_CODE.get(_norm(token))


def team_from_text(text: str, candidates: tuple[str, ...]) -> str | None:
    """Which of `candidates` a display text names ('Buffalo wins by over 22.5 points').
    A plain city shared by two teams ('New York') only counts if one candidate has it."""
    words = _norm(text)
    exact, loose = [], []
    for code in candidates:
        names = [_norm(n) for n in TEAM_NAMES.get(code, ())]
        if any(n in words for n in names):
            exact.append(code)
        elif code in ("NYG", "NYJ", "LA", "LAC") and names and names[0][:-1] in words:
            loose.append(code)  # "New York" / "Los Angeles" without the G/J/R/C
    for hits in (exact, loose):
        if len(hits) == 1:
            return hits[0]
    return None


def split_matchup(teams: str) -> tuple[str, str] | None:
    """'NYGWAS' -> ('NYG', 'WAS'), in ticker order (away, home), as nflverse codes.
    None if no split, or more than one split, gives two known teams."""
    splits = []
    for i in range(2, len(teams) - 1):
        a, b = to_nflverse(teams[:i]), to_nflverse(teams[i:])
        if a and b and a != b:
            splits.append((a, b))
    return splits[0] if len(splits) == 1 else None


@dataclass(frozen=True)
class EventKey:
    series: str
    date: dt.date
    suffix: str  # "25SEP07NYGWAS", shared by all three series for one game
    away: str | None  # nflverse codes in ticker order; None if the split was ambiguous
    home: str | None

    def ticker(self, series: str) -> str:
        return f"{series}-{self.suffix}"


def parse_event_ticker(event_ticker: str) -> EventKey | None:
    m = _EVENT_RE.match(event_ticker)
    if not m or m["mon"] not in _MONTHS:
        return None
    try:
        date = dt.date(2000 + int(m["yy"]), _MONTHS[m["mon"]], int(m["dd"]))
    except ValueError:
        return None
    teams = split_matchup(m["teams"])
    suffix = event_ticker.split("-", 1)[1]
    away, home = teams if teams else (None, None)
    return EventKey(m["series"], date, suffix, away, home)


def championship_events(game_type: str, season: int) -> list[str]:
    """Event tickers whose markets settle on this championship game's winner, to
    try in order (a conference game is in one of the two conference events)."""
    if game_type == "CON":
        return [f"KXNFLAFCCHAMP-{season % 100:02d}", f"KXNFLNFCCHAMP-{season % 100:02d}"]
    if game_type == "SB":
        return [f"KXSB-{(season + 1) % 100:02d}"]
    return []


def candidate_suffixes(gameday: str, away: str, home: str) -> list[str]:
    """Possible <YYMONDD><AWAY><HOME> suffixes for a game with no KXNFLGAME event,
    covering both team orders and each Kalshi spelling of the codes."""
    d = dt.date.fromisoformat(gameday)
    date = f"{d:%y}{list(_MONTHS)[d.month - 1]}{d:%d}"
    out = []
    for a in _KALSHI_SPELLINGS.get(away, (away,)):
        for h in _KALSHI_SPELLINGS.get(home, (home,)):
            out += [f"{date}{a}{h}", f"{date}{h}{a}"]
    return list(dict.fromkeys(out))


def market_team(market: dict) -> str | None:
    """The nflverse team a winner or champion market is about ('...-SEA' -> 'SEA')."""
    return team_from_token(market["ticker"].rsplit("-", 1)[1])


def match_events(
    event_tickers: Iterable[str], schedule: pl.DataFrame, tolerance_days: int = 1
) -> pl.DataFrame:
    """One row per event ticker with its nflverse game_id (if exactly one game
    matches) and a status: matched | no_game | ambiguous | unparsed."""
    index: dict[frozenset, list[dict]] = {}
    for g in schedule.select(
        "game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team"
    ).iter_rows(named=True):
        g["_date"] = dt.date.fromisoformat(g["gameday"])
        index.setdefault(frozenset((g["away_team"], g["home_team"])), []).append(g)

    rows = []
    for et in event_tickers:
        key = parse_event_ticker(et)
        row = {"event_ticker": et, "kalshi_date": key.date if key else None, "suffix": None,
               "game_id": None, "status": "unparsed"}
        if key is not None and key.away is not None:
            row["suffix"] = key.suffix
            cands = [
                g for g in index.get(frozenset((key.away, key.home)), [])
                if abs((g["_date"] - key.date).days) <= tolerance_days
            ]
            if len(cands) == 1:
                row.update(game_id=cands[0]["game_id"], status="matched")
            else:
                row["status"] = "no_game" if not cands else "ambiguous"
        rows.append(row)
    schema = {"event_ticker": pl.Utf8, "kalshi_date": pl.Date, "suffix": pl.Utf8,
              "game_id": pl.Utf8, "status": pl.Utf8}
    return pl.DataFrame(rows, schema=schema)


_SPREAD_SUFFIX_RE = re.compile(r"^(?P<team>[A-Z]+?)(?P<n>\d+)$")


def classify_market(market: dict, home: str, away: str) -> dict:
    """What a normalized market pays on, in home/away terms.

    kind   winner | spread | total
    side   home | away (winner/spread), None (total)
    strike YES pays if margin (spread) or total points (total) > strike
    """
    series = market["series_ticker"]
    tail = market["ticker"].rsplit("-", 1)[1]
    strike = market.get("floor_strike")
    if series in WINNER_SERIES:
        team, kind, strike = team_from_token(tail), "winner", None
    elif series == "KXNFLSPREAD":
        m = _SPREAD_SUFFIX_RE.match(tail)
        team, kind = (team_from_token(m["team"]) if m else None), "spread"
        if strike is None and m:
            strike = int(m["n"]) + 0.5
    elif series == "KXNFLTOTAL":
        team, kind = None, "total"
        if strike is None and tail.isdigit():
            strike = int(tail) + 0.5
    else:
        raise ValueError(f"not an NFL game market: {market['ticker']}")
    side = None
    if kind != "total":
        if team not in (home, away):  # last resort: the market's own words
            team = (team_from_text(market.get("yes_sub_title") or "", (home, away))
                    or team_from_text(market.get("title") or "", (home, away)))
        side = "home" if team == home else "away" if team == away else None
        if side is None:
            raise ValueError(f"{market['ticker']}: team {team!r} is neither {home} nor {away}")
    return {"kind": kind, "side": side, "strike": strike}
