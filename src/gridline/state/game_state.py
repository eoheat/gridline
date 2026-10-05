"""GameState: the one object every model and pricer consumes.

Home-team point of view throughout, so nothing flips sign when possession
changes. Built from a row of gridline.state.events (pre-snap values only; the
row's own result lives in *_post columns and is never read here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class GameState:
    game_id: str
    season: int
    season_type: str  # "REG" or "POST" (no ties in the postseason)
    home: str
    away: str
    home_score: int
    away_score: int
    qtr: int  # 1-4; 5 and up = overtime
    game_seconds_remaining: float
    half_seconds_remaining: float
    posteam: str | None  # None when nobody has the ball (e.g. before a kickoff)
    down: int | None
    ydstogo: int | None
    yardline_100: int | None  # yards from the opponent's end zone, posteam's view
    home_timeouts: int
    away_timeouts: int
    home_receives_2h_kickoff: bool | None
    pregame_spread: float | None  # points the home team is favoured by
    pregame_total: float | None

    @property
    def score_diff_home(self) -> int:
        return self.home_score - self.away_score

    @property
    def overtime(self) -> bool:
        return self.qtr >= 5

    @property
    def posteam_is_home(self) -> bool | None:
        return None if self.posteam is None else self.posteam == self.home

    @classmethod
    def pre_snap(cls, row: Mapping[str, Any]) -> "GameState":
        """State at the snap of an events row."""

        def opt_int(v: Any) -> int | None:
            return None if v is None else int(v)

        hok = row.get("home_opening_kickoff")
        return cls(
            game_id=row["game_id"],
            season=int(row["season"]),
            season_type=row["season_type"],
            home=row["home_team"],
            away=row["away_team"],
            home_score=int(row["home_score_pre"]),
            away_score=int(row["away_score_pre"]),
            qtr=int(row["qtr"]),
            game_seconds_remaining=float(row["game_seconds_remaining"]),
            half_seconds_remaining=float(row["half_seconds_remaining"]),
            posteam=row.get("posteam"),
            down=opt_int(row.get("down")),
            ydstogo=opt_int(row.get("ydstogo")),
            yardline_100=opt_int(row.get("yardline_100")),
            home_timeouts=int(row["home_timeouts"]),
            away_timeouts=int(row["away_timeouts"]),
            # the team that did NOT receive the opening kickoff receives in the 2nd half
            home_receives_2h_kickoff=None if hok is None else not bool(hok),
            pregame_spread=row.get("pregame_spread"),
            pregame_total=row.get("pregame_total"),
        )
