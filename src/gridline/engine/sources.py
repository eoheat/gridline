"""Event sources: where the engine learns that the game changed.

An Event says "the game just changed, and here is the new state". ReplaySource turns a
game's play-by-play into the events a live engine would have seen:

  pregame  before the opening kickoff (time: the kickoff)
  play     a play ended. The new state is the next snap's situation (score, possession,
           down, distance, yard line, and what comes next: a snap, a try or a kickoff).
           The clock is the play's own clock minus the play's length: the clock that
           runs between plays is not known yet (DESIGN.md decision 5). Timeouts are
           those before the play; a timeout called after it shows up at the next event.
  snap     the next snap. The clock that ran between plays is now known. Emitted where
           that matters: the 4th quarter, overtime, and the last 5 minutes of the 2nd
           quarter; and before an overtime kickoff, once the coin toss is known.
  final    the game is over.

When a play's result became public (Event.timing), conservatively, since information
may arrive late but never early:

  end      the play's end timestamp from the NFL feed
  imputed  the feed has no usable end: the snap plus the longest usual duration of that
           type of play (END_QUANTILE of plays that have both timestamps), capped by the
           next snap
  penalty  the result waited on a penalty ruling (enforced, declined or offsetting):
           PENALTY_LAG_S after the play, or the next snap if that came first
  review   the result waited on a replay review or a challenge: the next snap, the
           first moment the ruling is certainly known (REVIEW_LAG_S after the play if
           there is no next snap). Until then the replay keeps the price from before the
           play; the benchmark leaves those minutes out for model and market alike,
           since a live engine would have priced the call on the field instead.
  clock    the period ran out after the play with time still on the clock: when the
           clock reached zero (the play's end plus the clock that was left), capped by
           the next snap. Until then nobody knows no further play is coming.

A next snap whose own time was imputed proves nothing and caps nothing.

The engine learns an event `delay` seconds after its time (the delay is applied by
whoever consumes the log, so one replay serves every delay). A live feed would emit the
same Event objects, which keeps replay and live on one code path (decision 1).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, replace
from typing import Iterator

import polars as pl

from gridline.models import simulator as sim
from gridline.state.game_state import GameState

MAX_PLAY_SECONDS = 40.0
SNAP_EVENT_HALF_SECONDS = 300  # 2nd quarter: snap events in its last 5 minutes
END_QUANTILE = 0.99
DEFAULT_END_BOUND_S = 20.0  # a play type with no measured durations
PENALTY_LAG_S = 60.0  # a penalty is announced within about a minute of the play
REVIEW_LAG_S = 240.0  # a review with no next snap to show when it ended (the median gap is ~3.5 min)
REVIEW = re.compile(r"replay official|challenged|reviewed|reversed|upheld", re.IGNORECASE)
PENALTY = re.compile(r"penalty", re.IGNORECASE)


@dataclass(frozen=True)
class Event:
    game_id: str
    seq: int  # events row that produced it (the play that ended, or the snap); -1 pregame
    kind: str  # "pregame" | "play" | "snap" | "final"
    t: dt.datetime  # when it became public, before any feed delay (UTC)
    state: GameState
    phase: sim.Phase  # what comes next: "snap", "kickoff" or "pat" (ignored when final)
    kicker_home: bool | None = None  # phase "kickoff": who kicks (None: a coin toss to come)
    free_kick: bool = False  # the kick is a free kick after a safety
    ot_possessions: int = 0
    timing: str = "end"  # how t was set: pregame, end, imputed, penalty, review, clock or snap
    t_play_end: dt.datetime | None = None  # play and final events: the play's end in the feed


def end_bounds(events: pl.DataFrame, q: float = END_QUANTILE) -> dict[str, float]:
    """Longest usual wall-clock length of each type of play (snap to end): the q-quantile
    over plays whose snap and end both came from the feed. Measure it on seasons before
    the one replayed."""
    d = (events.filter(pl.col("play_type").is_not_null() & ~pl.col("snap_imputed") & ~pl.col("end_imputed"))
         .group_by("play_type")
         .agg(s=(pl.col("t_end") - pl.col("t_snap")).dt.total_seconds().quantile(q)))
    return dict(zip(d["play_type"].to_list(), d["s"].to_list()))


def _is_play(row: dict) -> bool:
    # plays change the game; timeouts (no possession team) and administrative rows don't
    return row["play_type"] is not None and row["posteam"] is not None


def _is_kickoff(row: dict) -> bool:
    # a kickoff called back by a penalty is logged as no_play, without a down
    return row["play_type"] == "kickoff" or (
        row["play_type"] == "no_play" and row["down"] is None and " kicks " in (row["desc"] or ""))


def _is_try(row: dict) -> bool:
    # every other row without a down is a try: an extra point, a two-point attempt, or a
    # try called back by a penalty (no_play)
    return row["down"] is None and not _is_kickoff(row)


def _phase(nxt: dict, home: str) -> tuple[sim.Phase, bool | None]:
    if _is_kickoff(nxt):
        return "kickoff", nxt["posteam"] != home  # on kickoffs, posteam is the receiver
    if _is_try(nxt):
        return "pat", None
    return "snap", None


def _needs_snap_event(row: dict) -> bool:
    q = row["qtr"]
    return q >= 4 or (q == 2 and row["half_seconds_remaining"] <= SNAP_EVENT_HALF_SECONDS)


def _length(row: dict) -> float:
    """Game clock the play itself used: its wall-clock length, capped."""
    s = min(max((row["t_end"] - row["t_snap"]).total_seconds(), 0.0), MAX_PLAY_SECONDS)
    return min(s, row["half_seconds_remaining"] if row["qtr"] <= 4 else row["game_seconds_remaining"])


def _clock_left(row: dict) -> float:
    """Clock left in the period after the play."""
    left = row["half_seconds_remaining"] if row["qtr"] <= 4 else row["game_seconds_remaining"]
    return max(left - _length(row), 0.0)


def _public(cur: dict, nxt: dict | None, bounds: dict[str, float], period_over: bool) -> tuple[dt.datetime, str]:
    """When the result of play `cur` became public (see the module docstring)."""
    cap = nxt["t_snap"] if nxt is not None and not nxt["snap_imputed"] else None
    t, timing = cur["t_end"], "end"
    if cur["end_imputed"]:
        t = cur["t_snap"] + dt.timedelta(seconds=bounds.get(cur["play_type"], DEFAULT_END_BOUND_S))
        timing = "imputed"
    if period_over and _clock_left(cur) > 0:
        t, timing = t + dt.timedelta(seconds=_clock_left(cur)), "clock"
    if timing != "end" and cap is not None:  # the next snap shows the play (and period) was over
        t = min(t, cap)
    desc = cur["desc"] or ""
    if REVIEW.search(desc):
        t, timing = max(t, cap if cap is not None else cur["t_end"] + dt.timedelta(seconds=REVIEW_LAG_S)), "review"
    elif PENALTY.search(desc):
        ruled = cur["t_end"] + dt.timedelta(seconds=PENALTY_LAG_S)
        t, timing = max(t, min(ruled, cap) if cap is not None else ruled), "penalty"
    return max(t, cur["t_end"]), timing


def game_events(rows: pl.DataFrame, bounds: dict[str, float] | None = None) -> Iterator[Event]:
    """Events of one game, in time order, from its rows of the events table. `bounds`:
    end_bounds() of earlier seasons, for plays with no end timestamp."""
    bounds = bounds or {}
    rows = rows.sort("seq")
    plays = [r for r in rows.iter_rows(named=True) if _is_play(r) and r["t_end"] is not None]
    if not plays:
        return
    first = plays[0]
    home = first["home_team"]
    hok = first["home_opening_kickoff"]  # 1: the home team received the opening kickoff
    kicker = None if hok is None else not bool(hok)
    # the opening kicker receives the 2nd-half kickoff (known once the coin toss is)
    s0 = replace(sim.kickoff_state(GameState.pre_snap(first)), home_receives_2h_kickoff=kicker)
    yield Event(first["game_id"], -1, "pregame", first["t_snap"], s0, "kickoff", kicker_home=kicker,
                timing="pregame")
    last_t = first["t_snap"]
    for i, cur in enumerate(plays[:-1]):
        nxt = plays[i + 1]
        phase, kicker_home = _phase(nxt, home)
        new_period = nxt["game_half"] != cur["game_half"]
        s = GameState.pre_snap(nxt)
        # the score after the play (the next row's pre-snap score, known at the whistle)
        s = replace(s, home_score=int(cur["home_score_post"]), away_score=int(cur["away_score_post"]))
        if not new_period:
            s = replace(s, home_timeouts=int(cur["home_timeouts"]), away_timeouts=int(cur["away_timeouts"]))
            if phase == "snap":  # the clock runs on until the next snap: not known yet
                length = _length(cur)
                s = replace(s, game_seconds_remaining=cur["game_seconds_remaining"] - length,
                            half_seconds_remaining=cur["half_seconds_remaining"] - length)
        toss = new_period and s.qtr >= 5 and cur["qtr"] <= 4  # overtime: its coin toss is still to come
        scored = {"home": cur["home_score_post"] - cur["home_score_pre"],
                  "away": cur["away_score_post"] - cur["away_score_pre"]}
        defence = "away" if cur["posteam"] == home else "home"
        offence = "home" if defence == "away" else "away"
        # a safety: two points to the defence, then a free kick (not a try returned for two)
        free = (phase == "kickoff" and not _is_try(cur) and scored[defence] == 2 and scored[offence] == 0)
        ot = sim.overtime_possessions(rows, nxt["seq"]) if s.qtr >= 5 else 0
        t, timing = _public(cur, nxt, bounds, period_over=new_period)
        t = max(t, last_t)
        yield Event(cur["game_id"], cur["seq"], "play", t, s, phase, None if toss else kicker_home, free, ot,
                    timing=timing, t_play_end=cur["t_end"])
        last_t = t
        if nxt["t_snap"] is not None and not nxt["snap_imputed"] and (
                toss or (phase == "snap" and _needs_snap_event(nxt))):
            exact = GameState.pre_snap(nxt)
            t = max(nxt["t_snap"], last_t)  # never before the play it follows
            yield Event(nxt["game_id"], nxt["seq"], "snap", t, exact, phase, kicker_home if toss else None,
                        False, ot, timing="snap")
            last_t = t
    last = plays[-1]
    s = GameState.pre_snap(last)
    s = replace(s, home_score=int(last["home_score_post"]), away_score=int(last["away_score_post"]),
                game_seconds_remaining=0.0, half_seconds_remaining=0.0)
    # a decided overtime ends at the last whistle; regulation, or an overtime still tied,
    # ends when the clock runs out
    decided_ot = last["qtr"] >= 5 and last["home_score_post"] != last["away_score_post"]
    t, timing = _public(last, None, bounds, period_over=not decided_ot)
    yield Event(last["game_id"], last["seq"], "final", max(t, last_t), s, "snap", timing=timing,
                t_play_end=last["t_end"])


class ReplaySource:
    """Historical games from the events table, as a live feed would have told them."""

    def __init__(self, events: pl.DataFrame, bounds: dict[str, float] | None = None):
        self._events = events
        self.bounds = bounds or {}
        self.game_ids: list[str] = events["game_id"].unique(maintain_order=True).to_list()

    def rows(self, game_id: str) -> pl.DataFrame:
        return self._events.filter(pl.col("game_id") == game_id)

    def events(self, game_id: str) -> Iterator[Event]:
        return game_events(self.rows(game_id), self.bounds)
