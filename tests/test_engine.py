"""Phase 3 engine: events from play-by-play without lookahead, the kickoff prior, the
pricer's contract prices, and the price log."""

import datetime as dt
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from gridline import config
from gridline.engine import engine as eng
from gridline.engine import prior as pr
from gridline.engine import sinks, sources
from gridline.models import simulator as sim
from gridline.models import transitions as tr
from gridline.pricing.contracts import ScoreDistribution

from test_simulator import state, stub_engine

T0 = dt.datetime(2025, 9, 7, 17, 0, tzinfo=dt.timezone.utc)


def game(rows: list[dict]) -> pl.DataFrame:
    base = dict(game_id="2025_01_AWAY_HOME", season=2025, season_type="REG", home_team="HOME",
                away_team="AWAY", down=None, ydstogo=None, yardline_100=None, home_timeouts=3,
                away_timeouts=3, pregame_spread=3.0, pregame_total=44.0, home_opening_kickoff=1,
                snap_imputed=False, end_imputed=False, final_margin_home=9, final_total=9, desc="")
    out = []
    for r in rows:
        r = base | r
        r["game_half"] = "Half1" if r["qtr"] <= 2 else "Half2" if r["qtr"] <= 4 else "Overtime"
        r["t_snap"], r["t_end"] = T0 + dt.timedelta(seconds=r.pop("snap")), T0 + dt.timedelta(seconds=r.pop("end"))
        out.append(r)
    return pl.DataFrame(out, infer_schema_length=None).with_columns(
        pl.col("down", "ydstogo", "yardline_100").cast(pl.Int64), pl.col("posteam").cast(pl.String),
        pl.col("play_type").cast(pl.String))


def q1(gsr, **kw):
    return dict(qtr=1, game_seconds_remaining=gsr, half_seconds_remaining=gsr - 1800, **kw)


def score(h0, a0, h1, a1):
    return dict(home_score_pre=h0, away_score_pre=a0, home_score_post=h1, away_score_post=a1)


PLAYS = [
    # the opening kickoff: away kicks, home (posteam) receives
    dict(seq=1, play_type="kickoff", posteam="HOME", snap=0, end=8, **q1(3600), **score(0, 0, 0, 0)),
    dict(seq=2, play_type="run", posteam="HOME", down=1, ydstogo=10, yardline_100=75, snap=40, end=46,
         **q1(3595), **score(0, 0, 0, 0)),
    # a home timeout (no possession team): not a play
    dict(seq=3, play_type=None, posteam=None, snap=60, end=60, **q1(3560), **score(0, 0, 0, 0), home_timeouts=3),
    dict(seq=4, play_type="pass", posteam="HOME", down=2, ydstogo=3, yardline_100=20, snap=100, end=108,
         **q1(3550), **score(0, 0, 6, 0), home_timeouts=2),
    dict(seq=5, play_type="extra_point", posteam="HOME", snap=140, end=145, **q1(3542), **score(6, 0, 7, 0),
         home_timeouts=2),
    dict(seq=6, play_type="kickoff", posteam="AWAY", snap=150, end=158, **q1(3542), **score(7, 0, 7, 0),
         home_timeouts=2),
    # away is tackled in its own end zone: a safety, then away's free kick to home
    dict(seq=7, play_type="run", posteam="AWAY", down=1, ydstogo=10, yardline_100=97, snap=190, end=196,
         **q1(3536), **score(7, 0, 9, 0), home_timeouts=2),
    dict(seq=8, play_type="kickoff", posteam="HOME", snap=230, end=238, **q1(3530), **score(9, 0, 9, 0),
         home_timeouts=2),
    # the 4th quarter (a new half: timeouts reset)
    dict(seq=9, play_type="run", posteam="HOME", down=1, ydstogo=10, yardline_100=60, snap=5000, end=5006,
         qtr=4, game_seconds_remaining=200, half_seconds_remaining=200, **score(9, 0, 9, 0)),
    dict(seq=10, play_type="run", posteam="HOME", down=2, ydstogo=4, yardline_100=54, snap=5040, end=5045,
         qtr=4, game_seconds_remaining=160, half_seconds_remaining=160, **score(9, 0, 9, 0)),
]


def by_key(events):
    return {(e.kind, e.seq): e for e in events}


def test_events_carry_what_is_known_when_each_play_ends():
    ev = list(sources.game_events(game(PLAYS)))
    assert [e.t for e in ev] == sorted(e.t for e in ev)
    e = by_key(ev)
    pre = ev[0]
    assert pre.kind == "pregame" and pre.t == T0 and pre.phase == "kickoff"
    assert pre.kicker_home is False and pre.state.home_receives_2h_kickoff is False  # away kicked off
    # after a play the clock is its own clock minus its length; the rest of the next snap is known
    s = e["play", 1].state
    assert (s.game_seconds_remaining, s.posteam, s.down, s.yardline_100) == (3592, "HOME", 1, 75)
    s = e["play", 2].state  # the timeout after the play is not known yet
    assert (s.game_seconds_remaining, s.down, s.ydstogo, s.home_timeouts) == (3589, 2, 3, 3)
    # a touchdown: the try comes next, with the six points on the board
    assert e["play", 4].phase == "pat" and (e["play", 4].state.home_score, e["play", 4].state.posteam) == (6, "HOME")
    k = e["play", 5]
    assert (k.phase, k.kicker_home, k.free_kick, k.state.home_score) == ("kickoff", True, False, 7)
    k = e["play", 7]  # the safety
    assert (k.phase, k.kicker_home, k.free_kick, k.state.home_score) == ("kickoff", False, True, 9)
    # into a new half: the next snap's own clock and timeouts
    s = e["play", 8].state
    assert (s.qtr, s.game_seconds_remaining, s.home_timeouts) == (4, 200, 3)
    # late in the game the next snap is an event of its own, with the exact clock
    assert e["play", 9].state.game_seconds_remaining == 194
    assert e["snap", 10].state.game_seconds_remaining == 160 and e["snap", 10].t == T0 + dt.timedelta(seconds=5040)
    f = ev[-1]
    assert f.kind == "final" and (f.state.home_score, f.state.away_score) == (9, 0)
    # the last play left 2:35 on a running clock: the game is over when it runs out
    assert f.timing == "clock" and f.t == T0 + dt.timedelta(seconds=5045 + 155)


def test_no_event_depends_on_anything_after_the_next_snap():
    full = game(PLAYS)
    ev_full = by_key(sources.game_events(full))
    plays = [p for p in PLAYS if p["play_type"] is not None]
    for i in range(1, len(plays)):
        cur, nxt = plays[i - 1], plays[i]
        # keep rows up to the next play; scramble the next play's result and timing
        rows = [dict(p) for p in PLAYS if p["seq"] <= nxt["seq"]]
        rows[-1].update(home_score_post=99, away_score_post=77, end=rows[-1]["snap"] + 39)
        cut = by_key(sources.game_events(game(rows)))
        assert cut["play", cur["seq"]] == ev_full["play", cur["seq"]]


def test_a_try_returned_for_two_is_not_a_safety():
    plays = [dict(p) for p in PLAYS[:5]]
    plays[4].update(play_type="pass", **score(6, 0, 6, 2))  # a two-point try, returned by the defence
    plays.append(dict(seq=6, play_type="kickoff", posteam="AWAY", snap=150, end=158, **q1(3542), **score(6, 2, 6, 2)))
    k = by_key(sources.game_events(game(plays)))["play", 5]
    assert (k.phase, k.kicker_home, k.free_kick) == ("kickoff", True, False)


def test_results_that_wait_on_a_ruling_or_have_no_end_time_are_timed_late():
    plays = [dict(p) for p in PLAYS]
    plays[1].update(desc="(15:00) 22-D.Henry right end to BUF 20 for 5 yards. PENALTY on BUF-99, Holding, declined.")
    plays[3].update(end_imputed=True)  # the touchdown pass: the feed has no end time
    e = by_key(sources.game_events(game(plays), bounds={"pass": 11.0}))
    # a penalty: announced within PENALTY_LAG_S, or by the next snap if that comes first
    assert e["play", 2].timing == "penalty" and e["play", 2].t == T0 + dt.timedelta(seconds=100)
    assert e["play", 4].timing == "imputed" and e["play", 4].t == T0 + dt.timedelta(seconds=111)  # snap + 11 s
    assert e["play", 1].timing == "end" and e["play", 1].t_play_end == e["play", 1].t
    plays[5].update(desc="(14:10) 12-T.Brady pass deep left to 13-M.Evans for 40 yards. The Replay Official "
                         "reviewed the pass completion ruling, and the play was REVERSED.")
    e = by_key(sources.game_events(game(plays), bounds={"pass": 11.0}))
    assert e["play", 6].timing == "review" and e["play", 6].t == T0 + dt.timedelta(seconds=190)  # next snap
    assert e["play", 6].t_play_end == T0 + dt.timedelta(seconds=158)
    plays[1].update(desc="PENALTY on AWAY-90, Offside, 5 yards, enforced.", end=46)
    plays[3].update(snap=48)  # the next snap only 2 s after the play ends: it caps the wait
    e = by_key(sources.game_events(game(plays), bounds={"pass": 11.0}))
    assert e["play", 2].t == T0 + dt.timedelta(seconds=48)


def test_penalty_on_a_try_is_still_a_try():
    plays = [dict(p) for p in PLAYS[:4]]
    plays.append(dict(seq=5, play_type="no_play", posteam="HOME", snap=140, end=145, **q1(3542), **score(6, 0, 6, 0),
                      desc="(Kick formation) PENALTY on HOME-74, False Start, 5 yards, enforced at AWAY 15 - No Play."))
    plays.append(dict(seq=6, play_type="extra_point", posteam="HOME", snap=170, end=174, **q1(3542),
                      **score(6, 0, 7, 0)))
    plays.append(dict(seq=7, play_type="kickoff", posteam="AWAY", snap=200, end=208, **q1(3542), **score(7, 0, 7, 0)))
    e = by_key(sources.game_events(game(plays)))
    assert e["play", 4].phase == "pat" and e["play", 5].phase == "pat" and e["play", 6].phase == "kickoff"
    assert e["play", 5].timing == "penalty"


def overtime_game():
    """Tied 20-20 at the end of regulation (the last play leaves 0:02 on the clock), then
    away wins the toss and receives; away punts, home scores a touchdown and wins."""
    def q4(gsr, **kw):
        return dict(qtr=4, game_seconds_remaining=gsr, half_seconds_remaining=gsr, **kw)

    def ot(gsr, **kw):
        return dict(qtr=5, game_seconds_remaining=gsr, half_seconds_remaining=gsr, **kw)

    return game([
        dict(seq=1, play_type="kickoff", posteam="HOME", snap=0, end=8, **q1(3600), **score(0, 0, 0, 0)),
        dict(seq=2, play_type="run", posteam="HOME", down=1, ydstogo=10, yardline_100=75, snap=9000, end=9005,
             **q4(8), **score(20, 20, 20, 20)),
        dict(seq=3, play_type="kickoff", posteam="AWAY", snap=9300, end=9308, **ot(600), **score(20, 20, 20, 20)),
        dict(seq=4, play_type="punt", posteam="AWAY", down=4, ydstogo=8, yardline_100=70, snap=9400, end=9410,
             **ot(560), **score(20, 20, 20, 20)),
        dict(seq=5, play_type="pass", posteam="HOME", down=1, ydstogo=10, yardline_100=40, snap=9450, end=9458,
             **ot(520), **score(20, 20, 26, 20)),
    ])


def test_overtime_coin_toss_is_not_known_before_the_kickoff():
    ev = list(sources.game_events(overtime_game()))
    e = by_key(ev)
    end_reg = e["play", 2]  # 0:08 left, the play ran 5 s, so regulation ended 3 s after it
    assert (end_reg.phase, end_reg.kicker_home, end_reg.timing) == ("kickoff", None, "clock")
    assert end_reg.t == T0 + dt.timedelta(seconds=9008) and end_reg.state.qtr == 5
    toss = e["snap", 3]
    assert (toss.phase, toss.kicker_home, toss.t) == ("kickoff", True, T0 + dt.timedelta(seconds=9300))
    assert e["play", 4].ot_possessions == 1
    f = ev[-1]  # the walk-off touchdown ends the game at the whistle, with no try
    assert (f.kind, f.timing, f.state.home_score) == ("final", "end", 26)
    assert f.t == T0 + dt.timedelta(seconds=9458)


def test_overtime_that_ends_on_a_stop_ends_at_the_whistle():
    rows = overtime_game().to_dicts()
    # away answered home's field goal and home's last drive ends on downs, 3:00 left
    rows[-1].update(play_type="pass", home_score_post=23, away_score_post=26, home_score_pre=23,
                    away_score_pre=26)
    f = list(sources.game_events(pl.DataFrame(rows)))[-1]
    assert (f.timing, f.t) == ("end", T0 + dt.timedelta(seconds=9458))
    rows[-1].update(away_score_post=23, away_score_pre=23)  # still tied: the clock has to run out
    f = list(sources.game_events(pl.DataFrame(rows)))[-1]
    assert f.timing == "clock" and f.t > T0 + dt.timedelta(seconds=9458)


def test_overtime_try_that_takes_the_lead_ends_the_game():
    # 2025 overtime: away scored first (21-14); home has just answered with a touchdown
    # (20-21) and always converts two: 22-21, and the game is over, no kickoff
    engine = stub_engine("td1", 300.0, pat=(0.0, 0.0, 1.0))
    s = state(qtr=5, game_seconds_remaining=300.0, half_seconds_remaining=300.0, home_score=20, away_score=21)
    r = sim.simulate(engine, s, n=2, phase="pat", ot_possessions=1)
    assert (r.home == 22).all() and (r.away == 21).all()


def test_free_kick_uses_the_free_kick_pool():
    engine = stub_engine("end_half", 0.0)
    kicks = tr.Pool(np.array([0, 0, 1]), np.array([1, 1, 1]), np.array([False, False]), np.array([75.0, 75.0]),
                    np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 6.0]))  # free kicks: a return TD
    engine.transitions.kicks = kicks
    s = state(qtr=4, game_seconds_remaining=100.0, half_seconds_remaining=100.0, posteam=None, down=None,
              ydstogo=None, yardline_100=None)
    normal = sim.simulate(engine, s, n=1, phase="kickoff", kicker_home=True)
    free = sim.simulate(engine, s, n=1, phase="kickoff", kicker_home=True, free_kick=True)
    assert (normal.home[0], normal.away[0]) == (0.0, 0.0) and (free.home[0], free.away[0]) == (0.0, 6.0)


def test_opening_kicker_receives_the_second_half_kickoff():
    # every drive is a whole-half field goal: the opening receiver (away) scores in the
    # 1st half and the opening kicker (home) in the 2nd, so regulation ends 3-3
    s = sim.kickoff_state(state())
    for known in (None, True):
        s0 = s if known is None else replace(s, home_receives_2h_kickoff=known)
        r = sim.simulate(stub_engine("fg", 1800.0), s0, n=4, phase="kickoff", kicker_home=True)
        assert r.overtime.all() and (np.minimum(r.home, r.away) == 3).all()


def test_ladder_median_reads_the_crossing():
    assert sim.ladder_median(np.array([40.5, 41.5, 42.5]), np.array([0.7, 0.6, 0.4])) == pytest.approx(42.0)
    assert sim.ladder_median(np.array([42.5, 40.5, 41.5]), np.array([0.4, 0.7, 0.6])) == pytest.approx(42.0)
    assert np.isnan(sim.ladder_median(np.array([40.5, 41.5]), np.array([0.8, 0.6])))
    # a rung out of line is pooled with its neighbour (isotonic regression), not read as
    # a crossing: 0.45 and 0.7 become 0.575 each
    assert sim.ladder_median(np.array([40.5, 41.5, 42.5, 43.5]),
                             np.array([0.8, 0.45, 0.7, 0.3])) == pytest.approx(42.5 + 0.075 / 0.275)
    assert sim.median_total(np.full(100, 44.0)) == pytest.approx(44.0)


def test_solver_keeps_its_best_iterate_and_reports_it():
    # the stub ignores the knobs: the target is out of reach
    engine, s = stub_engine("fg", 1800.0), sim.kickoff_state(state())
    info = {}
    knobs = sim.solve_knobs(engine, s, None, 44.0, n=50, max_iter=3, target_home_win=0.9, keep_best=True,
                            info=info, kicker_home=True)
    assert knobs == (0.0, 44.0) and info["iterations"] == 3 and not info["converged"]
    assert 0 <= info["home_win"] <= 1 and info["total"] == 9.0


def test_quote_prices_match_the_score_distribution():
    rng = np.random.default_rng(1)
    home, away = rng.integers(0, 45, 3000).astype(float), rng.integers(0, 45, 3000).astype(float)
    q, d = eng.Quote.from_scores(home, away), ScoreDistribution(home, away)
    assert q.p_home == pytest.approx(d.winner("home")) and 1 - q.p_home == pytest.approx(d.winner("away"))
    for k in np.arange(0.5, 30, 1.0):
        assert q.price("spread", "home", k) == pytest.approx(d.spread("home", k), abs=1e-6)
        assert q.price("spread", "away", k) == pytest.approx(d.spread("away", k), abs=1e-6)
    for k in np.arange(20.5, 80, 1.0):
        assert q.price("total", None, k) == pytest.approx(d.over(k), abs=1e-6)
    final = eng.Quote.from_scores(np.array([20.0]), np.array([20.0]))
    assert (final.p_home, final.p_tie) == (0.5, 1.0)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    for name, sub in [("DATA_DIR", ""), ("DERIVED_DIR", "derived"), ("LOGS_DIR", "logs")]:
        monkeypatch.setattr(config, name, tmp_path / sub)
    return tmp_path


def candle(ticker, minutes_before, mid, spread=0.01, game_id="G"):
    return {"game_id": game_id, "ticker": ticker, "end_ts": T0 - dt.timedelta(minutes=minutes_before),
            "mid_close": mid, "spread_close": spread}


def test_kalshi_prior_reads_the_last_quotes_before_kickoff(data_dir):
    (data_dir / "derived/kalshi").mkdir(parents=True)
    pl.DataFrame({"ticker": ["H", "A", "T44", "T45", "T46", "U44"], "kind": ["winner"] * 2 + ["total"] * 3 + ["total"],
                  "side": ["home", "away", None, None, None, None], "strike": [None, None, 44.5, 45.5, 46.5, 44.5],
                  "game_id": ["G"] * 5 + ["U"]}).write_parquet(data_dir / "derived/kalshi/markets_2025.parquet")
    rows = [candle("H", 5, 0.60), candle("H", 1, 0.62), candle("H", -1, 0.90),  # after kickoff: unseen
            candle("A", 1, 0.40),
            candle("T44", 1, 0.62), candle("T45", 1, 0.54), candle("T46", 1, 0.44),
            candle("U44", 1, 0.50, spread=0.30, game_id="U")]  # too wide to count
    pl.DataFrame(rows).write_parquet(data_dir / "derived/kalshi/candles_2025.parquet")
    p = pr.kalshi_priors(2025, {"G": T0, "U": T0}, {"G": (3.0, 44.0), "U": (-2.0, 41.5)})
    g = p["G"]
    assert g.home_win == pytest.approx(0.61) and g.win_source == "kalshi"
    assert g.total == pytest.approx(45.5 + 0.04 / 0.10) and g.total_source == "kalshi"
    assert g.quoted_at == T0 - dt.timedelta(minutes=1)
    u = p["U"]
    assert u.win_source == "line" and u.home_spread == -2.0 and (u.total, u.total_source) == (41.5, "line")


def test_price_log_roundtrip_and_resume(data_dir):
    log = sinks.PriceLog("r")
    assert log.done() == set()
    ev = list(sources.game_events(game(PLAYS)))
    quotes = [eng.Quote.from_scores(np.array([e.state.home_score + 3.0]), np.array([float(e.state.away_score)]))
              for e in ev]
    p = pr.Prior("2025_01_AWAY_HOME", 0.6, 44.0, "kalshi", "kalshi", T0)
    frame = eng.price_frame(ev, quotes, [None] * len(ev), [0.0] * len(ev), p, (3.5, 43.0), {"home_win": 0.6})
    log.write("2025_01_AWAY_HOME", frame)
    assert log.done() == {"2025_01_AWAY_HOME"}
    back = sinks.load("r")
    assert back.height == len(ev) and back["order"].to_list() == list(range(len(ev)))
    assert back["cdf_margin"][0].shape == (eng.MARGIN_GRID.size,) and back["knob_spread"][0] == 3.5
