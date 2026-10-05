"""Simulator rules, checked with a stub drive model whose every drive has the same
result and length, so each game plays out deterministically."""

import numpy as np
import pytest

from gridline.models import drive_model as dm
from gridline.models import simulator as sim
from gridline.models import transitions as tr
from gridline.state.game_state import GameState


def stub_engine(result: str, seconds: float, pat=(0.0, 1.0, 0.0)) -> sim.Engine:
    """Every drive ends in `result` after exactly `seconds`; kicks and changes of
    possession hand the ball over at the receiver's 25 with no clock used."""
    k = dm.END_HALF if result == "end_half" else dm.TIMED.index(result) * dm.N_BINS + 5
    w0, b0 = np.zeros((dm.N_FEATURES, 4), np.float32), np.zeros(4, np.float32)
    w1, b1 = np.zeros((4, dm.N_CLASSES), np.float32), np.full(dm.N_CLASSES, -50.0, np.float32)
    b1[k] = 50.0
    step = dm.StepModel([w0, w1], [b0, b1], 2030, np.full((dm.N_BINS, dm.N_QUANTILES), seconds))

    def pool(n_buckets):
        return tr.Pool(np.zeros(n_buckets, np.int64), np.ones(n_buckets, np.int64), np.array([False]),
                       np.array([75.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))

    pats = np.tile(np.asarray(pat), (len(tr.PAT_DIFFS) * 2, 1))
    return sim.Engine(step, tr.Transitions(pool(3), pool(tr.N_POSSESSION_BUCKETS), np.array([0.0, 1.0, 0.0]),
                                           pats, pool(tr.N_TIMEOUT_BUCKETS)))


def state(season=2025, season_type="REG", **kw) -> GameState:
    base = dict(game_id="G", season=season, season_type=season_type, home="HOME", away="AWAY",
                home_score=0, away_score=0, qtr=1, game_seconds_remaining=3600.0,
                half_seconds_remaining=1800.0, posteam="HOME", down=1, ydstogo=10, yardline_100=75,
                home_timeouts=3, away_timeouts=3, home_receives_2h_kickoff=False,
                pregame_spread=0.0, pregame_total=44.0)
    return GameState(**(base | kw))


def test_touchdown_every_300s_alternates_and_ties_after_both_teams_score_in_overtime():
    # 6 drives a half, alternating possession: 21-21 at the half, 42-42 after regulation.
    # 2025 regular-season overtime (10 min, both teams possess): TD, TD, clock out -> tie.
    r = sim.simulate(stub_engine("td1", 300.0), state(), n=3)
    assert (r.home == 49).all() and (r.away == 49).all() and r.overtime.all()


def test_both_teams_possess_then_sudden_death():
    # 250 s drives: 8 a half (the 8th ends at 0:00), 56-56 after regulation; in overtime
    # TD, TD (63-63), then the third possession's TD ends it at 0:00, with no try.
    r = sim.simulate(stub_engine("td1", 250.0), state(), n=1)
    assert {r.home[0], r.away[0]} == {69.0, 63.0} and r.drives[0] == 19


@pytest.mark.parametrize("season", [2011, 2023])
def test_first_possession_touchdown_ends_older_overtime_without_a_try(season):
    r = sim.simulate(stub_engine("td1", 300.0), state(season=season), n=1)
    assert {r.home[0], r.away[0]} == {48.0, 42.0} and r.overtime[0]


def test_playoff_overtime_plays_on_until_both_teams_have_possessed():
    # one field goal per half: 3-3. Overtime periods are 15 minutes and each field goal
    # drive uses a whole period: FG (6-3), the other team answers in the next period
    # (6-6), then the next score wins.
    r = sim.simulate(stub_engine("fg", 1800.0), state(season=2025, season_type="POST"), n=1)
    assert sorted([r.home[0], r.away[0]]) == [6.0, 9.0] and r.overtime[0] and r.drives[0] == 5
    # the regular season's 10-minute overtime ends when time does, before the answer
    r = sim.simulate(stub_engine("fg", 1800.0), state(season=2025), n=1)
    assert sorted([r.home[0], r.away[0]]) == [3.0, 6.0] and r.drives[0] == 3
    # a period that never produces a score: the regular season can end tied
    reg = sim.simulate(stub_engine("end_half", 0.0), state(season=2025), n=2)
    assert reg.overtime.all() and (reg.margin == 0).all()


def test_untimed_down_at_zero_is_played():
    s = state(qtr=4, game_seconds_remaining=0.0, half_seconds_remaining=0.0, home_score=26, away_score=28,
              yardline_100=27, down=4, ydstogo=3)
    r = sim.simulate(stub_engine("fg", 5.0), s, n=1)
    assert (r.home[0], r.away[0]) == (29.0, 28.0) and not r.overtime[0]


def test_overtime_state_keeps_the_possessions_already_played():
    # 2025 regular-season overtime, 7:00 left: away kicked a field goal on the first
    # possession; home's touchdown now wins it (no try)
    s = state(qtr=5, game_seconds_remaining=420.0, half_seconds_remaining=420.0, home_score=20,
              away_score=23)
    r = sim.simulate(stub_engine("td1", 300.0), s, n=1, ot_possessions=1)
    assert (r.home[0], r.away[0]) == (26.0, 23.0)


def test_halftime_kickoff_goes_to_the_second_half_receiver():
    # each half is a single 1800 s drive ending in a field goal: the team with the ball
    # (home) scores in the 1st half, the 2nd-half receiver in the 2nd
    r = sim.simulate(stub_engine("fg", 1800.0), state(home_receives_2h_kickoff=True), n=1)
    assert (r.home[0], r.away[0]) == (6.0, 0.0) and not r.overtime[0]
    # away receives: 3-3, then one overtime possession (10 min) for the coin-toss winner
    r = sim.simulate(stub_engine("fg", 1800.0), state(home_receives_2h_kickoff=False), n=1)
    assert sorted([r.home[0], r.away[0]]) == [3.0, 6.0] and r.overtime[0]


def test_pat_phase_adds_the_conversion_then_kicks_off():
    eng = stub_engine("end_half", 0.0, pat=(0.0, 0.0, 1.0))  # always a two-point conversion
    s = state(home_score=6, posteam="HOME", qtr=4, game_seconds_remaining=100.0, half_seconds_remaining=100.0)
    r = sim.simulate(eng, s, n=1, phase="pat")
    assert (r.home[0], r.away[0]) == (8.0, 0.0)


def test_opening_coin_toss_and_common_random_numbers():
    eng = stub_engine("fg", 1800.0)
    s = sim.kickoff_state(state())
    u = sim.uniforms(400, seed=3)
    a = sim.simulate(eng, s, 400, u=u, phase="kickoff", kicker_home=None)
    b = sim.simulate(eng, s, 400, u=u, phase="kickoff", kicker_home=None)
    assert np.array_equal(a.home, b.home)
    # whoever receives the opening kick scores in the 1st half, the other in the 2nd,
    # and the overtime coin toss decides who gets the only overtime possession
    assert (a.total == 9).all() and a.overtime.all()
    assert 0.4 < np.mean(a.home == 6) < 0.6


def test_overtime_rules_by_season():
    assert sim.OvertimeRules.for_game(2025, False) == sim.OvertimeRules(600.0, "both", True, 2)
    assert sim.OvertimeRules.for_game(2020, False).mode == "modified"
    assert sim.OvertimeRules.for_game(2016, False).length == 900.0
    assert sim.OvertimeRules.for_game(2021, True).mode == "modified"
    assert sim.OvertimeRules.for_game(2022, True) == sim.OvertimeRules(900.0, "both", False, 3)
