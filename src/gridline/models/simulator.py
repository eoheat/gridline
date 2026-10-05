"""Monte Carlo game simulator (Phase 2).

From a game state, play the rest of the game out N times, one drive at a time:

1. The drive-step model gives every simulated game the joint distribution of how the
   current drive ends and how much clock it uses; one (result, duration) is drawn.
2. The result is scored and the clock runs. If the half ends, the next half starts
   with a kickoff; if regulation ends tied, overtime follows that season's rules.
3. Otherwise what happens before the next snap is drawn from history (transitions.py):
   a kick after a score, or a change of possession.
4. Repeat until every game is over.

All N games advance together as numpy arrays, and finished games drop out. Every
random draw is read from a fixed table of uniforms indexed by (game, drive step), so
consecutive states of a real game reuse the same simulated "luck" (common random
numbers): prices then move because the game moved, not because of the dice.

The simulation is a function of the game state plus two numbers per game, the spread
and total fed to the drive model. solve_knobs() picks them so that at kickoff the
simulated margin and total match a target (DESIGN.md decision 3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from gridline.models import drive_model as dm
from gridline.models import transitions as tr
from gridline.models.drives import RESULTS
from gridline.state.game_state import GameState

MAX_STEPS = 72  # drive steps with their own uniforms; longer games reuse them cyclically
N_UNIFORMS = 6  # per step: class, duration, conversion, transition, period end, timeouts
R = {r: i for i, r in enumerate(RESULTS)}
OFF_POINTS = np.zeros(len(RESULTS), dtype=np.int64)
for _r, _p in (("td0", 6), ("td1", 7), ("td2", 8), ("fg", 3)):
    OFF_POINTS[R[_r]] = _p
IS_TD = np.isin(np.arange(len(RESULTS)), [R["td0"], R["td1"], R["td2"]])
KICKS_AFTER = np.isin(np.arange(len(RESULTS)), [R[r] for r in ("td0", "td1", "td2", "fg", "opp_td", "safety")])
POSSESSION_IDX = np.full(len(RESULTS), -1)
for _i, _r in enumerate(tr.POSSESSION_RESULTS):
    POSSESSION_IDX[R[_r]] = _i

Phase = Literal["snap", "kickoff", "pat"]


@dataclass(frozen=True)
class OvertimeRules:
    length: float  # seconds in an overtime period
    mode: str  # "sudden": any score wins; "modified": a first-possession FG gets an
    #            answer; "both": each team possesses at least once
    can_tie: bool
    timeouts: int

    @classmethod
    def for_game(cls, season: int, postseason: bool) -> "OvertimeRules":
        if postseason:
            mode = "both" if season >= 2022 else "modified" if season >= 2010 else "sudden"
            return cls(900.0, mode, False, 3)
        mode = "both" if season >= 2024 else "modified" if season >= 2012 else "sudden"
        return cls(600.0 if season >= 2017 else 900.0, mode, True, 2)


@dataclass
class Engine:
    step: dm.StepModel
    transitions: tr.Transitions

    @classmethod
    def load(cls, last_season: int) -> "Engine":
        return cls(dm.StepModel.load(dm.step_model_path(last_season)),
                   tr.Transitions.load(tr.transitions_path(last_season)))


def uniforms(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(per-step uniforms [n, MAX_STEPS, N_UNIFORMS], start-phase uniforms [n, 4])."""
    rng = np.random.default_rng(seed)
    return rng.random((n, MAX_STEPS, N_UNIFORMS)), rng.random((n, 4))


@dataclass
class Result:
    home: np.ndarray  # final scores, one per simulated game
    away: np.ndarray
    overtime: np.ndarray  # bool: the game went to overtime
    drives: np.ndarray  # drive steps simulated

    @property
    def margin(self) -> np.ndarray:
        return self.home - self.away

    @property
    def total(self) -> np.ndarray:
        return self.home + self.away


class _Games:
    """State of n simulated games (home-team view)."""

    def __init__(self, s: GameState, n: int, knobs: tuple[float, float], rules: OvertimeRules):
        self.n = n
        self.season = s.season
        self.post = s.season_type == "POST"
        self.rules = rules
        self.spread, self.total = knobs
        self.home = np.full(n, s.home_score, dtype=np.int64)
        self.away = np.full(n, s.away_score, dtype=np.int64)
        self.pos_home = np.full(n, s.posteam == s.home if s.posteam else True)
        self.yardline = np.full(n, float(s.yardline_100 or 75))
        self.down = np.full(n, int(s.down or 1))
        self.togo = np.full(n, float(s.ydstogo or 10))
        self.half = np.full(n, 2 if s.qtr >= 5 else (1 if s.qtr >= 3 else 0), dtype=np.int64)
        self.clock = np.full(n, float(s.game_seconds_remaining if s.qtr >= 5 else s.half_seconds_remaining))
        self.to_home = np.full(n, int(s.home_timeouts))
        self.to_away = np.full(n, int(s.away_timeouts))
        self.recv2h_home = np.full(n, bool(s.home_receives_2h_kickoff) if s.home_receives_2h_kickoff is not None else False)
        self.active = np.ones(n, dtype=bool)
        self.overtime = self.half == 2
        self.ot_poss = np.zeros(n, dtype=np.int64)  # overtime possessions completed
        self.gap = np.zeros(n, dtype=np.int64)  # points scored between the last drive and the next snap
        self.steps = np.zeros(n, dtype=np.int64)


def simulate(engine: Engine, s: GameState, n: int = 10_000, knobs: tuple[float, float] | None = None,
             u: tuple[np.ndarray, np.ndarray] | None = None, phase: Phase = "snap",
             kicker_home: bool | None = None, seed: int = 0, ot_possessions: int = 0,
             free_kick: bool = False) -> Result:
    """Final scores of n simulated games from state `s`.

    phase "snap": a scrimmage snap (s.posteam, down, distance, yard line).
    phase "kickoff": a kick is about to happen; kicker_home=None means a coin toss
      (before the opening kickoff), which also decides who receives in the 2nd half.
      free_kick: the kick is the free kick after a safety.
    phase "pat": s.posteam has just scored a touchdown; the conversion comes next.
    knobs: (home spread, total) fed to the drive model; default: the pregame line.
    ot_possessions: for a state in overtime, the possessions already completed in it
      (the overtime rules depend on them; see overtime_possessions()).
    """
    if knobs is None:
        knobs = (s.pregame_spread if s.pregame_spread is not None else 0.0,
                 s.pregame_total if s.pregame_total is not None else dm.TOTAL_BASE)
    steps_u, start_u = u if u is not None else uniforms(n, seed)
    steps_u, start_u = steps_u[:n], start_u[:n]
    g = _Games(s, n, knobs, OvertimeRules.for_game(s.season, s.season_type == "POST"))
    if s.qtr >= 5:
        g.ot_poss[:] = ot_possessions
    if s.home_receives_2h_kickoff is None and g.half[0] == 0:
        # who receives in the 2nd half is not in the state: at the opening kickoff it is
        # the kicking team; otherwise the coin toss is unknown and is drawn
        opening = (phase == "kickoff" and kicker_home is not None and s.home_score == 0 and s.away_score == 0
                   and s.half_seconds_remaining >= 1800)
        g.recv2h_home = np.full(n, bool(kicker_home)) if opening else start_u[:, 2] < 0.5
    all_idx = np.arange(n)
    if phase == "snap":
        # a snap at 0:00 is an untimed down (after a defensive penalty on the last play):
        # it is played, as the period's last play
        g.clock = np.maximum(g.clock, 1.0)
    if phase == "pat":
        scorer_home = np.full(n, s.posteam == s.home)
        lead = np.where(scorer_home, g.home - g.away, g.away - g.home)
        late = (g.half >= 1) & (g.clock <= 900)
        p = engine.transitions.pat[tr.pat_context(lead, late)]
        extra = (start_u[:, 0][:, None] > np.cumsum(p, axis=1)).sum(1)
        g.home += np.where(scorer_home, extra, 0)
        g.away += np.where(scorer_home, 0, extra)
        if s.qtr >= 5:  # the scoring drive is over (ot_possessions did not count it): did it end overtime?
            _overtime_after_possession(g, all_idx, defensive_score=np.zeros(n, bool), offense_td=np.ones(n, bool))
        timed = g.active & (g.clock > 0)  # no kickoff after a try with no time left, or after the game
        _kick(engine, g, all_idx[timed], kicker_home=scorer_home[timed], u=start_u[timed, 1])
    elif phase == "kickoff":
        if kicker_home is None:  # opening kickoff after a coin toss
            kick_home = start_u[:, 2] < 0.5
            g.recv2h_home = kick_home.copy()  # the opening kicker receives in the 2nd half
        else:
            kick_home = np.full(n, kicker_home)
        ctx = np.full(n, tr.KICK_FREE) if free_kick else None
        _kick(engine, g, all_idx, kicker_home=kick_home, u=start_u[:, 1], context=ctx)
    _run(engine, g, steps_u)
    return Result(g.home.astype(float), g.away.astype(float), g.overtime, g.steps)


def _kick(engine: Engine, g: _Games, idx: np.ndarray, kicker_home: np.ndarray, u: np.ndarray,
          context: np.ndarray | None = None) -> None:
    """A kick by `kicker_home` for games idx: sets possession, yard line and clock for the
    next snap and adds any points scored in between."""
    if idx.size == 0:
        return
    if context is None:
        lead = np.where(kicker_home, g.home[idx] - g.away[idx], g.away[idx] - g.home[idx])
        late = (g.half[idx] == 1) & (g.clock[idx] <= tr.ONSIDE_SECONDS)
        context = np.where(late & (lead < 0), tr.KICK_ONSIDE, tr.KICK_NORMAL)
    pool = engine.transitions.kicks
    row = pool.draw(context, u)
    to_kicker = pool.points_a[row].astype(np.int64)
    to_receiver = pool.points_b[row].astype(np.int64)
    g.home[idx] += np.where(kicker_home, to_kicker, to_receiver)
    g.away[idx] += np.where(kicker_home, to_receiver, to_kicker)
    g.gap[idx] = to_kicker + to_receiver
    g.pos_home[idx] = np.where(pool.same[row], kicker_home, ~kicker_home)
    _new_series(g, idx, pool.yardline[row])
    g.clock[idx] -= np.nan_to_num(pool.seconds[row])


def _new_series(g: _Games, idx: np.ndarray, yardline: np.ndarray) -> None:
    y = np.clip(yardline, 1, 99)
    g.yardline[idx] = y
    g.down[idx] = 1
    g.togo[idx] = np.minimum(10, y)


def _run(engine: Engine, g: _Games, u: np.ndarray, max_steps: int = 400) -> None:
    step = 0
    while True:
        idx = np.flatnonzero(g.active)
        if step >= max_steps:  # never reached in practice; a guard against a stuck clock
            g.active[idx] = False
            return
        # a kick at the end of a period can leave no time: resolve those first
        expired = idx[g.clock[idx] <= 0]
        if expired.size:
            _period_over(engine, g, expired, u[expired, step % MAX_STEPS, 4])
            idx = np.flatnonzero(g.active)
        if idx.size == 0:
            return
        _drive(engine, g, idx, u[idx, step % MAX_STEPS])
        step += 1


def _drive(engine: Engine, g: _Games, idx: np.ndarray, u: np.ndarray) -> None:
    """One drive for every game in idx: draw its result and clock, score it, then set
    up the next snap (or end the period)."""
    g.steps[idx] += 1
    g.gap[idx] = 0
    ph = g.pos_home[idx]
    diff = np.where(ph, g.home[idx] - g.away[idx], g.away[idx] - g.home[idx])
    x = dm.features(
        g.yardline[idx], g.down[idx], g.togo[idx], g.clock[idx], g.half[idx] >= 1, diff,
        np.where(ph, g.to_home[idx], g.to_away[idx]), np.where(ph, g.to_away[idx], g.to_home[idx]),
        np.where(ph, g.spread, -g.spread), np.full(idx.size, g.total),
        np.full(idx.size, min(g.season, engine.step.season_cap)), np.full(idx.size, g.post))
    k = engine.step.sample(x, u[:, 0])
    result = dm.CLASS_RESULT[k]
    b = np.maximum(dm.CLASS_BIN[k], 0)
    clock = g.clock[idx]
    used = np.where(k == dm.END_HALF, clock, engine.step.duration(b, u[:, 1]))
    g.clock[idx] = clock - np.minimum(used, clock)

    # points: the offence's, then the defence's (a safety, or a return for a touchdown)
    off = OFF_POINTS[result]
    in_ot = g.half[idx] == 2
    if in_ot.any():  # a touchdown that wins in overtime is not followed by a try
        td = in_ot & IS_TD[result]
        if td.any():
            ti = idx[td]
            six = np.where(ph[td], 6, 0)
            home6, away6 = g.home[ti] + six, g.away[ti] + 6 - six
            ahead = np.where(ph[td], home6 > away6, away6 > home6)  # behind or level: the try matters
            ends = ahead & _ot_decided(g, home6, away6, g.ot_poss[ti] + 1,
                                       defensive_score=np.zeros(ti.size, bool),
                                       offense_td=np.ones(ti.size, bool))
            off[td] = np.where(ends, 6, off[td])
    conv = (u[:, 2:3] > np.cumsum(engine.transitions.opp_td_conversion)).sum(1)
    dfn = np.where(result == R["safety"], 2, np.where(result == R["opp_td"], 6 + conv, 0))
    g.home[idx] += np.where(ph, off, dfn)
    g.away[idx] += np.where(ph, dfn, off)

    # overtime: does this possession end the game?
    if in_ot.any():
        _overtime_after_possession(g, idx[in_ot], defensive_score=dfn[in_ot] > 0,
                                   offense_td=IS_TD[result[in_ot]])

    live = g.active[idx] & (g.clock[idx] > 0) & (k != dm.END_HALF)
    # timeouts the two teams use before the next snap (the pool is keyed by the drive's
    # starting clock and score)
    if live.any():
        li = idx[live]
        pool = engine.transitions.timeouts
        row = pool.draw(tr.timeout_bucket(clock[live], diff[live], g.half[li] >= 1), u[live, 5])
        used_off, used_def = pool.points_a[row].astype(np.int64), pool.points_b[row].astype(np.int64)
        oh = ph[live]
        g.to_home[li] = np.maximum(g.to_home[li] - np.where(oh, used_off, used_def), 0)
        g.to_away[li] = np.maximum(g.to_away[li] - np.where(oh, used_def, used_off), 0)
    # scores (and safeties) are followed by a kick; the kicker is the team that scored,
    # or after a safety the team that gave it up (a free kick)
    kick = live & KICKS_AFTER[result]
    if kick.any():
        ki = idx[kick]
        scored_by_offense = OFF_POINTS[result[kick]] > 0
        safety = result[kick] == R["safety"]
        kicker_home = np.where(scored_by_offense | safety, ph[kick], ~ph[kick])
        ctx = None
        if safety.any():
            lead = np.where(kicker_home, g.home[ki] - g.away[ki], g.away[ki] - g.home[ki])
            late = (g.half[ki] == 1) & (g.clock[ki] <= tr.ONSIDE_SECONDS)
            ctx = np.where(safety, tr.KICK_FREE, np.where(late & (lead < 0), tr.KICK_ONSIDE, tr.KICK_NORMAL))
        _kick(engine, g, ki, kicker_home, u[kick, 3], context=ctx)
    change = live & (POSSESSION_IDX[result] >= 0)
    if change.any():
        ci = idx[change]
        pool = engine.transitions.possession
        bucket = tr.possession_bucket(POSSESSION_IDX[result[change]], g.yardline[ci], g.down[ci])
        row = pool.draw(bucket, u[change, 3])
        off_home = ph[change]
        g.home[ci] += np.where(off_home, pool.points_a[row], pool.points_b[row]).astype(np.int64)
        g.away[ci] += np.where(off_home, pool.points_b[row], pool.points_a[row]).astype(np.int64)
        g.gap[ci] = (pool.points_a[row] + pool.points_b[row]).astype(np.int64)
        g.pos_home[ci] = np.where(pool.same[row], off_home, ~off_home)
        _new_series(g, ci, pool.yardline[row])
        g.clock[ci] -= np.nan_to_num(pool.seconds[row])
    # points between drives (a return for a touchdown) can also end overtime
    _overtime_after_return(g, idx[(g.half[idx] == 2) & g.active[idx]])

    done = g.active[idx] & ((g.clock[idx] <= 0) | (k == dm.END_HALF))
    if done.any():
        _period_over(engine, g, idx[done], u[done, 4])


def _ot_decided(g: _Games, home: np.ndarray, away: np.ndarray, possessions: np.ndarray,
                defensive_score: np.ndarray, offense_td: np.ndarray) -> np.ndarray:
    """Whether overtime is over once a possession ends with these scores, `possessions`
    counting that one."""
    differ = home != away
    mode = g.rules.mode
    if mode == "sudden":
        return differ
    if mode == "modified":  # a first-possession TD wins; a field goal gets an answer
        return differ & ((possessions >= 2) | defensive_score | (offense_td & (possessions == 1)))
    return differ & ((possessions >= 2) | defensive_score)  # both teams possess


def _overtime_after_possession(g: _Games, idx: np.ndarray, defensive_score: np.ndarray,
                               offense_td: np.ndarray) -> None:
    g.ot_poss[idx] += 1
    over = _ot_decided(g, g.home[idx], g.away[idx], g.ot_poss[idx], defensive_score, offense_td)
    g.active[idx[over]] = False


def _overtime_after_return(g: _Games, idx: np.ndarray) -> None:
    """A kick or punt returned for a touchdown ends overtime in sudden death, and under
    the modified rules at any time (a touchdown always wins there)."""
    if idx.size == 0:
        return
    differ = g.home[idx] != g.away[idx]
    mode = g.rules.mode
    if mode == "sudden":
        over = differ
    elif mode == "modified":
        over = differ & ((g.ot_poss[idx] >= 2) | (g.gap[idx] >= 6))
    else:
        over = differ & (g.ot_poss[idx] >= 2)
    g.active[idx[over]] = False


def _period_over(engine: Engine, g: _Games, idx: np.ndarray, u: np.ndarray) -> None:
    """The half (or overtime period) has ended for games idx."""
    h = g.half[idx].copy()
    first, u_first = idx[h == 0], u[h == 0]
    reg, u_reg = idx[h == 1], u[h == 1]
    ot, u_ot = idx[h == 2], u[h == 2]
    if first.size:  # halftime: the other team kicks to the 2nd-half receiver
        g.half[first] = 1
        g.clock[first] = 1800.0
        g.to_home[first] = 3
        g.to_away[first] = 3
        _kick(engine, g, first, kicker_home=~g.recv2h_home[first], u=u_first,
              context=np.full(first.size, tr.KICK_NORMAL))
    if reg.size:  # end of regulation
        tied = g.home[reg] == g.away[reg]
        g.active[reg[~tied]] = False
        _start_overtime(engine, g, reg[tied], u_reg[tied])
    if ot.size:
        if g.rules.can_tie:  # regular season: time runs out, whatever the score
            g.active[ot] = False
        else:  # playoffs: the game goes on into another period (not decided yet, or the
            # possession rules would have ended it)
            _start_overtime(engine, g, ot, u_ot, new_game=False)


def _start_overtime(engine: Engine, g: _Games, idx: np.ndarray, u: np.ndarray,
                    new_game: bool = True) -> None:
    if idx.size == 0:
        return
    g.half[idx] = 2
    g.overtime[idx] = True
    g.clock[idx] = g.rules.length
    g.to_home[idx] = g.rules.timeouts
    g.to_away[idx] = g.rules.timeouts
    if new_game:
        g.ot_poss[idx] = 0
        kicker_home = u < 0.5  # coin toss
    else:  # a later playoff period: the team that had the ball hands it over
        kicker_home = g.pos_home[idx].copy()
    _kick(engine, g, idx, kicker_home, u=(u * 2) % 1.0, context=np.full(idx.size, tr.KICK_NORMAL))
    _overtime_after_return(g, idx)  # the opening kick returned for a touchdown


def overtime_possessions(rows, seq: int) -> int:
    """Overtime possessions completed before events row `seq`, from one game's events rows
    (a polars frame): the runs of scrimmage snaps by one team, not counting the drive in
    progress."""
    import polars as pl

    snaps = rows.filter((pl.col("qtr") >= 5) & (pl.col("seq") < seq) & pl.col("down").is_not_null()
                        & pl.col("posteam").is_not_null()).sort("seq")["posteam"].to_list()
    runs = sum(1 for i, t in enumerate(snaps) if i == 0 or t != snaps[i - 1])
    now = rows.filter(pl.col("seq") == seq)["posteam"].to_list()
    if snaps and now and snaps[-1] == now[0]:
        runs -= 1  # that team's drive is still going
    return runs


# ------------------------------------------------------------------ prior matching

WIN_TOL = 0.002  # solve_knobs: how close the simulated win price must get to its target


def median_total(totals: np.ndarray) -> float:
    """The total with a 50% chance of being exceeded, read off the half-point ladder
    (P(total > k + 0.5) for every integer k) by linear interpolation between the two
    rungs that straddle 50%: the same way it is read off Kalshi's total ladder, and a
    smooth function of the knobs, unlike the plain median of integer scores."""
    k = np.arange(0, 161) + 0.5
    p = 1.0 - np.searchsorted(np.sort(totals), k, side="right") / totals.size
    return ladder_median(k, p)


def ladder_median(strikes: np.ndarray, p_over: np.ndarray) -> float:
    """Strike at which P(over) crosses 50%, interpolating between neighbouring rungs.
    P(over) must fall with the strike; a non-monotone ladder (quotes are noisy) is made
    monotone first by isotonic regression. NaN if no two rungs straddle 50%."""
    from sklearn.isotonic import isotonic_regression

    order = np.argsort(strikes)
    k = np.asarray(strikes, dtype=float)[order]
    p = isotonic_regression(np.asarray(p_over, dtype=float)[order], increasing=False)
    above = np.flatnonzero(p >= 0.5)
    if above.size == 0 or above[-1] == k.size - 1:
        return float("nan")
    i = above[-1]
    lo, hi = p[i], p[i + 1]
    return float(k[i] + (lo - 0.5) / (lo - hi) * (k[i + 1] - k[i])) if lo > hi else float(k[i])


def solve_knobs(engine: Engine, s: GameState, target_margin: float | None, target_total: float,
                n: int = 10_000, seed: int = 0, tol: float = 0.05, max_iter: int = 12,
                phase: Phase = "kickoff", kicker_home: bool | None = None,
                target_home_win: float | None = None,
                u: tuple[np.ndarray, np.ndarray] | None = None,
                total_stat: str = "mean", keep_best: bool = False,
                info: dict | None = None) -> tuple[float, float]:
    """(spread, total) inputs such that, simulating from state s, the total hits
    target_total and either the mean margin hits target_margin or the home win price
    (ties count half) hits target_home_win (within WIN_TOL). total_stat says which
    total: "mean", or "median" (median_total(), for a target read off a total ladder).

    The same uniforms (`u`, or uniforms(n, seed)) are used on every iteration, so the
    targets move almost smoothly with the knobs. Almost: a changed knob changes some
    simulated drives, and everything after them, so at small n the targets jitter and
    the iteration may not settle. keep_best returns the iterate closest to the targets
    rather than the last update (the default, kept so the Phase 2 gate reproduces).
    `info`, if given, receives what the returned knobs achieve."""
    u = u if u is not None else uniforms(n, seed)
    stat = {"mean": np.mean, "median": median_total}[total_stat]
    spread = target_margin if target_margin is not None else 0.0
    total = target_total
    best, best_err, achieved, it = (spread, total), np.inf, {}, 0
    for it in range(1, max_iter + 1):
        r = simulate(engine, s, n, (spread, total), u, phase=phase, kicker_home=kicker_home)
        dt_ = target_total - stat(r.total)
        win = float(np.mean(r.margin > 0) + 0.5 * np.mean(r.margin == 0))
        if target_home_win is None:
            ds = target_margin - r.margin.mean()
            err_s = abs(ds) / tol
        else:
            # near a pick'em a point of spread moves the win price by about 3 points
            ds = float(np.clip((target_home_win - win) / 0.03, -7, 7))
            err_s = abs(target_home_win - win) / WIN_TOL
        err = max(err_s, abs(dt_) / tol)
        if err < best_err:
            best, best_err = (spread, total), err
            achieved = {"home_win": win, "total": float(stat(r.total)), "margin_mean": float(r.margin.mean())}
        if err < 1:
            break
        spread += ds
        total += dt_
    if info is not None:
        info.update(achieved, iterations=it, converged=bool(best_err < 1))
    return best if keep_best or best_err < 1 else (spread, total)


def kickoff_state(s: GameState) -> GameState:
    """The state before the opening kickoff of s's game (for pregame pricing)."""
    return replace(s, home_score=0, away_score=0, qtr=1, game_seconds_remaining=3600.0,
                   half_seconds_remaining=1800.0, posteam=None, down=None, ydstogo=None,
                   yardline_100=None, home_timeouts=3, away_timeouts=3,
                   home_receives_2h_kickoff=None)
