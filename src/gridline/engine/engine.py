"""The pricing engine: events in, prices out (DESIGN.md decision 1).

For one game:

1. Before kickoff, the two knobs (simulator.solve_knobs) are solved so that the
   simulated pregame home win price and median total match the prior (prior.py).
2. At every event (sources.py) the simulator plays the rest of the game out N_SIMS
   times from the new state, with the game's own fixed table of uniforms (common random
   numbers), and the engine publishes a Quote: the joint distribution of the final score
   summarised as the CDFs of the final margin and total. Every Kalshi contract is priced
   off it (Quote.price).
3. The direct win-probability model (Phase 1) prices the same states wherever it can
   (a scrimmage snap comes next), as a second opinion.

GamePricer.on_event is the live path; replay() runs every game of a season through it
and writes the price log (sinks.py). Nothing here reads a market price after kickoff.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from gridline.engine import sinks, sources
from gridline.engine.prior import Prior
from gridline.eval import distribution as dist
from gridline.models import simulator as sim

Log = Callable[[str], None]

N_SIMS = 2000  # simulated games per event
N_KNOB = 10_000  # simulated games when solving the knobs
MARGIN_GRID = np.arange(-60, 61)
TOTAL_GRID = np.arange(0, 121)


def game_seed(game_id: str) -> int:
    return zlib.crc32(game_id.encode())


@dataclass(frozen=True)
class Quote:
    """The engine's price of every contract of a game, at one moment."""

    p_home: float  # fair YES price of the home winner contract (ties count half)
    p_tie: float
    margin_mean: float
    total_mean: float
    cdf_margin: np.ndarray  # P(final home margin <= z) for z in MARGIN_GRID
    cdf_total: np.ndarray  # P(final total <= z) for z in TOTAL_GRID

    @classmethod
    def from_scores(cls, home: np.ndarray, away: np.ndarray) -> "Quote":
        m, t = home - away, home + away
        return cls(float(np.mean(m > 0) + 0.5 * np.mean(m == 0)), float(np.mean(m == 0)),
                   float(m.mean()), float(t.mean()),
                   dist.cdf_from_samples(m, MARGIN_GRID).astype(np.float32),
                   dist.cdf_from_samples(t, TOTAL_GRID).astype(np.float32))

    def price(self, kind: str, side: str | None, strike: float | None) -> float:
        """Fair YES price of a Kalshi contract (DESIGN.md decision 4)."""
        if kind == "winner":
            return self.p_home if side == "home" else 1.0 - self.p_home
        k = int(np.floor(strike))  # strikes are N.5: "over N.5" is ">= N + 1"
        if kind == "spread":
            if side == "home":  # P(margin > k.5) = 1 - F(k)
                return 1.0 - _at(self.cdf_margin, MARGIN_GRID, k)
            return _at(self.cdf_margin, MARGIN_GRID, -k - 1)  # P(margin <= -(k + 1))
        if kind == "total":
            return 1.0 - _at(self.cdf_total, TOTAL_GRID, k)
        raise ValueError(f"unknown contract kind {kind!r}")


def _at(cdf: np.ndarray, grid: np.ndarray, z: int) -> float:
    if z < grid[0]:
        return 0.0
    if z > grid[-1]:
        return 1.0
    return float(cdf[z - grid[0]])


class GamePricer:
    """Prices one game, event by event, from a prior fixed at kickoff.

    The knobs are solved with N_KNOB simulated games on their own uniforms: at the
    2,000 games used per event, the simulated win price jitters by about a point as the
    knobs move, too much to match the prior to WIN_TOL. So the pregame price the
    engine publishes carries the same Monte Carlo noise as every other price
    (about +-1 point at 50%), around a prior matched to within 0.2 points."""

    def __init__(self, engine: sim.Engine, prior: Prior, pregame: sources.Event, n: int = N_SIMS,
                 n_knob: int = N_KNOB):
        assert pregame.kind == "pregame"
        self.engine, self.prior, self.n = engine, prior, n
        self.u = sim.uniforms(n, game_seed(pregame.game_id))
        self.knob_info: dict = {}
        kw = dict(n=n_knob, seed=game_seed(pregame.game_id) + 1, phase="kickoff", kicker_home=pregame.kicker_home,
                  total_stat="median", keep_best=True, info=self.knob_info)
        if prior.win_source == "kalshi":
            self.knobs = sim.solve_knobs(engine, pregame.state, None, prior.total,
                                         target_home_win=prior.home_win, **kw)
        else:  # no winner quote: match the sportsbook spread instead
            self.knobs = sim.solve_knobs(engine, pregame.state, prior.home_spread or 0.0, prior.total, **kw)

    def on_event(self, e: sources.Event) -> Quote:
        s = e.state
        if e.kind == "final":
            return Quote.from_scores(np.array([float(s.home_score)]), np.array([float(s.away_score)]))
        r = sim.simulate(self.engine, s, self.n, self.knobs, self.u, phase=e.phase,
                         kicker_home=e.kicker_home, ot_possessions=e.ot_possessions, free_kick=e.free_kick)
        return Quote.from_scores(r.home, r.away)


# ------------------------------------------------------------ the direct model


def _direct_rows(rows: pl.DataFrame, events: list[sources.Event]) -> tuple[list[int], pl.DataFrame]:
    """Events a scrimmage snap follows, as events-table rows the Phase 1 features accept.
    The game's first play goes first, as it does in the table, because nflfastR's
    receive_2h_ko reads the first possession team."""
    idx = [i for i, e in enumerate(events) if e.kind in ("play", "snap") and e.phase == "snap"
           and e.state.down is not None and e.state.posteam is not None and e.state.qtr <= 4]
    first = rows.filter(pl.col("posteam").is_not_null()).sort("seq").head(1)
    final_margin = first["final_margin_home"][0]
    pseudo = [{
        "game_id": e.game_id, "season": e.state.season, "seq": 1_000_000 + j, "qtr": e.state.qtr,
        "game_seconds_remaining": e.state.game_seconds_remaining,
        "half_seconds_remaining": e.state.half_seconds_remaining, "posteam": e.state.posteam,
        "home_team": e.state.home, "away_team": e.state.away, "down": e.state.down,
        "ydstogo": e.state.ydstogo, "yardline_100": e.state.yardline_100,
        "home_timeouts": e.state.home_timeouts, "away_timeouts": e.state.away_timeouts,
        "home_score_pre": e.state.home_score, "away_score_pre": e.state.away_score,
        "pregame_spread": e.state.pregame_spread, "final_margin_home": final_margin,
    } for j, e in enumerate(events[i] for i in idx)]
    cols = list(pseudo[0]) if pseudo else list(first.columns)
    frame = pl.concat([first.select(cols), pl.DataFrame(pseudo, schema=first.select(cols).schema)])
    return idx, frame


def direct_wp(booster, rows: pl.DataFrame, events: list[sources.Event]) -> list[float | None]:
    """The Phase 1 model's home win probability at each event (None where it has no
    snap to price: kicks, tries, the pregame and the final whistle, overtime)."""
    from gridline.models import features, wp

    out: list[float | None] = [None] * len(events)
    idx, frame = _direct_rows(rows, events)
    if not idx:
        return out
    f = features.wp_features(frame).filter(pl.col("seq") >= 1_000_000).sort("seq")
    p = wp.predict(booster, f, "wp_spread")
    for i, pi, team in zip(idx, p, f["posteam"].to_list()):
        out[i] = float(pi) if team == events[i].state.home else float(1 - pi)
    return out


# ------------------------------------------------------------ replay


def price_frame(events: list[sources.Event], quotes: list[Quote], wp_direct: list[float | None],
                seconds: list[float], prior: Prior, knobs: tuple[float, float],
                knob_info: dict | None = None, n_sims: int = N_SIMS) -> pl.DataFrame:
    """Price-log rows for one game (n_sims: simulated games behind each price; 0 for the
    final whistle, which is not simulated)."""
    st = [e.state for e in events]
    return pl.DataFrame({
        "game_id": [e.game_id for e in events],
        "order": np.arange(len(events), dtype=np.int32),
        "seq": [e.seq for e in events],
        "kind": [e.kind for e in events],
        "t": pl.Series([e.t for e in events], dtype=pl.Datetime("us", "UTC")),
        "timing": [e.timing for e in events],
        "t_play_end": pl.Series([e.t_play_end for e in events], dtype=pl.Datetime("us", "UTC")),
        "phase": [e.phase for e in events],
        "qtr": [s.qtr for s in st],
        "game_seconds_remaining": [s.game_seconds_remaining for s in st],
        "home_score": [s.home_score for s in st],
        "away_score": [s.away_score for s in st],
        "posteam_is_home": [s.posteam_is_home for s in st],
        "down": pl.Series([s.down for s in st], dtype=pl.Int8),
        "ydstogo": pl.Series([s.ydstogo for s in st], dtype=pl.Int16),
        "yardline_100": pl.Series([s.yardline_100 for s in st], dtype=pl.Int16),
        "home_timeouts": [s.home_timeouts for s in st],
        "away_timeouts": [s.away_timeouts for s in st],
        "p_home": [q.p_home for q in quotes],
        "p_tie": [q.p_tie for q in quotes],
        "margin_mean": [q.margin_mean for q in quotes],
        "total_mean": [q.total_mean for q in quotes],
        "cdf_margin": pl.Series([q.cdf_margin for q in quotes], dtype=pl.List(pl.Float32)),
        "cdf_total": pl.Series([q.cdf_total for q in quotes], dtype=pl.List(pl.Float32)),
        "wp_direct_home": pl.Series(wp_direct, dtype=pl.Float64),
        "seconds": seconds,
        "n_sims": np.where(np.array([e.kind for e in events]) == "final", 0, n_sims).astype(np.int32),
        "prior_home_win": prior.home_win,
        "prior_total": prior.total,
        "prior_win_source": prior.win_source,
        "prior_total_source": prior.total_source,
        "knob_spread": knobs[0],
        "knob_total": knobs[1],
        # what the knobs achieve at kickoff with N_KNOB simulated games
        "knob_home_win": (knob_info or {}).get("home_win"),
        "knob_total_median": (knob_info or {}).get("total"),
        "knob_converged": (knob_info or {}).get("converged"),
    })


_CACHE: dict = {}


def _game_task(args) -> tuple[str, int, float]:
    rows, prior, engine_season, wp_path, n, run, bounds = args
    import xgboost as xgb

    if ("engine", engine_season) not in _CACHE:
        _CACHE[("engine", engine_season)] = sim.Engine.load(engine_season)
    if ("wp", wp_path) not in _CACHE:
        b = xgb.Booster()
        b.load_model(wp_path)
        _CACHE[("wp", wp_path)] = b
    engine, booster = _CACHE[("engine", engine_season)], _CACHE[("wp", wp_path)]
    events = list(sources.game_events(rows, bounds))
    t0 = time.time()
    pricer = GamePricer(engine, prior, events[0], n)
    quotes, seconds = [], []
    for e in events:
        t1 = time.perf_counter()
        quotes.append(pricer.on_event(e))
        seconds.append(time.perf_counter() - t1)
    frame = price_frame(events, quotes, direct_wp(booster, rows, events), seconds, prior, pricer.knobs,
                        pricer.knob_info, n)
    sinks.PriceLog(run).write(events[0].game_id, frame)
    return events[0].game_id, len(events), time.time() - t0


def run_name(season: int) -> str:
    return f"replay_{season}"


def replay(season: int, engine_season: int, wp_path: str | Path, n: int = N_SIMS, processes: int = 2,
           games: int | None = None, run: str | None = None, log: Log = print) -> str:
    """Price every event of a season's games and write the price log. Games already in
    the log are skipped, so an interrupted run picks up where it stopped."""
    from gridline.engine.prior import kalshi_priors
    from gridline.inventory import load_events

    run = run or run_name(season)
    ev = load_events(season)
    # how long plays last, for plays the feed gives no end time: from the seasons before
    bounds = sources.end_bounds(pl.concat([load_events(s) for s in range(season - 3, season)], how="diagonal_relaxed"))
    src = sources.ReplaySource(ev, bounds)
    ids = src.game_ids[:games]
    kickoffs, lines = {}, {}
    for g in ids:
        rows = src.rows(g)
        first = next(sources.game_events(rows, bounds), None)
        if first is None:
            continue
        kickoffs[g] = first.t
        lines[g] = (first.state.pregame_spread, first.state.pregame_total)
    priors = kalshi_priors(season, kickoffs, lines)
    n_kalshi = sum(p.win_source == "kalshi" for p in priors.values())
    n_total = sum(p.total_source == "kalshi" for p in priors.values())
    log(f"{len(priors)} games; prior from Kalshi: winner {n_kalshi}, total {n_total} "
        f"(the rest from the sportsbook line)")
    done = sinks.PriceLog(run).done()
    todo = [g for g in kickoffs if g not in done]
    if len(todo) < len(kickoffs):
        log(f"  {len(kickoffs) - len(todo)} games already priced in {sinks.run_dir(run)}")
    tasks = [(src.rows(g), priors[g], engine_season, str(wp_path), n, run, bounds) for g in todo]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # one BLAS thread per worker: the workers already use every core
    t0, n_events = time.time(), 0
    ctx = mp.get_context("spawn")  # polars and OpenMP thread pools do not survive fork()
    with ctx.Pool(processes) as pool:
        for i, (gid, k, secs) in enumerate(pool.imap_unordered(_game_task, tasks), 1):
            n_events += k
            if i % 10 == 0 or i == len(tasks):
                log(f"  {i}/{len(tasks)} games, {n_events:,} events, {time.time() - t0:.0f}s")
    log(f"  price log: {sinks.run_dir(run)}")
    return run
