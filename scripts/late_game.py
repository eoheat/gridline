"""Late-game diagnostics behind docs/LATE_GAME.md: where gridline trails Kalshi in the
final minutes of 2025, and what the gap is made of.

Every number the note quotes, apart from the benchmark's own (log loss, total minutes),
is computed here and saved to data/derived/late_game/late_game_<season>.json; the charts
go to docs/img/late_*.svg. Needs:
- the replay's price log (`gridline replay run --season 2025`) and the consolidated
  Kalshi tables (`gridline kalshi consolidate --season 2025`);
- the drive tables (`gridline drives build`) and the Phase 2 models trained through 2023
  and 2024 (`gridline sim train --last 2023`, `--last 2024`);
- the Phase 2 gate's prices for 2024 and 2025 (`gridline sim gate --season 2024 --every
  3`, `gridline sim gate --season 2025`) and the Phase 1 held-out predictions (`gridline
  wp cv`).

    uv run --group docs python scripts/late_game.py

Conventions (the same as the benchmark's, eval/benchmark.py):
- A minute's state is the model's last event known by then: "the last five minutes"
  means that event's clock is at most 5:00 in the 4th quarter. A game is close when
  that event's score margin is at most 8.
- Brier differences are gridline minus Kalshi, gridline's corrected for Monte Carlo
  noise; positive means Kalshi was more accurate. Intervals are 90%, resampling games.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from gridline import config
from gridline.eval import benchmark, bootstrap, market

SEASON = 2025
RUN = "replay_2025"
ENGINE_SEASON = 2024  # the replay's models were trained through 2024
LATE = 300.0  # the last five minutes of regulation
CLOSE = 8  # a one-score game
DELTAS = (0, 20)
WINDOWS = (("15:00 to 10:00", 600, 900), ("10:00 to 5:00", 300, 600), ("5:00 to 2:00", 120, 300),
           ("2:00 to 0:00", -1, 120))  # 4th-quarter clock of the model's last event: (lo, hi]
FG_RANGE = 35  # yardline_100 at most this: a field goal of about 52 yards or less
OUT_DIR = config.DERIVED_DIR / "late_game"


def out_path(season: int = SEASON) -> Path:
    return OUT_DIR / f"late_game_{season}.json"


def interval(values: np.ndarray, games: np.ndarray, weights: np.ndarray | None = None,
             n_boot: int = bootstrap.N_BOOT) -> dict:
    """Mean of values (weighted) with a 90% interval from resampling games."""
    w0 = np.ones(len(values)) if weights is None else weights
    draws = [np.average(values, weights=w * w0) for w in bootstrap.game_weights(games, n_boot)]
    return {"mean": float(np.average(values, weights=w0)), "p05": float(np.quantile(draws, 0.05)),
            "p95": float(np.quantile(draws, 0.95)), "rows": int(len(values)),
            "games": int(len(np.unique(games)))}


# ------------------------------------------------------------------------ the gap


def grid(prices: pl.DataFrame, minutes: pl.DataFrame, delta: float) -> pl.DataFrame:
    """The benchmark's winner grid with the state of the model's last event: its score
    margin (home), what comes next (phase), and how long ago the model's price was set."""
    g = benchmark.winner_grid(prices, minutes, delta)
    st = prices.select("q", "phase", "timing", "posteam_is_home", "down", "ydstogo", "yardline_100",
                       margin=pl.col("home_score") - pl.col("away_score"))
    return (g.join(st, on="q", how="left")
            .with_columns(close=pl.col("margin").abs() <= CLOSE,
                          age=(pl.col("t") - pl.col("t_known")).dt.total_microseconds() / 1e6)
            .sort("game_id", "t"))


def late(g: pl.DataFrame) -> pl.DataFrame:
    return g.filter((pl.col("qtr") == 4) & (pl.col("game_seconds_remaining") <= LATE))


def score(g: pl.DataFrame) -> dict:
    """Brier of gridline (corrected) and Kalshi on the rows of g, and their difference."""
    if g.height == 0:
        return {"rows": 0}
    r = benchmark.paired(g["model"].to_numpy(), g["market"].to_numpy(), g["y"].to_numpy(),
                         g["game_id"].to_numpy(), g["var_mc"].to_numpy())
    keys = ("rows", "games", "brier_model_corrected", "brier_market", "brier_corrected_diff",
            "brier_corrected_diff_p05", "brier_corrected_diff_p95")
    return {k: r[k] for k in keys}


def by_window(g: pl.DataFrame) -> dict:
    q4 = g.filter(pl.col("qtr") == 4)
    out = {}
    for name, lo, hi in WINDOWS:
        w = q4.filter((pl.col("game_seconds_remaining") > lo) & (pl.col("game_seconds_remaining") <= hi))
        out[name] = {"all": score(w), "close": score(w.filter(pl.col("close"))),
                     "not_close": score(w.filter(~pl.col("close")))}
    lt = late(g)
    out["last 5 minutes"] = {"all": score(lt), "close": score(lt.filter(pl.col("close"))),
                             "not_close": score(lt.filter(~pl.col("close")))}
    return out


def excess(g: pl.DataFrame, model: str = "model", var: str = "var_mc") -> pl.Expr:
    """Per-row Brier of gridline (corrected) minus Kalshi's."""
    return (pl.col(model) - pl.col("y")) ** 2 - pl.col(var) - (pl.col("market") - pl.col("y")) ** 2


def per_game(g: pl.DataFrame) -> pl.DataFrame:
    """Each game's summed excess Brier over its late minutes."""
    return (late(g).with_columns(d=excess(g))
            .group_by("game_id").agg(excess=pl.col("d").sum(), minutes=pl.len(), y=pl.col("y").first())
            .sort("excess", "game_id", descending=[True, False]))


def concentration(g: pl.DataFrame, top: int = 10) -> dict:
    pg = per_game(g)
    total = float(pg["excess"].sum())
    return {"games": pg.height, "excess_total": total, "positive_games": int((pg["excess"] > 0).sum()),
            "top_excess": float(pg["excess"].head(top).sum()), "top_n": top,
            "top_share": float(pg["excess"].head(top).sum() / total) if total else float("nan"),
            "top": pg.head(top).to_dicts()}


# ------------------------------------------------------------------ what the gap is made of


def staleness(g: pl.DataFrame) -> dict:
    """Late minutes by how old the model's price was (seconds since it became known)."""
    lt = late(g)
    out = {}
    for name, lo, hi in (("up to 15 s", -1, 15), ("15 to 30 s", 15, 30), ("30 to 60 s", 30, 60),
                         ("over 60 s", 60, 1e9)):
        part = lt.filter((pl.col("age") > lo) & (pl.col("age") <= hi))
        out[name] = score(part) | {"share": part.height / lt.height}
    return out


def waiting_for(g: pl.DataFrame, prices: pl.DataFrame) -> dict:
    """Late minutes by what gridline was missing when the minute closed, judged from its
    last known event and the next event in the price log:

      play under way       its last event is a snap and that play had not ended yet
      held by the rules    the play had ended, but the replay's timing rules had not yet
                           made its result public (a penalty, a review, a missing end
                           timestamp, the clock), or an earlier result was still held
      feed delay           the result, or the next snap, was public but gridline had not
                           heard yet (only with a delay)
      between plays        gridline had the last play's result, and the next snap had
                           not come yet
      kickoff, try         what comes next is a kickoff, or a try after a touchdown, and
                           its result was not public yet
    """
    p = prices.sort("game_id", "order").with_columns(
        public=pl.col("t").cum_max().over("game_id"))  # the replay's public time, before any delay
    nxt = p.select("q", nxt_kind=pl.col("kind").shift(-1).over("game_id"),
                   nxt_end=pl.col("t_play_end").shift(-1).over("game_id"),
                   nxt_public=pl.col("public").shift(-1).over("game_id"))
    lt = late(g).join(nxt, on="q", how="left")
    t = pl.col("t")
    lt = lt.with_columns(waiting=(
        pl.when(pl.col("event_kind") == "snap").then(
            pl.when(pl.col("nxt_end").is_null() | (t < pl.col("nxt_end"))).then(pl.lit("play under way"))
            .when(t < pl.col("nxt_public")).then(pl.lit("held by the rules"))
            .otherwise(pl.lit("feed delay")))
        .when(pl.col("nxt_public").is_not_null() & (t >= pl.col("nxt_public"))).then(pl.lit("feed delay"))
        .when(pl.col("phase") == "snap").then(pl.lit("between plays"))
        .when(pl.col("phase") == "kickoff").then(pl.lit("kickoff"))
        .when(pl.col("phase") == "pat").then(pl.lit("try"))
        .otherwise(pl.lit("other"))))
    out = {}
    for (w,), part in lt.group_by("waiting"):
        out[w] = score(part) | {"share": part.height / lt.height,
                                "excess_sum": float(part.select(excess(part)).to_series().sum())}
    return out


def oracle_clock(prices: pl.DataFrame) -> pl.DataFrame:
    """The price log as if the clock (and timeouts) of the next snap were known at the
    whistle: a play event followed by a snap event takes the snap event's price."""
    p = prices.sort("game_id", "order")
    nxt = pl.col("kind").shift(-1).over("game_id")
    swap = (pl.col("kind") == "play") & (pl.col("phase") == "snap") & (nxt == "snap")
    return p.with_columns(**{c: pl.when(swap).then(pl.col(c).shift(-1).over("game_id")).otherwise(pl.col(c))
                             for c in ("p_home", "p_tie", "n_sims")})


def fast_timing(prices: pl.DataFrame) -> pl.DataFrame:
    """The price log as if every play result were public at the play's end timestamp
    (a missing one at the snap plus the median length): no waiting for penalties,
    reviews or the clock. An upper bound: it knows reviewed rulings early."""
    return prices.with_columns(t=pl.when(pl.col("kind").is_in(["play", "final"])).then(pl.col("t_play_end"))
                               .otherwise(pl.col("t")))


def counterfactual(base_late: pl.DataFrame, prices_cf: pl.DataFrame, delta: float) -> dict:
    """The late gap with the model's prices taken from another price log, on the same
    minutes and against the same market, and its change from the base (a paired
    difference; the interval resamples games)."""
    rows = base_late.select("game_id", "t", "market", "y", "model", "var_mc")
    cf = benchmark.model_asof(prices_cf, rows.select("game_id", "t"), delta)
    cf = cf.with_columns(var_cf=pl.Series(benchmark.mc_var(cf["p_home"].to_numpy(), cf["p_tie"].to_numpy(),
                                                           cf["n_sims"].to_numpy().astype(float))))
    j = rows.join(cf.select("game_id", "t", model_cf="p_home", var_cf="var_cf"), on=["game_id", "t"])
    assert j.height == rows.height
    d0 = j.select(excess(j)).to_series().to_numpy()
    d1 = j.select(excess(j, "model_cf", "var_cf")).to_series().to_numpy()
    games = j["game_id"].to_numpy()
    return {"gap": interval(d1, games), "change": interval(d1 - d0, games),
            "moved": float(np.mean(np.abs(j["model_cf"].to_numpy() - j["model"].to_numpy()) > 1e-9))}


def clock_between_plays(prices: pl.DataFrame) -> dict:
    """4th-quarter play events followed by a snap event: the clock that ran between the
    whistle and the snap (which the play event does not know yet), and how far the
    price moved when the snap made it known."""
    p = prices.sort("game_id", "order").with_columns(
        nxt_kind=pl.col("kind").shift(-1).over("game_id"), nxt_p=pl.col("p_home").shift(-1).over("game_id"),
        nxt_gsr=pl.col("game_seconds_remaining").shift(-1).over("game_id"),
        nxt_t=pl.col("t").shift(-1).over("game_id"))
    pairs = p.filter((pl.col("kind") == "play") & (pl.col("phase") == "snap") & (pl.col("nxt_kind") == "snap")
                     & (pl.col("qtr") == 4)).with_columns(
        runoff=pl.col("game_seconds_remaining") - pl.col("nxt_gsr"),
        jump=pl.col("nxt_p") - pl.col("p_home"),
        margin=pl.col("home_score") - pl.col("away_score"))
    pairs = pairs.with_columns(to_leader=pl.col("jump") * pl.col("margin").sign())
    out = {"pairs_q4": pairs.height, "runoff_median_q4": float(pairs["runoff"].median())}
    for name, f in (("last 2 minutes, close", (pl.col("game_seconds_remaining") <= 120) & (pl.col("margin").abs() <= CLOSE)),
                    ("5:00 to 2:00, close", pl.col("game_seconds_remaining").is_between(120, LATE, closed="right")
                     & (pl.col("margin").abs() <= CLOSE)),
                    ("last 5 minutes", pl.col("game_seconds_remaining") <= LATE)):
        s = pairs.filter(f)
        lead = s.filter(pl.col("margin") != 0)
        out[name] = {"pairs": s.height, "runoff_median": float(s["runoff"].median()),
                     "runoff_mean": float(s["runoff"].mean()),
                     "abs_jump_mean": float(s["jump"].abs().mean()),
                     "abs_jump_p90": float(s["jump"].abs().quantile(0.9)),
                     "to_leader_mean": float(lead["to_leader"].mean()),
                     "to_leader": interval(lead["to_leader"].to_numpy(), lead["game_id"].to_numpy())}
    return out


# --------------------------------------------------------------------- snap moments


def snap_moments(prices: pl.DataFrame, trades: pl.DataFrame, sp: pl.DataFrame) -> pl.DataFrame:
    """Every 4th-quarter snap event (both sides know the exact state: the play has not
    started), with gridline's price and Kalshi's last trade at least a second before the
    snap, and how the game ended."""
    s = (prices.filter((pl.col("kind") == "snap") & (pl.col("qtr") == 4))
         .join(sp.select("game_id", "y"), on="game_id")
         .sort("game_id", "t"))
    by = {gid: (g["ts"].dt.epoch("us").to_numpy(), g["price"].to_numpy())
          for (gid,), g in trades.filter(pl.col("game_id").is_in(s["game_id"].unique().implode())).group_by("game_id")}
    mkt = np.full(s.height, np.nan)
    age = np.full(s.height, np.nan)
    t = s["t"].dt.epoch("us").to_numpy()
    for i, gid in enumerate(s["game_id"].to_list()):
        if gid not in by:
            continue
        ts, px = by[gid]
        j = np.searchsorted(ts, t[i] - 1_000_000, side="right") - 1
        if j >= 0:
            mkt[i], age[i] = px[j], (t[i] - ts[j]) / 1e6
    s = s.with_columns(market=pl.Series(mkt), trade_age=pl.Series(age),
                       model=pl.col("p_home"), margin=pl.col("home_score") - pl.col("away_score"))
    s = s.filter(pl.col("market").is_not_null() & pl.col("market").is_not_nan())
    return s.with_columns(var_mc=pl.Series(benchmark.mc_var(s["model"].to_numpy(), s["p_tie"].to_numpy(),
                                                           s["n_sims"].to_numpy().astype(float))),
                          close=pl.col("margin").abs() <= CLOSE)


def snap_accuracy(s: pl.DataFrame) -> dict:
    """The benchmark's comparison at the snaps: both prices scored against the outcome,
    in the same clock windows as the minute grid."""
    gsr = pl.col("game_seconds_remaining")
    out = {}
    windows = [(name, (gsr > lo) & (gsr <= hi)) for name, lo, hi in WINDOWS]
    for name, f in (*windows, ("last 5 minutes", gsr <= LATE),
                    ("last 5 minutes, close", (gsr <= LATE) & pl.col("close")),
                    ("last 2 minutes, close", (gsr <= 120) & pl.col("close"))):
        part = s.filter(f)
        out[name] = score(part) | {"trade_age_median": float(part["trade_age"].median())}
    return out


# ---------------------------------------------------------------- against the outcomes


def leader_view(s: pl.DataFrame) -> pl.DataFrame:
    """Snap moments with a leader, from the leader's side: its lead, and the win prices
    of gridline and Kalshi and whether it won (a tie counts half)."""
    home = pl.col("margin") > 0
    return (s.filter(pl.col("margin") != 0)
            .with_columns(lead=pl.col("margin").abs(),
                          won=pl.when(home).then(pl.col("y")).otherwise(1 - pl.col("y")),
                          p_model=pl.when(home).then(pl.col("model")).otherwise(1 - pl.col("model")),
                          p_market=pl.when(home).then(pl.col("market")).otherwise(1 - pl.col("market"))))


def offense_view(s: pl.DataFrame) -> pl.DataFrame:
    """Snap moments from the side of the team with the ball."""
    home = pl.col("posteam_is_home")
    return (s.filter(pl.col("posteam_is_home").is_not_null())
            .with_columns(off_margin=pl.when(home).then(pl.col("margin")).otherwise(-pl.col("margin")),
                          won=pl.when(home).then(pl.col("y")).otherwise(1 - pl.col("y")),
                          p_model=pl.when(home).then(pl.col("model")).otherwise(1 - pl.col("model")),
                          p_market=pl.when(home).then(pl.col("market")).otherwise(1 - pl.col("market"))))


def against_outcome(v: pl.DataFrame) -> dict:
    """How often the side won, against gridline's and Kalshi's average price for it (each
    minus the actual rate, with intervals), and the paired Brier difference. Snaps are
    weighted equally; `per_game` repeats the comparison with every game weighted equally
    (a game with many snaps in the situation counts once)."""
    if v.height == 0:
        return {"rows": 0}
    games = v["game_id"].to_numpy()
    won, pm, pk = v["won"].to_numpy(), v["p_model"].to_numpy(), v["p_market"].to_numpy()
    n = v.group_by("game_id").len()
    w = 1.0 / v.select("game_id").join(n, on="game_id", how="left")["len"].to_numpy()
    return {"rows": v.height, "games": int(len(np.unique(games))), "won": float(won.mean()),
            "model": float(pm.mean()), "market": float(pk.mean()),
            "model_minus_won": interval(pm - won, games), "market_minus_won": interval(pk - won, games),
            "brier_diff": interval((pm - won) ** 2 - v["var_mc"].to_numpy() - (pk - won) ** 2, games),
            "per_game": {"won": float(np.average(won, weights=w)), "model": float(np.average(pm, weights=w)),
                         "market": float(np.average(pk, weights=w)),
                         "model_minus_won": interval(pm - won, games, w),
                         "market_minus_won": interval(pk - won, games, w)}}


def lead_calibration(s: pl.DataFrame) -> dict:
    """Snap moments of the 4th quarter by situation: the leader's lead, or the score and
    field position of the team with the ball."""
    gsr = pl.col("game_seconds_remaining")
    lv, ov = leader_view(s), offense_view(s)
    out = {}
    for name, f in (("last 5 minutes, lead 1-3", (gsr <= LATE) & pl.col("lead").is_between(1, 3)),
                    ("last 5 minutes, lead 4-8", (gsr <= LATE) & pl.col("lead").is_between(4, 8)),
                    ("last 5 minutes, lead 9-16", (gsr <= LATE) & pl.col("lead").is_between(9, 16)),
                    ("15:00 to 5:00, lead 1-3", (gsr > LATE) & pl.col("lead").is_between(1, 3)),
                    ("15:00 to 5:00, lead 4-8", (gsr > LATE) & pl.col("lead").is_between(4, 8))):
        out[name] = against_outcome(lv.filter(f))
    m, yl = pl.col("off_margin"), pl.col("yardline_100")
    for name, f in (("tied or down 1-3", m.is_between(-3, 0)),
                    ("tied or down 1-2, in field-goal range", m.is_between(-2, 0) & (yl <= FG_RANGE)),
                    ("tied or down 1-2, outside field-goal range", m.is_between(-2, 0) & (yl > FG_RANGE)),
                    ("down 3", m == -3),
                    ("down 4-8", m.is_between(-8, -4)),
                    ("ahead 1-8", m.is_between(1, 8))):
        out[f"last 2 minutes, offense {name}"] = against_outcome(ov.filter((gsr <= 120) & f))
        out[f"5:00 to 2:00, offense {name}"] = against_outcome(
            ov.filter(gsr.is_between(120, LATE, closed="right") & f))
    return out


def surprise_test(g: pl.DataFrame, prices: pl.DataFrame) -> dict:
    """Does how a game has gone against the kickoff expectation (the score margin now
    minus the pregame mean margin times the share of the game played) predict what
    gridline's late price misses? gridline keeps its kickoff view of the two teams; if
    that hurts, the team that has outplayed it should beat gridline's price (a positive
    slope). The same for Kalshi, which can revise its view."""
    pre = prices.filter(pl.col("kind") == "pregame").select("game_id", pre_margin="margin_mean")
    lt = late(g).join(pre, on="game_id").with_columns(
        surprise=(pl.col("margin") - pl.col("pre_margin") * (3600 - pl.col("game_seconds_remaining")) / 3600) / 7)
    games = lt["game_id"].to_numpy()
    x = lt["surprise"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    out = {"rows": lt.height, "surprise_sd_touchdowns": float(np.std(x))}
    for who in ("model", "market"):
        r = lt["y"].to_numpy() - lt[who].to_numpy()
        beta = np.linalg.lstsq(X, r, rcond=None)[0]
        draws = []
        for w in bootstrap.game_weights(games, 500):
            sw = np.sqrt(w)
            draws.append(np.linalg.lstsq(X * sw[:, None], r * sw, rcond=None)[0][1])
        out[who] = {"slope_per_touchdown": float(beta[1]), "p05": float(np.quantile(draws, 0.05)),
                    "p95": float(np.quantile(draws, 0.95))}
    return out


# ------------------------------------------------------------------ the drive model late


OUTCOMES = {"touchdown": ("td0", "td1", "td2"), "field goal made": ("fg",), "field goal missed": ("fg_miss",),
            "punt": ("punt",), "turnover": ("turnover", "opp_td"), "turnover on downs": ("downs",),
            "end of half": ("end_half",), "safety": ("safety",)}


def drive_predictions(prices: pl.DataFrame | None, season: int = SEASON,
                      engine_season: int = ENGINE_SEASON) -> pl.DataFrame:
    """Every clean regulation snap of the season with the drive model's probabilities of
    how the drive ends: the model trained through engine_season, fed each game's knobs
    from the replay's price log as the replay did (or, with prices None, the sportsbook
    line, as the Phase 2 gate did)."""
    from gridline.inventory import load_events
    from gridline.models import drive_model as dm
    from gridline.models import drives
    from gridline.models.simulator import Engine

    snaps = dm.training_snaps(drives.load_snaps([season]))
    if prices is not None:
        knobs = prices.filter(pl.col("kind") == "pregame").select("game_id", "knob_spread", "knob_total")
        snaps = (snaps.join(knobs, on="game_id")
                 .with_columns(spread_off=pl.when(pl.col("is_home")).then(pl.col("knob_spread"))
                               .otherwise(-pl.col("knob_spread")), pregame_total=pl.col("knob_total")))
    snaps = snaps.join(load_events(season).select("game_id", "seq", "play_type", "desc"), on=["game_id", "seq"],
                       how="left")
    step = Engine.load(engine_season).step
    x = dm.frame_features(snaps, step.season_cap)
    probs = step.probs(x)
    rp = np.zeros((probs.shape[0], len(drives.RESULTS)))
    np.add.at(rp.T, dm.CLASS_RESULT, probs.T)
    cols = {}
    for name, rs in OUTCOMES.items():
        idx = [drives.RESULTS.index(r) for r in rs]
        cols[f"p_{name}"] = rp[:, idx].sum(1)
        cols[f"y_{name}"] = snaps["result"].is_in(list(rs)).to_numpy().astype(float)
    # the drive's clock: the model's mean and where the actual fell (probability
    # integral transform, mid-point where the half's end puts a point mass)
    clock = np.clip(snaps["half_seconds_remaining"].to_numpy().astype(float), 0, None)
    actual = np.minimum(snaps["seconds"].to_numpy().astype(float), clock)
    pb = np.zeros((probs.shape[0], dm.N_BINS))
    timed = dm.CLASS_BIN >= 0
    np.add.at(pb.T, dm.CLASS_BIN[timed], probs[:, timed].T)
    grid = np.linspace(0, 1, dm.N_QUANTILES)
    below, mean = np.zeros(len(clock)), probs[:, dm.END_HALF] * clock
    for b in range(dm.N_BINS):
        q = step.quantiles[b]
        below += pb[:, b] * np.interp(np.minimum(actual, clock), q, grid, left=0.0, right=1.0)
        mids = (q[:-1] + q[1:]) / 2
        mean += pb[:, b] * np.minimum(mids[None, :], clock[:, None]).mean(1)
    at_end = actual >= clock - 1e-6
    pit = np.where(at_end, below + 0.5 * (1 - below), below)
    return snaps.with_columns(**{k: pl.Series(v) for k, v in cols.items()},
                              pit=pl.Series(pit), clock_mean=pl.Series(mean), clock_actual=pl.Series(actual))


def drive_calibration(d: pl.DataFrame) -> dict:
    """Predicted against actual drive outcomes (and clock), by stage and situation."""
    gsr, diff = pl.col("game_seconds_remaining"), pl.col("score_diff")
    late_ = (pl.col("qtr") == 4) & (gsr <= LATE)
    groups = {
        "1st to 3rd quarter": pl.col("qtr") <= 3,
        "4th quarter to 5:00": (pl.col("qtr") == 4) & (gsr > LATE),
        "last 5 minutes": late_,
        "last 5 minutes, close": late_ & (diff.abs() <= CLOSE),
        "last 2 minutes, close": late_ & (gsr <= 120) & (diff.abs() <= CLOSE),
        "last 5 minutes, offense down 1-8": late_ & diff.is_between(-8, -1),
        "last 5 minutes, offense tied": late_ & (diff == 0),
        "last 5 minutes, offense up 1-8": late_ & diff.is_between(1, 8),
        "last 2 minutes of the 1st half": (pl.col("qtr") == 2) & (pl.col("half_seconds_remaining") <= 120),
    }
    out = {}
    for name, f in groups.items():
        part = d.filter(f)
        games = part["game_id"].to_numpy()
        res = {"snaps": part.height, "games": int(len(np.unique(games))), "drives": part.select(
            pl.struct("game_id", "fixed_drive").n_unique()).item()}
        for o in OUTCOMES:
            p, y = part[f"p_{o}"].to_numpy(), part[f"y_{o}"].to_numpy()
            res[o] = {"predicted": float(p.mean()), "actual": float(y.mean()),
                      "actual_minus_predicted": interval(y - p, games)}
        res["clock_pit_mean"] = float(part["pit"].mean())
        res["clock_mean_predicted"] = float(part["clock_mean"].mean())
        res["clock_mean_actual"] = float(part["clock_actual"].mean())
        out[name] = res
    return out


def field_goal_calibration(d: pl.DataFrame) -> dict:
    """At the snaps of field-goal attempts: the drive model's make share (made over made
    plus missed) against how often the kicks went in, by distance."""
    fg = d.filter((pl.col("play_type") == "field_goal") & pl.col("result").is_in(["fg", "fg_miss"])).with_columns(
        share=pl.col("p_field goal made") / (pl.col("p_field goal made") + pl.col("p_field goal missed")),
        made=(pl.col("result") == "fg").cast(pl.Float64), distance=pl.col("yardline_100") + 17)
    late_close = (pl.col("qtr") == 4) & (pl.col("game_seconds_remaining") <= LATE) & (pl.col("score_diff").abs() <= CLOSE)
    out = {}
    for name, f in (("all", pl.lit(True)), ("under 40 yards", pl.col("distance") < 40),
                    ("40-49 yards", pl.col("distance").is_between(40, 49)), ("50+ yards", pl.col("distance") >= 50),
                    ("last 5 minutes, close", late_close)):
        part = fg.filter(f)
        games = part["game_id"].to_numpy()
        out[name] = {"kicks": part.height, "made": float(part["made"].mean()), "model": float(part["share"].mean()),
                     "made_minus_model": interval(part["made"].to_numpy() - part["share"].to_numpy(), games)}
    return out


# ------------------------------------------------------------------ special situations


def kneel_then_kick(s: pl.DataFrame, events: pl.DataFrame, d: pl.DataFrame) -> list[dict]:
    """4th-quarter drives in which a team that was tied or trailing by 1-2, already in
    field-goal range, knelt to run the clock down before kicking: at its first kneel,
    gridline's and Kalshi's price for it, what the drive model expected of the drive from
    there (d: drive_predictions), and how the kick and the game went."""
    ev = events.sort("game_id", "seq").with_columns(
        off_margin=pl.when(pl.col("posteam_is_home")).then(pl.col("home_score_pre") - pl.col("away_score_pre"))
        .otherwise(pl.col("away_score_pre") - pl.col("home_score_pre")))
    kneels = ev.filter((pl.col("play_type") == "qb_kneel") & (pl.col("qtr") == 4) & pl.col("off_margin").is_between(-2, 0)
                       & (pl.col("yardline_100") <= FG_RANGE))
    out = []
    for (gid, team), k in kneels.group_by("game_id", "posteam", maintain_order=True):
        first = k.sort("seq").row(0, named=True)
        after = ev.filter((pl.col("game_id") == gid) & (pl.col("seq") > first["seq"]) & (pl.col("posteam") == team)
                          & pl.col("play_type").is_in(["field_goal", "run", "pass"]))
        kick = after.filter(pl.col("play_type") == "field_goal").head(1)
        if kick.height == 0:
            continue
        snap = s.filter((pl.col("game_id") == gid) & (pl.col("seq") == first["seq"]))
        if snap.height == 0:
            continue
        r = snap.row(0, named=True)
        home = bool(first["posteam_is_home"])
        side = (lambda p: p) if home else (lambda p: 1 - p)
        row = {"game_id": gid, "team": team, "clock": first["game_seconds_remaining"], "qtr": first["qtr"],
               "margin": first["off_margin"], "yardline_100": first["yardline_100"],
               "timeouts_def": first["away_timeouts"] if home else first["home_timeouts"],
               "model": side(r["model"]), "market": side(r["market"]), "won": side(r["y"]),
               "kick": kick["desc"][0][:60]}
        dm_row = d.filter((pl.col("game_id") == gid) & (pl.col("seq") == first["seq"]))
        if dm_row.height:
            x = dm_row.row(0, named=True)
            row |= {"drive_field_goal_made": x["p_field goal made"], "drive_touchdown": x["p_touchdown"],
                    "drive_field_goal_missed": x["p_field goal missed"], "drive_end_of_half": x["p_end of half"],
                    "drive_turnover": x["p_turnover"], "drive_clock_mean": x["clock_mean"],
                    "half_clock": x["half_seconds_remaining"]}
        out.append(row)
    return out


def at_least(p: np.ndarray, k: int) -> float:
    """P(at least k successes) from independent chances p (Poisson binomial)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for x in p:
        dist[1:] = dist[1:] * (1 - x) + dist[:-1] * x
        dist[0] *= 1 - x
    return float(dist[k:].sum())


def kneel_summary(kk: list[dict]) -> dict:
    """The kneel-then-kick drives together: average prices, how many won, both prices'
    Brier scores at the first kneel, and how likely that many wins was under each."""
    won = np.array([r["won"] for r in kk])
    pm, pk = np.array([r["model"] for r in kk]), np.array([r["market"] for r in kk])
    k = int(round(won.sum()))
    return {"n": len(kk), "won": float(won.mean()), "model": float(pm.mean()), "market": float(pk.mean()),
            "gap_min": float((pk - pm).min()), "gap_max": float((pk - pm).max()),
            "brier_model": float(((pm - won) ** 2).mean()), "brier_market": float(((pk - won) ** 2).mean()),
            "p_at_least_won_model": at_least(pm, k), "p_at_least_won_market": at_least(pk, k),
            # the drive model's price if the opponent never got the ball back: the drive
            # scores and wins, or fails (a tied team then gets overtime, taken as 50/50)
            "outcome_only": float(np.mean([r["drive_touchdown"] + r["drive_field_goal_made"]
                                           + (0.5 if r["margin"] == 0 else 0.0)
                                           * (1 - r["drive_touchdown"] - r["drive_field_goal_made"]) for r in kk])),
            "drive_no_score": float(np.mean([1 - r["drive_touchdown"] - r["drive_field_goal_made"] for r in kk])),
            "drive_field_goal_missed": float(np.mean([r["drive_field_goal_missed"] for r in kk])),
            "time_left_expected_mean": float(np.mean([r["half_clock"] - r["drive_clock_mean"] for r in kk])),
            "time_left_expected_median": float(np.median([r["half_clock"] - r["drive_clock_mean"] for r in kk])),
            "drive_field_goal_made": float(np.mean([r["drive_field_goal_made"] for r in kk])),
            "drive_touchdown": float(np.mean([r["drive_touchdown"] for r in kk])),
            "drive_clock_used": float(sum(r["drive_clock_mean"] for r in kk)),
            "clock_left": float(sum(r["half_clock"] for r in kk))}


def clock_kill_history(first: int = 2015, last: int = SEASON) -> list[dict]:
    """How often a team in field-goal range late in a close game knelt to run the clock
    down, season by season: drives with a 4th-quarter snap with 1:40 or less left, the
    offense tied or down 1-2 and at the opponent's 35 or closer; knelt means a kneel-down
    from that snap on."""
    from gridline.inventory import load_events
    from gridline.models import drives

    rows = []
    for season in range(first, last + 1):
        sn = drives.load_snaps([season]).join(load_events(season).select("game_id", "seq", "play_type"),
                                              on=["game_id", "seq"], how="left")
        s = sn.filter((pl.col("qtr") == 4) & (pl.col("game_seconds_remaining") <= 100)
                      & pl.col("score_diff").is_between(-2, 0) & (pl.col("yardline_100") <= FG_RANGE))
        first_seq = s.group_by("game_id", "fixed_drive").agg(seq0=pl.col("seq").min())
        d = sn.join(first_seq, on=["game_id", "fixed_drive"]).filter(pl.col("seq") >= pl.col("seq0"))
        per = d.group_by("game_id", "fixed_drive").agg(knelt=(pl.col("play_type") == "qb_kneel").any())
        rows.append({"season": season, "drives": per.height, "knelt": int(per["knelt"].sum())})
    return rows


def sim_eval_calibration(season: int) -> dict:
    """The Phase 2 gate's prices (every snap of the season, from the models trained
    through the season before, anchored to the sportsbook line) in the late situations
    of lead_calibration, for the simulator and the Phase 1 model: the same models a
    season earlier are the check on whether a 2025 miss is new."""
    from gridline.inventory import load_events

    path = config.DERIVED_DIR / "sim_eval" / f"sim_eval_{season}.parquet"
    se = pl.read_parquet(path).filter(pl.col("kind") != "pregame")
    ev = load_events(season).select("game_id", "seq", "home_score_pre", "away_score_pre", "yardline_100",
                                    pos_home="posteam_is_home")
    base = se.join(ev, on=["game_id", "seq"]).filter(pl.col("qtr") == 4).with_columns(
        margin=pl.col("home_score_pre") - pl.col("away_score_pre"),
        y=pl.when(pl.col("final_margin") > 0).then(1.0).when(pl.col("final_margin") < 0).then(0.0).otherwise(0.5))
    gsr = pl.col("game_seconds_remaining")
    out = {"states": base.height}
    for who in ("wp_sim_home", "wp_direct_home"):
        x = base.filter(pl.col(who).is_not_null())
        home = pl.col("margin") > 0
        lv = x.filter(pl.col("margin") != 0).with_columns(
            lead=pl.col("margin").abs(), won=pl.when(home).then(pl.col("y")).otherwise(1 - pl.col("y")),
            p=pl.when(home).then(pl.col(who)).otherwise(1 - pl.col(who)))
        oh = pl.col("pos_home")
        ov = x.filter(oh.is_not_null()).with_columns(
            om=pl.when(oh).then(pl.col("margin")).otherwise(-pl.col("margin")),
            won=pl.when(oh).then(pl.col("y")).otherwise(1 - pl.col("y")),
            p=pl.when(oh).then(pl.col(who)).otherwise(1 - pl.col(who)))
        res = {}
        for name, v in (("last 5 minutes, lead 1-3", lv.filter((gsr <= LATE) & pl.col("lead").is_between(1, 3))),
                        ("last 5 minutes, lead 4-8", lv.filter((gsr <= LATE) & pl.col("lead").is_between(4, 8))),
                        ("last 2 minutes, offense tied or down 1-3", ov.filter((gsr <= 120) & pl.col("om").is_between(-3, 0))),
                        ("last 2 minutes, offense down 4-8", ov.filter((gsr <= 120) & pl.col("om").is_between(-8, -4))),
                        ("last 2 minutes, offense ahead 1-8", ov.filter((gsr <= 120) & pl.col("om").is_between(1, 8)))):
            res[name] = {"won": float(v["won"].mean()), "price": float(v["p"].mean()),
                         "price_minus_won": interval(v["p"].to_numpy() - v["won"].to_numpy(), v["game_id"].to_numpy())}
        out[{"wp_sim_home": "simulator", "wp_direct_home": "phase1"}[who]] = res
    return out


def historical_calibration(first: int = 2000, last: int = 2019) -> dict:
    """How much the late calibration of a win-probability model moves from one season to
    the next by chance: the Phase 1 model's held-out predictions (leave one season out,
    with the spread) in two late situations, season by season."""
    from gridline.inventory import load_events

    cv = pl.read_parquet(config.DERIVED_DIR / "wp_cv" / "wp_spread.parquet")
    rows = []
    for season in range(first, last + 1):
        ev = load_events(season).select("game_id", "seq", "game_seconds_remaining", "posteam_is_home",
                                        "home_score_pre", "away_score_pre")
        d = cv.filter((pl.col("season") == season) & (pl.col("qtr") == 4)).join(ev, on=["game_id", "seq"]).with_columns(
            om=pl.when(pl.col("posteam_is_home")).then(pl.col("home_score_pre") - pl.col("away_score_pre"))
            .otherwise(pl.col("away_score_pre") - pl.col("home_score_pre")))
        gsr = pl.col("game_seconds_remaining")
        lead = d.filter((gsr <= LATE) & pl.col("om").abs().is_between(1, 3)).with_columns(
            p=pl.when(pl.col("om") > 0).then(pl.col("wp")).otherwise(1 - pl.col("wp")),
            won=pl.when(pl.col("om") > 0).then(pl.col("label")).otherwise(1 - pl.col("label")).cast(pl.Float64))
        off = d.filter((gsr <= 120) & pl.col("om").is_between(-3, 0)).with_columns(
            p=pl.col("wp"), won=pl.col("label").cast(pl.Float64))
        rows.append({"season": season, "lead_1_3_games": lead["game_id"].n_unique(),
                     "lead_1_3": float((lead["p"] - lead["won"]).mean()),
                     "offense_tied_or_down_1_3_games": off["game_id"].n_unique(),
                     "offense_tied_or_down_1_3": float((off["p"] - off["won"]).mean())})
    t = pl.DataFrame(rows)
    return {"by_season": rows,
            "lead_1_3": {"mean": float(t["lead_1_3"].mean()), "sd": float(t["lead_1_3"].std()),
                         "min": float(t["lead_1_3"].min()), "max": float(t["lead_1_3"].max())},
            "offense_tied_or_down_1_3": {"mean": float(t["offense_tied_or_down_1_3"].mean()),
                                         "sd": float(t["offense_tied_or_down_1_3"].std()),
                                         "min": float(t["offense_tied_or_down_1_3"].min()),
                                         "max": float(t["offense_tied_or_down_1_3"].max())}}


def late_drive_history(first: int = 2015, last: int = SEASON) -> list[dict]:
    """How drives in the last two minutes of close games ended, season by season (each
    drive once, from its first snap in that window), and how often an offense that was
    tied or down by 1-3 scored."""
    from gridline.models import drive_model as dm
    from gridline.models import drives

    sn = dm.training_snaps(drives.load_snaps(range(first, last + 1)))
    w = sn.filter((pl.col("qtr") == 4) & (pl.col("game_seconds_remaining") <= 120) & (pl.col("score_diff").abs() <= CLOSE))
    f = w.sort("game_id", "seq").group_by("game_id", "fixed_drive").agg(pl.all().first())
    res = pl.col("result")
    t = f.group_by("season").agg(
        drives=pl.len(), touchdown=res.is_in(["td0", "td1", "td2"]).mean(), field_goal_made=(res == "fg").mean(),
        field_goal_missed=(res == "fg_miss").mean(), punt=(res == "punt").mean(),
        turnover=res.is_in(["turnover", "opp_td"]).mean(), downs=(res == "downs").mean(),
        end_of_half=(res == "end_half").mean(),
        tied_or_down_1_3_drives=pl.col("score_diff").is_between(-3, 0).sum(),
        tied_or_down_1_3_scored=res.is_in(["td0", "td1", "td2", "fg"]).filter(pl.col("score_diff").is_between(-3, 0)).mean())
    return t.sort("season").to_dicts()


def onside_kicks(season: int = SEASON, engine_season: int = ENGINE_SEASON, first: int = 2019) -> dict:
    """Onside kicks: how often the kicking team recovered an attempt, by season, and the
    simulator's pool of kicks by a team trailing in the last five minutes (the context
    in which it draws onside attempts) against the same context in the season replayed."""
    from gridline.data import nflverse
    from gridline.models import drives
    from gridline.models import transitions as tr

    p = nflverse.load_pbp(list(range(first, season + 1)), columns=["season", "kickoff_attempt", "desc",
                                                                   "own_kickoff_recovery"])
    k = p.filter((pl.col("kickoff_attempt") == 1) & pl.col("desc").str.contains("(?i)onside"))
    by = k.group_by("season").agg(attempts=pl.len(), recovered=pl.col("own_kickoff_recovery").sum()).sort("season")
    pool = tr.Transitions.load(tr.transitions_path(engine_season)).kicks
    lo, n = pool.start[tr.KICK_ONSIDE], pool.count[tr.KICK_ONSIDE]
    d = drives.load_drives([season])
    now = tr._kick_frame(d.filter(pl.col("result").is_not_null())).filter(pl.col("bucket") == tr.KICK_ONSIDE)
    return {"by_season": by.to_dicts(), "pool_kicks": int(n), "pool_kicker_keeps": float(pool.same[lo:lo + n].mean()),
            "season_kicks": now.height, "season_kicker_keeps": float(now["same"].mean())}


def kickoff_start(season: int = SEASON, engine_season: int = ENGINE_SEASON) -> dict:
    """Where the receiving team's first snap was after an ordinary kickoff: the
    simulator's pool (one season, the one before the replay) against the season replayed."""
    from gridline.models import drives
    from gridline.models import transitions as tr

    pool = tr.Transitions.load(tr.transitions_path(engine_season)).kicks
    lo, n = pool.start[tr.KICK_NORMAL], pool.count[tr.KICK_NORMAL]
    rec = ~pool.same[lo:lo + n]
    d = drives.load_drives([season])
    now = tr._kick_frame(d.filter(pl.col("result").is_not_null())).filter(
        (pl.col("bucket") == tr.KICK_NORMAL) & ~pl.col("same") & pl.col("yardline").is_not_null())
    return {"pool_kicks": int(rec.sum()), "pool_start_yardline_100": float(np.nanmean(pool.yardline[lo:lo + n][rec])),
            "season_kicks": now.height, "season_start_yardline_100": float(now["yardline"].mean())}


def two_point(season: int = SEASON, engine_season: int = ENGINE_SEASON) -> dict:
    """Tries after a touchdown late in the game (the last 15 minutes): the simulator's
    table of 0, 1 and 2 extra points against the season replayed, by the scorer's lead
    after the six points."""
    from gridline.models import drives
    from gridline.models import transitions as tr

    table = tr.Transitions.load(tr.transitions_path(engine_season)).pat
    d = drives.load_drives([season]).filter(pl.col("result").is_in(["td0", "td1", "td2"])
                                            & pl.col("home_pre").is_not_null() & pl.col("away_pre").is_not_null())
    home_off = pl.col("offense") == pl.col("home_team")
    d = d.with_columns(lead6=pl.when(home_off).then(pl.col("home_pre") - pl.col("away_pre"))
                       .otherwise(pl.col("away_pre") - pl.col("home_pre")) + 6,
                       late=((pl.col("half") == "Half2") & (pl.col("end_gsr") <= 900)).fill_null(False),
                       k=pl.col("result").replace_strict({"td0": 0, "td1": 1, "td2": 2}, return_dtype=pl.Int64))
    d = d.filter(pl.col("late"))
    ctx = tr.pat_context(d["lead6"].to_numpy().astype(np.int64), np.ones(d.height, bool))
    pred = table[ctx]
    out = {}
    for name, lo, hi in (("all", -99, 99), ("trailing after the six", -99, -1), ("tied or leading by 1-2", 0, 2),
                         ("leading by 3+", 3, 99)):
        m = (d["lead6"].to_numpy() >= lo) & (d["lead6"].to_numpy() <= hi)
        out[name] = {"touchdowns": int(m.sum()),
                     "predicted": [float(x) for x in pred[m].mean(0)],
                     "actual": [float((d["k"].to_numpy()[m] == j).mean()) for j in range(3)]}
    return out


def leader_kneels(s: pl.DataFrame, events: pl.DataFrame) -> dict:
    """Kneel-downs by the leading team in the last two minutes (victory formation):
    gridline's and Kalshi's price for the kneeling team at the snap."""
    k = events.filter((pl.col("play_type") == "qb_kneel") & (pl.col("qtr") == 4)
                      & (pl.col("game_seconds_remaining") <= 120)).select("game_id", "seq", "posteam_is_home")
    j = s.join(k, on=["game_id", "seq"], suffix="_k").with_columns(
        lead=pl.when(pl.col("posteam_is_home_k")).then(pl.col("margin")).otherwise(-pl.col("margin")),
        p_model=pl.when(pl.col("posteam_is_home_k")).then(pl.col("model")).otherwise(1 - pl.col("model")),
        p_market=pl.when(pl.col("posteam_is_home_k")).then(pl.col("market")).otherwise(1 - pl.col("market"))
    ).filter(pl.col("lead") > 0)
    return {"snaps": j.height, "games": j["game_id"].n_unique(), "model_median": float(j["p_model"].median()),
            "model_min": float(j["p_model"].min()), "market_median": float(j["p_market"].median())}


def comebacks(g: pl.DataFrame) -> dict:
    """The late excess split by how the game went from its first late minute: the team
    then leading held on, or did not (lost or tied), or the game was tied."""
    lt = late(g).with_columns(d=excess(g))
    first = lt.sort("t").group_by("game_id").agg(m0=pl.col("margin").first(), y=pl.col("y").first(),
                                                 model0=pl.col("model").first(), market0=pl.col("market").first())
    held = ((pl.col("m0") > 0) & (pl.col("y") == 1)) | ((pl.col("m0") < 0) & (pl.col("y") == 0))
    first = first.with_columns(how=pl.when(pl.col("m0") == 0).then(pl.lit("tied"))
                               .when(held).then(pl.lit("leader held on")).otherwise(pl.lit("leader did not win")))
    per = lt.group_by("game_id").agg(excess=pl.col("d").sum(), minutes=pl.len()).join(first, on="game_id")
    out = {}
    for (how,), part in per.group_by("how"):
        out[how] = {"games": part.height, "minutes": int(part["minutes"].sum()), "excess": float(part["excess"].sum()),
                    "excess_per_minute": float(part["excess"].sum() / part["minutes"].sum()),
                    "positive_games": int((part["excess"] > 0).sum())}
    # one row per game: the leader's price at the first late minute, and whether it won
    led = first.filter(pl.col("m0") != 0).with_columns(
        won=pl.when(pl.col("m0") > 0).then(pl.col("y")).otherwise(1 - pl.col("y")),
        p_model=pl.when(pl.col("m0") > 0).then(pl.col("model0")).otherwise(1 - pl.col("model0")),
        p_market=pl.when(pl.col("m0") > 0).then(pl.col("market0")).otherwise(1 - pl.col("market0")))
    for name, f in (("leaders", pl.lit(True)), ("leaders by 1-8", pl.col("m0").abs() <= CLOSE)):
        part = led.filter(f)
        games = part["game_id"].to_numpy()
        out[f"first late minute, {name}"] = {
            "games": part.height, "won": float(part["won"].mean()), "model": float(part["p_model"].mean()),
            "market": float(part["p_market"].mean()),
            "model_minus_won": interval(part["p_model"].to_numpy() - part["won"].to_numpy(), games),
            "market_minus_won": interval(part["p_market"].to_numpy() - part["won"].to_numpy(), games)}
    return out


def league_trends(first: int = 2015, last: int = SEASON) -> list[dict]:
    """Kicking and fourth-down decisions by season (all games, from nflverse)."""
    from gridline.data import nflverse

    p = nflverse.load_pbp(list(range(first, last + 1)), columns=[
        "season", "game_id", "play_type", "down", "field_goal_attempt", "field_goal_result", "kick_distance", "qtr",
        "game_seconds_remaining", "score_differential"])
    fg = p.filter(pl.col("field_goal_attempt") == 1).with_columns(
        made=(pl.col("field_goal_result") == "made").cast(pl.Float64), dist=pl.col("kick_distance"))
    late_close = (pl.col("qtr") == 4) & (pl.col("game_seconds_remaining") <= LATE) & (pl.col("score_differential").abs() <= CLOSE)
    k = fg.group_by("season").agg(
        fg_attempts=pl.len(), fg_made=pl.col("made").mean(), fg_distance=pl.col("dist").mean(),
        fg_40_49=pl.col("made").filter(pl.col("dist").is_between(40, 49)).mean(),
        fg_50_plus=pl.col("made").filter(pl.col("dist") >= 50).mean(), fg_50_plus_attempts=(pl.col("dist") >= 50).sum(),
        fg_55_plus=pl.col("made").filter(pl.col("dist") >= 55).mean(), fg_55_plus_attempts=(pl.col("dist") >= 55).sum(),
        fg_late_close=pl.col("made").filter(late_close).mean(), fg_late_close_attempts=late_close.sum())
    fourth = p.filter((pl.col("down") == 4) & pl.col("play_type").is_in(["run", "pass", "punt", "field_goal"])).group_by(
        "season").agg(fourth_downs=pl.len(), go_rate=pl.col("play_type").is_in(["run", "pass"]).mean(),
                      punt_rate=(pl.col("play_type") == "punt").mean())
    games = p.group_by("season").agg(games=pl.col("game_id").n_unique())
    return k.join(fourth, on="season").join(games, on="season").sort("season").to_dicts()


CASES = ("2025_19_GB_CHI", "2025_02_NYG_DAL", "2025_12_NYG_DET", "2025_07_PIT_CIN")


def timelines(grids: dict, games=CASES) -> dict:
    """The late minutes of a few games, for the case studies: the market, and gridline
    at each delay, with the clock and score margin of gridline's last event. A minute
    that is late at one delay only still carries the market's price."""
    out = {}
    for gid in games:
        rows = None
        for d, g in grids.items():
            x = late(g).filter(pl.col("game_id") == gid).select(
                "t", pl.col("market").alias(f"market_{d}"), pl.col("y").alias(f"y_{d}"),
                pl.col("model").alias(f"model_{d}"), pl.col("game_seconds_remaining").alias(f"clock_{d}"),
                pl.col("margin").alias(f"margin_{d}"))
            rows = x if rows is None else rows.join(x, on="t", how="full", coalesce=True)
        rows = rows.with_columns(market=pl.coalesce([f"market_{d}" for d in grids]),
                                 y=pl.coalesce([f"y_{d}" for d in grids])).drop(
            [f"market_{d}" for d in grids] + [f"y_{d}" for d in grids])
        out[gid] = rows.sort("t").with_columns(pl.col("t").dt.strftime("%H:%M")).to_dicts()
    return out


# ----------------------------------------------------------------------------- run


def run(season: int = SEASON) -> dict:
    from gridline.inventory import load_events

    prices = benchmark.load_prices(RUN)
    sp = benchmark.spans(prices)
    minutes = market.winner_minutes(season)
    trades = market.winner_trades(season)
    events = load_events(season)
    res: dict = {"season": season, "run": RUN, "engine_season": ENGINE_SEASON, "deltas": list(DELTAS),
                 "late_seconds": LATE, "close_margin": CLOSE, "fg_range_yardline_100": FG_RANGE}
    grids = {d: grid(prices, minutes, d) for d in DELTAS}
    res["gap"] = {str(d): by_window(grids[d]) for d in DELTAS}
    res["mc_noise_late"] = float(late(grids[0])["var_mc"].mean())
    res["timelines"] = timelines(grids)
    res["concentration"] = {str(d): concentration(grids[d]) for d in DELTAS}
    res["comebacks"] = {str(d): comebacks(grids[d]) for d in DELTAS}
    res["staleness"] = {str(d): staleness(grids[d]) for d in DELTAS}
    res["waiting_for"] = {str(d): waiting_for(grids[d], prices) for d in DELTAS}
    cfs = {"oracle clock": oracle_clock(prices), "fast timing": fast_timing(prices)}
    cfs["both"] = fast_timing(cfs["oracle clock"])
    res["counterfactual"] = {str(d): {k: counterfactual(late(grids[d]), p, d) for k, p in cfs.items()} for d in DELTAS}
    res["clock_between_plays"] = clock_between_plays(prices)
    s = snap_moments(prices, trades, sp)
    res["snaps"] = snap_accuracy(s)
    res["lead_calibration"] = lead_calibration(s)
    res["surprise"] = surprise_test(grids[0], prices)
    d = drive_predictions(prices, season)
    res["drives"] = drive_calibration(d)
    res["field_goals"] = field_goal_calibration(d)
    res["trends"] = league_trends(last=season)
    res["late_drive_history"] = late_drive_history(last=season)
    # the same check a season earlier, with the models trained through the season before
    # it (fed the sportsbook line: there is no Kalshi replay of that season)
    res["drives_previous_season"] = drive_calibration(drive_predictions(None, season - 1, ENGINE_SEASON - 1))
    res["sim_eval"] = {str(y): sim_eval_calibration(y) for y in (season - 1, season)}
    res["history"] = historical_calibration()
    kk = kneel_then_kick(s, events, d)
    res["kneel_then_kick"] = {"drives": kk} | kneel_summary(kk)
    res["clock_kill_history"] = clock_kill_history(last=season)
    res["leader_kneels"] = leader_kneels(s, events)
    res["onside"] = onside_kicks(season)
    res["kickoff_start"] = kickoff_start(season)
    res["two_point"] = two_point(season)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path(season).write_text(json.dumps(res, indent=1, default=str))
    print("saved", out_path(season).relative_to(config.REPO_ROOT))
    return res


def flat(x, prefix: str = "") -> dict:
    """Every number in the result as one flat {path: value} dict, for checking prose."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out |= flat(v, f"{prefix}{k}/")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out |= flat(v, f"{prefix}{i}/")
    else:
        out[prefix.rstrip("/")] = x
    return out




# --------------------------------------------------------------------------- charts


def _figures():
    """make_figures.py's palette, style and save(), so these charts match the others."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_figures as mf

    mf.style()
    return mf


def _forest(ax, mf, rows: list[str], series: list[tuple], scale: float = 100) -> None:
    """Horizontal dot-and-interval rows: series = [(label, color, [interval dict per row])]."""
    k = len(series)
    for j, (label, color, vals) in enumerate(series):
        dy = ((k - 1) / 2 - j) * 0.22  # the first series on top
        for i, v in enumerate(vals):
            if v is None:
                continue
            y = len(rows) - 1 - i + dy
            ax.hlines(y, v["p05"] * scale, v["p95"] * scale, color=color, lw=2, alpha=0.55)
            ax.plot(v["mean"] * scale, y, "o", ms=6, color=color, label=label if i == 0 else None, zorder=3)
    ax.axvline(0, color=mf.INK2, lw=1, zorder=1)
    ax.set_yticks(range(len(rows)), list(reversed(rows)))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color=mf.GRID, lw=0.75)


def fig_late_gap(res: dict) -> Path:
    mf = _figures()
    plt = mf.plt
    names = [w[0] for w in WINDOWS]
    labels = ["15:00 to 10:00", "10:00 to 5:00", "5:00 to 2:00", "2:00 to 0:00"]
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    ax.axhline(0, color=mf.INK2, lw=1, zorder=1)
    series = ((-0.2, lambda n: res["gap"]["0"][n]["all"], mf.ORANGE, "every minute, no feed delay (δ = 0)"),
              (0.0, lambda n: res["gap"]["20"][n]["all"], mf.BLUE, "every minute, a 20-second feed delay (δ = 20 s)"),
              (0.2, lambda n: res["snaps"][n], mf.INK2, "at the snaps only, when both know the exact state"))
    for dx, get, color, label in series:
        xs = [i + dx for i in range(len(names))]
        mid = [get(n)["brier_corrected_diff"] * 1000 for n in names]
        lo = [get(n)["brier_corrected_diff_p05"] * 1000 for n in names]
        hi = [get(n)["brier_corrected_diff_p95"] * 1000 for n in names]
        ax.vlines(xs, lo, hi, color=color, lw=2, alpha=0.55)
        ax.plot(xs, mid, "o", ms=6, color=color, label=label)
    ax.set_xticks(range(len(names)), labels)
    ax.set_xlabel("time left in the 4th quarter")
    ax.set_ylabel("Brier score difference (× 1000)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncols=1, handlelength=1.2)
    titles_note = ("Brier score of gridline's win price minus Kalshi's (above zero: Kalshi more accurate), "
                   f"{res['season']}, with 90% intervals from resampling games.")
    mf.titles(fig, "In the last five minutes gridline trails Kalshi most between the snaps", titles_note)
    return mf.save(fig, "late_gap")


CAL_ROWS = (("last 5 minutes, lead 1-3", "last 5 min, the team leading by 1–3"),
            ("last 5 minutes, lead 4-8", "last 5 min, the team leading by 4–8"),
            ("last 2 minutes, offense tied or down 1-3", "last 2 min, the offense, tied or down 1–3"),
            ("last 2 minutes, offense down 4-8", "last 2 min, the offense, down 4–8"),
            ("last 2 minutes, offense ahead 1-8", "last 2 min, the offense, ahead 1–8"))


def fig_late_calibration(res: dict) -> Path:
    mf = _figures()
    plt = mf.plt
    lc, prev = res["lead_calibration"], res["sim_eval"][str(res["season"] - 1)]["simulator"]
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    fig.subplots_adjust(left=0.33)
    _forest(ax, mf, [label for _, label in CAL_ROWS], [
        (f"gridline, {res['season']}", mf.ORANGE, [lc[k]["model_minus_won"] for k, _ in CAL_ROWS]),
        (f"Kalshi, {res['season']}", mf.BLUE, [lc[k]["market_minus_won"] for k, _ in CAL_ROWS]),
        (f"gridline's simulator on {res['season'] - 1} (models through {res['season'] - 2}, sportsbook line)",
         mf.MUTED, [prev[k]["price_minus_won"] for k, _ in CAL_ROWS])])
    ax.set_xlabel("average price minus how often that team won (points; right of zero: priced too high)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncols=1, handlelength=1.2)
    mf.titles(fig, f"Against the outcomes: in {res['season']}, gridline was too sure of small late leads",
              "4th-quarter snaps, each price scored against how the game ended (a tie counts half); "
              "90% intervals from resampling games.")
    return mf.save(fig, "late_calibration")


DRIVE_ROWS = ("touchdown", "field goal made", "field goal missed", "punt", "turnover", "turnover on downs",
              "end of half")


def fig_late_drives(res: dict) -> Path:
    mf = _figures()
    plt = mf.plt
    w = "last 2 minutes, close"
    cur, prev = res["drives"][w], res["drives_previous_season"][w]
    mid = res["drives"]["1st to 3rd quarter"]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    fig.subplots_adjust(left=0.16)
    _forest(ax, mf, list(DRIVE_ROWS), [
        (f"last 2 minutes of close games, {res['season']} (model trained through {res['season'] - 1})", mf.ORANGE,
         [cur[o]["actual_minus_predicted"] for o in DRIVE_ROWS]),
        (f"1st to 3rd quarter, {res['season']}", mf.BLUE, [mid[o]["actual_minus_predicted"] for o in DRIVE_ROWS]),
        (f"last 2 minutes of close games, {res['season'] - 1} (model trained through {res['season'] - 2})", mf.MUTED,
         [prev[o]["actual_minus_predicted"] for o in DRIVE_ROWS])])
    ax.set_xlabel("how often drives ended that way minus the drive model's probability (percentage points)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncols=1, handlelength=1.2)
    mf.titles(fig, f"In {res['season']}'s final two minutes, drives ended in more made field goals than expected",
              "At every snap, how its drive ended minus the drive model's probability of that end; close games: "
              "margin of 8 or less. 90% intervals from resampling games.")
    return mf.save(fig, "late_drives")


def charts(res: dict) -> list[Path]:
    return [fig_late_gap(res), fig_late_calibration(res), fig_late_drives(res)]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--charts-only", action="store_true", help="redraw the charts from the saved results")
    a = ap.parse_args(argv)
    res = json.loads(out_path().read_text()) if a.charts_only else run()
    charts(res)


if __name__ == "__main__":
    main()
