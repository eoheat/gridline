"""Phase 2 gate: is the simulator's distribution of final scores calibrated, and does
its win probability agree with the direct model (Phase 1)?

For every game of a test season: solve the two knobs at kickoff so that the simulated
mean margin and total match the closing line (nflverse's spread_line and total_line),
then simulate from the pregame state and from every k-th scrimmage snap in regulation.
Each state gets a predicted distribution of the final margin and final total, scored
with CRPS and PIT (eval/distribution.py) against:

  a normal baseline: final = current + line * f, sd = sigma * sqrt(f), where f is the
  share of the game left (Stern 1994); sigma is fitted on the training seasons.

The simulated win probability (ties count half, as Kalshi settles them) is compared with
the Phase 1 model's on the same snaps.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from gridline import config
from gridline.eval import bootstrap, distribution as dist, metrics
from gridline.models import simulator as sim

Log = Callable[[str], None]

# The gate (fixed before the test season was run; see DESIGN.md):
GATE_COVERAGE_TOL = 0.03  # 50% and 90% interval coverage within 3 points of nominal
GATE_WP_MAE = 0.03  # mean |simulated WP - direct WP|
GATE_WP_LOGLOSS = 0.01  # simulated WP log loss at most this much above the direct model's


def eval_path(season: int) -> Path:
    return config.DERIVED_DIR / "sim_eval" / f"sim_eval_{season}.parquet"


def fit_sigmas(seasons) -> tuple[float, float]:
    """SD of (final margin - spread line) and (final total - total line), one row per game."""
    from gridline.inventory import load_events

    g = pl.concat([load_events(s).group_by("game_id").first() for s in seasons], how="diagonal_relaxed")
    g = g.filter(pl.col("pregame_spread").is_not_null() & pl.col("pregame_total").is_not_null())
    return (float((g["final_margin_home"] - g["pregame_spread"]).std()),
            float((g["final_total"] - g["pregame_total"]).std()))


def _seed(game_id: str) -> int:
    return zlib.crc32(game_id.encode())


def _state_record(r: sim.Result, s, y_margin: float, y_total: float, f: float, sigmas, v) -> dict:
    cm = dist.cdf_from_samples(r.margin, dist.MARGIN_GRID)
    ct = dist.cdf_from_samples(r.total, dist.TOTAL_GRID)
    mu_m = s.home_score - s.away_score + (s.pregame_spread or 0.0) * f
    mu_t = s.home_score + s.away_score + (s.pregame_total or 44.0) * f
    bm = dist.cdf_normal(mu_m, sigmas[0] * np.sqrt(max(f, 1e-4)), dist.MARGIN_GRID)
    bt = dist.cdf_normal(mu_t, sigmas[1] * np.sqrt(max(f, 1e-4)), dist.TOTAL_GRID)
    return {
        "wp_sim_home": float(np.mean(r.margin > 0) + 0.5 * np.mean(r.margin == 0)),
        "p_tie": float(np.mean(r.margin == 0)),
        "margin_mean": float(r.margin.mean()), "total_mean": float(r.total.mean()),
        # the two pregame lines as contracts: home covers, and the game goes over
        "p_home_cover": float(np.mean(r.margin > (s.pregame_spread or 0.0))),
        "p_over": float(np.mean(r.total > (s.pregame_total or 44.0))),
        "crps_margin": dist.crps(cm, dist.MARGIN_GRID, y_margin),
        "crps_total": dist.crps(ct, dist.TOTAL_GRID, y_total),
        "pit_margin": dist.pit(cm, dist.MARGIN_GRID, y_margin, v[0]),
        "pit_total": dist.pit(ct, dist.TOTAL_GRID, y_total, v[1]),
        "crps_margin_base": dist.crps(bm, dist.MARGIN_GRID, y_margin),
        "crps_total_base": dist.crps(bt, dist.TOTAL_GRID, y_total),
        "pit_margin_base": dist.pit(bm, dist.MARGIN_GRID, y_margin, v[0]),
        "pit_total_base": dist.pit(bt, dist.TOTAL_GRID, y_total, v[1]),
    }


def _game_task(args) -> list[dict]:
    rows, engine_season, wp_path, n, n_knob, every, sigmas = args
    import xgboost as xgb

    from gridline.models import features, wp
    from gridline.state.game_state import GameState

    engine = _ENGINES.setdefault(engine_season, sim.Engine.load(engine_season))
    if wp_path not in _BOOSTERS:
        b = xgb.Booster()
        b.load_model(wp_path)
        _BOOSTERS[wp_path] = b
    booster = _BOOSTERS[wp_path]
    game_id = rows["game_id"][0]
    first = rows.row(0, named=True)
    y_margin, y_total = float(first["final_margin_home"]), float(first["final_total"])
    s0 = sim.kickoff_state(GameState.pre_snap(first))
    spread = s0.pregame_spread if s0.pregame_spread is not None else 0.0
    total = s0.pregame_total if s0.pregame_total is not None else 44.0
    t0 = time.time()
    knobs = sim.solve_knobs(engine, s0, spread, total, n=n_knob, seed=_seed(game_id) + 1)
    u = sim.uniforms(n, _seed(game_id))
    rng = np.random.default_rng(_seed(game_id) + 2)
    base = {"game_id": game_id, "season": first["season"], "season_type": first["season_type"],
            "final_margin": y_margin, "final_total": y_total, "spread_line": spread, "total_line": total,
            "knob_spread": knobs[0], "knob_total": knobs[1]}
    out = []
    r = sim.simulate(engine, s0, n, knobs, u, phase="kickoff", kicker_home=None)
    out.append(base | {"seq": -1, "kind": "pregame", "qtr": 1, "game_seconds_remaining": 3600.0,
                       "posteam_is_home": None, "wp_direct_home": None}
               | _state_record(r, s0, y_margin, y_total, 1.0, sigmas, rng.random(2)))
    feats = features.wp_features(rows)
    feats = feats.with_columns(direct=pl.Series(wp.predict(booster, feats, "wp_spread"))).select("seq", "direct")
    snaps = (rows.filter(pl.col("down").is_not_null() & pl.col("posteam").is_not_null()
                         & pl.col("yardline_100").is_not_null() & (pl.col("qtr") <= 4))
             .join(feats, on="seq", how="left"))
    for i, row in enumerate(snaps.iter_rows(named=True)):
        v = rng.random(2)  # drawn for every snap so the PITs don't depend on `every`
        if i % every:
            continue
        s = GameState.pre_snap(row)
        r = sim.simulate(engine, s, n, knobs, u)
        home = row["posteam"] == row["home_team"]
        d = row["direct"]
        out.append(base | {"seq": row["seq"], "kind": "snap", "qtr": row["qtr"],
                           "game_seconds_remaining": row["game_seconds_remaining"],
                           "posteam_is_home": home,
                           "wp_direct_home": None if d is None else (d if home else 1 - d)}
                   | _state_record(r, s, y_margin, y_total, row["game_seconds_remaining"] / 3600, sigmas, v))
    for rec in out:
        rec["seconds_spent"] = (time.time() - t0) / len(out)
    return out


_ENGINES: dict = {}
_BOOSTERS: dict = {}


def evaluate(season: int, engine_season: int, wp_path: str | Path, n: int = 2000, n_knob: int = 4000,
             every: int = 1, processes: int = 2, games: int | None = None, log: Log = print) -> pl.DataFrame:
    from gridline.inventory import load_events

    ev = load_events(season)
    sigmas = fit_sigmas(range(engine_season - 18, engine_season + 1))
    log(f"normal baseline: sd {sigmas[0]:.2f} (margin), {sigmas[1]:.2f} (total), fitted on "
        f"{engine_season - 18}-{engine_season}")
    ids = ev["game_id"].unique(maintain_order=True).to_list()[:games]
    tasks = [(ev.filter(pl.col("game_id") == g), engine_season, str(wp_path), n, n_knob, every, sigmas) for g in ids]
    t0 = time.time()
    out = []
    # one BLAS thread per worker: the workers already use every core
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    ctx = mp.get_context("spawn")  # polars and OpenMP thread pools do not survive fork()
    with ctx.Pool(processes) as pool:
        for i, recs in enumerate(pool.imap_unordered(_game_task, tasks), 1):
            out.extend(recs)
            if i % 25 == 0 or i == len(tasks):
                log(f"  {i}/{len(tasks)} games, {len(out):,} states, {time.time() - t0:.0f}s")
    df = pl.DataFrame(out, infer_schema_length=None).sort("game_id", "seq")
    path = eval_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    log(f"  saved {path}")
    return df


def summarize(df: pl.DataFrame, log: Log = print) -> dict:
    """The gate's numbers, with 90% intervals from resampling games."""
    games = df["game_id"].to_numpy()
    res = {"states": df.height, "games": df["game_id"].n_unique()}
    for what in ("margin", "total"):
        for who, suffix in (("sim", ""), ("base", "_base")):
            p = df[f"pit_{what}{suffix}"].to_numpy()
            res[f"{what}_{who}_cov50"] = dist.coverage(p, 0.5)
            res[f"{what}_{who}_cov90"] = dist.coverage(p, 0.9)
            res[f"{what}_{who}_crps"] = float(df[f"crps_{what}{suffix}"].mean())
        a, b = df[f"crps_{what}"].to_numpy(), df[f"crps_{what}_base"].to_numpy()
        d = bootstrap.spread(lambda w: float(np.average(a - b, weights=w)), games)
        res[f"{what}_crps_diff"] = float(np.mean(a - b))
        res[f"{what}_crps_diff_p05"], res[f"{what}_crps_diff_p95"] = d["p05"], d["p95"]
    snaps = df.filter((pl.col("kind") == "snap") & pl.col("wp_direct_home").is_not_null()
                      & (pl.col("final_margin") != 0))
    y = (snaps["final_margin"] > 0).to_numpy().astype(float)
    ws, wd = snaps["wp_sim_home"].to_numpy(), snaps["wp_direct_home"].to_numpy()
    q = snaps["qtr"].to_numpy()
    res["wp_states"] = snaps.height
    res["wp_mae"] = float(np.mean(np.abs(ws - wd)))
    res["wp_sim"] = metrics.summary(ws, y, q)
    res["wp_direct"] = metrics.summary(wd, y, q)
    res["seconds_per_state"] = float(df["seconds_spent"].mean())
    return res


def report(res: dict, log: Log = print) -> bool:
    ok_cov = all(abs(res[f"{w}_sim_cov{lvl}"] - lvl / 100) <= GATE_COVERAGE_TOL
                 for w in ("margin", "total") for lvl in (50, 90))
    ok_crps = res["margin_crps_diff"] < 0 and res["total_crps_diff"] < 0
    ok_wp = (res["wp_mae"] <= GATE_WP_MAE
             and res["wp_sim"]["log_loss"] <= res["wp_direct"]["log_loss"] + GATE_WP_LOGLOSS)
    log(f"{res['games']} games, {res['states']:,} states ({res['seconds_per_state'] * 1000:.0f} ms each)")
    for w in ("margin", "total"):
        log(f"  {w}: coverage of the 50% / 90% intervals: simulator {res[f'{w}_sim_cov50']:.3f} / "
            f"{res[f'{w}_sim_cov90']:.3f}, normal {res[f'{w}_base_cov50']:.3f} / {res[f'{w}_base_cov90']:.3f}")
        log(f"  {w}: CRPS simulator {res[f'{w}_sim_crps']:.3f}, normal {res[f'{w}_base_crps']:.3f}, difference "
            f"{res[f'{w}_crps_diff']:+.3f} [{res[f'{w}_crps_diff_p05']:+.3f}, {res[f'{w}_crps_diff_p95']:+.3f}]")
    s, d = res["wp_sim"], res["wp_direct"]
    log(f"  win probability on {res['wp_states']:,} snaps: mean |simulated - direct| {res['wp_mae']:.4f}; "
        f"log loss {s['log_loss']:.4f} vs {d['log_loss']:.4f}, Brier {s['brier']:.4f} vs {d['brier']:.4f}, "
        f"calibration error {s['calibration_error']:.4f} vs {d['calibration_error']:.4f}")
    ok = ok_cov and ok_crps and ok_wp
    log(f"Phase 2 gate: coverage {'ok' if ok_cov else 'NO'}, CRPS {'ok' if ok_crps else 'NO'}, "
        f"win probability {'ok' if ok_wp else 'NO'} -> {'PASS' if ok else 'NOT YET'}")
    return ok
