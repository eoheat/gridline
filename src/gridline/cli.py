"""Command line: `gridline <area> <action>`.

  gridline nflverse pull [--seasons 1999-2026] [--force]
  gridline events build --seasons 2025,2026
  gridline kalshi probe                      # connectivity + both tiers, ~6 requests
  gridline kalshi pull --season 2025 [--no-trades] [--max-games N] [--rate 10] [--redo GAME ...]
  gridline kalshi consolidate --season 2025  # one file per table, for analysis
  gridline inventory --season 2025           # Phase 0 exit gate
  gridline wp cv [--model wp_spread] [--seasons 2000-2019]      # Phase 1 gate: nflfastR's protocol
  gridline wp test [--model wp_spread] --train 2000-2024 --season 2025   # forward test
  gridline drives build --seasons 2001-2026  # drive-level training data (Phase 2)
  gridline sim train --last 2024             # drive-step model + transition pools
  gridline sim gate --season 2025 [--every 1] [--n 2000]   # Phase 2 gate
  gridline sim price --game 2025_22_SEA_NE [--seq N] [--n 20000]   # price one state
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

from gridline import config


def parse_seasons(spec: str) -> list[int]:
    """'1999-2026' | '2025,2026' | '2025' -> list of seasons."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out.extend(range(a, b + 1))
        elif part:
            out.append(int(part))
    return out


def _nflverse_pull(args: argparse.Namespace) -> None:
    from gridline.data import nflverse

    seasons = parse_seasons(args.seasons or f"{config.FIRST_PBP_SEASON}-{nflverse.current_season()}")
    nflverse.pull_schedules()
    nflverse.pull_pbp(seasons, force=args.force)


def _events_build(args: argparse.Namespace) -> None:
    from gridline.inventory import build_events

    build_events(parse_seasons(args.seasons))


def _kalshi_probe(args: argparse.Namespace) -> None:
    from gridline.data.kalshi import KalshiPublic

    c = KalshiPublic()
    cut = c.cutoff()
    print("historical cutoff:", {k: v.isoformat() for k, v in cut.items()})
    for label, event in [("historical tier", "KXNFLGAME-25SEP07NYGWAS"), ("live tier", None)]:
        if event is None:  # most recently settled game -> live tier after the cutoff
            recent = c.get("/events", {"series_ticker": config.GAME_SERIES,
                                       "status": "settled", "limit": 1})["events"]
            event = recent[0]["event_ticker"]
        markets = c.markets(event)
        m = markets[0]
        end = m["close_time"]
        candles = c.candles(m, end - dt.timedelta(hours=4), end)
        quoted = [x for x in candles if x["mid_close"] is not None]
        print(f"{label}: {event} -> {len(markets)} markets; {m['ticker']} "
              f"historical={c.market_is_historical(m)} candles={len(candles)} quoted={len(quoted)}")
        if quoted:
            q = quoted[len(quoted) // 2]
            print(f"   sample: {q['end_ts']:%Y-%m-%d %H:%M}Z bid {q['bid_close']} "
                  f"ask {q['ask_close']} mid {q['mid_close']:.3f} vol {q['volume']}")
    print(f"ok ({c.n_requests} requests)")


def _kalshi_pull(args: argparse.Namespace) -> None:
    from gridline.data.kalshi import KalshiPublic
    from gridline.data.kalshi_pull import pull_season

    client = KalshiPublic(min_interval_s=1.0 / args.rate)
    failures = pull_season(args.season, client, trades=not args.no_trades,
                           max_games=args.max_games, redo=set(args.redo or []))
    sys.exit(1 if failures else 0)


def _kalshi_consolidate(args: argparse.Namespace) -> None:
    from gridline.data.kalshi_pull import consolidate

    consolidate(args.season)


def _wp_cv(args: argparse.Namespace) -> None:
    from gridline.models import wp

    seasons = parse_seasons(args.seasons)
    df = wp.training_frame(seasons)
    kinds = ["wp", "wp_spread"] if args.model == "both" else [args.model]
    results = {}
    for k in kinds:
        cv = wp.loso_cv(df, k, seasons)
        results[k] = wp.report(cv, k)
        print(f"  held-out predictions saved to {wp.save_cv(cv, k)}")
    ok = all(results[k]["calibration_error"] <= wp.MODELS[k]["published"]["calibration_error"]
             + wp.GATE_TOLERANCE for k in kinds)
    print(f"Phase 1 gate (calibration error within {wp.GATE_TOLERANCE} of nflfastR's): "
          f"{'PASS' if ok else 'NOT YET'}")
    sys.exit(0 if ok else 1)


def _wp_test(args: argparse.Namespace) -> None:
    from gridline.models import wp

    wp.forward_test(parse_seasons(args.train), args.season, args.model)


def _drives_build(args: argparse.Namespace) -> None:
    from gridline.models import drives

    drives.build_seasons(parse_seasons(args.seasons))


def _sim_train(args: argparse.Namespace) -> None:
    from gridline.models import drive_model, drives, transitions

    last = args.last
    seasons = range(last - args.history + 1, last + 1)
    snaps = drives.load_snaps(seasons)
    print(f"drive-step model: seasons {seasons.start}-{last}, {args.epochs} epochs")
    model = drive_model.train(snaps, epochs=args.epochs)
    print(f"  saved {model.save(drive_model.step_model_path(last))}")
    t = transitions.build(drives.load_drives(seasons), snaps, last)
    print(f"  saved {t.save(transitions.transitions_path(last))}")


def _sim_gate(args: argparse.Namespace) -> None:
    from gridline.eval import sim_gate
    from gridline.models import wp

    last = args.season - 1
    wp_path = wp.model_dir() / f"wp_spread_2000_{last}.json"
    if not wp_path.exists():
        print(f"training the direct model for comparison: {wp_path.name}")
        wp.save(wp.fit(wp.training_frame(range(2000, last + 1)), "wp_spread"), "wp_spread", range(2000, last + 1))
    df = sim_gate.evaluate(args.season, last, wp_path, n=args.n, every=args.every, processes=args.processes)
    sys.exit(0 if sim_gate.report(sim_gate.summarize(df)) else 1)


def _sim_price(args: argparse.Namespace) -> None:
    import numpy as np
    import polars as pl

    from gridline.inventory import load_events
    from gridline.models import simulator as sim
    from gridline.pricing.contracts import ScoreDistribution
    from gridline.state.game_state import GameState

    season = int(args.game[:4])
    ev = load_events(season).filter(pl.col("game_id") == args.game)
    if ev.height == 0:
        sys.exit(f"no events for {args.game}")
    engine = sim.Engine.load(args.engine or season - 1)
    s0 = sim.kickoff_state(GameState.pre_snap(ev.row(0, named=True)))
    knobs = sim.solve_knobs(engine, s0, s0.pregame_spread or 0.0, s0.pregame_total or 44.0, n=args.n)
    if args.seq is None:
        s, phase, label = s0, "kickoff", "before kickoff"
    else:
        row = ev.filter(pl.col("seq") == args.seq).row(0, named=True)
        s, phase, label = GameState.pre_snap(row), "snap", row["desc"]
    r = sim.simulate(engine, s, args.n, knobs, phase=phase, kicker_home=None,
                     ot_possessions=sim.overtime_possessions(ev, args.seq) if args.seq is not None else 0)
    d = ScoreDistribution(r.home, r.away)
    print(f"{args.game}: {label}")
    print(f"  line: {s.home} {-(s0.pregame_spread or 0.0):+g}, total {s0.pregame_total:g}; "
          f"knobs {knobs[0]:+.2f}, {knobs[1]:.2f}")
    print(f"  {s.home} {s.home_score} - {s.away} {s.away_score}; {args.n:,} simulated games, "
          f"{np.mean(r.overtime):.1%} to overtime")
    print(f"  winner: {s.home} {d.winner('home'):.3f}, {s.away} {d.winner('away'):.3f}")
    fav = "home" if d.margin.mean() >= 0 else "away"
    team = s.home if fav == "home" else s.away
    for k in np.arange(0.5, 15, 3.0):
        print(f"  {team} wins by over {k:4.1f}: {d.spread(fav, k):.3f}")
    mid = round(float(d.total.mean()))
    for k in np.arange(mid - 9.5, mid + 10, 3.0):
        print(f"  total over {k:5.1f}: {d.over(k):.3f}")


def _replay_run(args: argparse.Namespace) -> None:
    from gridline.engine import engine
    from gridline.models import wp

    last = args.engine or args.season - 1
    wp_path = wp.model_dir() / f"wp_spread_2000_{last}.json"
    if not wp_path.exists():
        print(f"training the direct model: {wp_path.name}")
        wp.save(wp.fit(wp.training_frame(range(2000, last + 1)), "wp_spread"), "wp_spread", range(2000, last + 1))
    engine.replay(args.season, last, wp_path, n=args.n, processes=args.processes, games=args.games, run=args.run)


def _market_benchmark(args: argparse.Namespace) -> None:
    from gridline.engine import engine
    from gridline.eval import benchmark

    benchmark.run(args.season, args.run or engine.run_name(args.season))


def _inventory(args: argparse.Namespace) -> None:
    from gridline.inventory import phase0_report

    ok = [phase0_report(s) for s in parse_seasons(args.season)]
    sys.exit(0 if all(ok) else 1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="gridline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    area = p.add_subparsers(dest="area", required=True)

    nv = area.add_parser("nflverse").add_subparsers(dest="action", required=True)
    a = nv.add_parser("pull", help="download play-by-play + schedules")
    a.add_argument("--seasons", help="e.g. 1999-2026 (default: all)")
    a.add_argument("--force", action="store_true", help="re-download cached seasons")
    a.set_defaults(fn=_nflverse_pull)

    ev = area.add_parser("events").add_subparsers(dest="action", required=True)
    a = ev.add_parser("build", help="clean + time play events from cached pbp")
    a.add_argument("--seasons", required=True)
    a.set_defaults(fn=_events_build)

    k = area.add_parser("kalshi").add_subparsers(dest="action", required=True)
    a = k.add_parser("probe", help="check connectivity and both data tiers")
    a.set_defaults(fn=_kalshi_probe)
    a = k.add_parser("pull", help="pull a season's markets, candles, trades")
    a.add_argument("--season", type=int, required=True)
    a.add_argument("--no-trades", action="store_true")
    a.add_argument("--max-games", type=int)
    a.add_argument("--rate", type=float, default=10.0, help="max requests per second")
    a.add_argument("--redo", nargs="+", metavar="GAME_ID", help="re-pull these games even if DONE")
    a.set_defaults(fn=_kalshi_pull)
    a = k.add_parser("consolidate", help="one file per table for a pulled season (for analysis)")
    a.add_argument("--season", type=int, required=True)
    a.set_defaults(fn=_kalshi_consolidate)

    w = area.add_parser("wp").add_subparsers(dest="action", required=True)
    a = w.add_parser("cv", help="leave-one-season-out CV with nflfastR's protocol (Phase 1 gate)")
    a.add_argument("--model", choices=["wp", "wp_spread", "both"], default="both")
    a.add_argument("--seasons", default="2000-2019")
    a.set_defaults(fn=_wp_cv)
    a = w.add_parser("test", help="train on some seasons, test on a later one vs nflverse's model")
    a.add_argument("--model", choices=["wp", "wp_spread"], default="wp_spread")
    a.add_argument("--train", default="2000-2024")
    a.add_argument("--season", type=int, default=2025)
    a.set_defaults(fn=_wp_test)

    dv = area.add_parser("drives").add_subparsers(dest="action", required=True)
    a = dv.add_parser("build", help="drive results, clock and transitions for every snap")
    a.add_argument("--seasons", required=True)
    a.set_defaults(fn=_drives_build)

    sm = area.add_parser("sim").add_subparsers(dest="action", required=True)
    a = sm.add_parser("train", help="fit the drive-step model and transition pools")
    a.add_argument("--last", type=int, default=2024, help="last training season")
    a.add_argument("--history", type=int, default=19, help="seasons of history (default 19)")
    a.add_argument("--epochs", type=int, default=30)
    a.set_defaults(fn=_sim_train)
    a = sm.add_parser("gate", help="Phase 2 gate on a test season (models from the season before)")
    a.add_argument("--season", type=int, default=2025)
    a.add_argument("--every", type=int, default=1, help="use every k-th snap")
    a.add_argument("--n", type=int, default=2000, help="simulated games per state")
    a.add_argument("--processes", type=int, default=os.cpu_count() or 2, help="worker processes")
    a.set_defaults(fn=_sim_gate)
    a = sm.add_parser("price", help="fair prices for one state of a game")
    a.add_argument("--game", required=True)
    a.add_argument("--seq", type=int, help="events row (default: before kickoff)")
    a.add_argument("--n", type=int, default=20_000)
    a.add_argument("--engine", type=int, help="last training season of the models (default: season - 1)")
    a.set_defaults(fn=_sim_price)

    rp = area.add_parser("replay").add_subparsers(dest="action", required=True)
    a = rp.add_parser("run", help="price every event of a season's games into a price log (Phase 3)")
    a.add_argument("--season", type=int, default=2025)
    a.add_argument("--n", type=int, default=2000, help="simulated games per event")
    a.add_argument("--processes", type=int, default=os.cpu_count() or 2, help="worker processes")
    a.add_argument("--games", type=int, help="only the first N games (a quick check)")
    a.add_argument("--engine", type=int, help="last training season of the models (default: season - 1)")
    a.add_argument("--run", help="price log folder under data/logs (default: replay_<season>)")
    a.set_defaults(fn=_replay_run)

    mk = area.add_parser("market").add_subparsers(dest="action", required=True)
    a = mk.add_parser("benchmark", help="the Phase 3 report: model against Kalshi, from a replay's price log")
    a.add_argument("--season", type=int, default=2025)
    a.add_argument("--run", help="price log folder under data/logs (default: replay_<season>)")
    a.set_defaults(fn=_market_benchmark)

    a = area.add_parser("inventory", help="Phase 0 data inventory + exit gate")
    a.add_argument("--season", default="2025")
    a.set_defaults(fn=_inventory)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
