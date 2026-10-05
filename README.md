# gridline

A live NFL in-play pricing engine, benchmarked against Kalshi's markets.

Every play updates a game state, a model turns that state into a distribution of
final scores, and the distribution prices every contract Kalshi lists for the
game: the winner, each spread rung and each total rung. The price log is then
compared with Kalshi's own prices, with no lookahead, to answer one question:
**during a game, does the model update faster or better than the market?**

It mirrors the core of what official-data companies like Genius Sports sell to
sportsbooks and prediction-market makers: live data turned into calibrated
prices. It is read-only by design. There are no keys and no order code, and it
uses only public market data.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Data: nflverse play-by-play + timestamps, Kalshi markets/candles/trades, game ↔ ticker catalog | **done**: all 285 games of 2025 timed, Kalshi data for 283+ |
| 1 | Win-probability baseline matching nflfastR | **done**: calibration error 0.0054 / 0.0067 vs nflfastR's 0.0055 / 0.0066; level with nflverse's model on 2025 ([results](docs/RESULTS.md)) |
| 2 | Drive simulator → joint final-score distribution → prices for all three market families | **done**: on 2025, final scores land in the predicted 50% / 90% ranges 49.4% / 88.4% of the time (margins); win probability level with Phase 1 ([results](docs/RESULTS.md#phase-2-a-simulator-that-prices-every-contract)) |
| 3 | Replay engine + benchmark vs Kalshi on 2025, then 2026 week by week | **done** for 2025: as accurate as Kalshi in play (Brier 0.1738 vs 0.1736); Kalshi makes half of a big play's move within 5 s of the NFL's timestamp, so only a faster feed has an edge ([results](docs/RESULTS.md#phase-3-gridline-against-kalshi-every-play-of-2025)) |
| 4 | Integrity monitor (market moves no game event explains) + write-up | |

**How it works, with pictures: [docs/PIPELINE.md](docs/PIPELINE.md).** Results so far: [docs/RESULTS.md](docs/RESULTS.md). Why it is built this way: [docs/DESIGN.md](docs/DESIGN.md). What the data looks like, verified: [docs/DATA.md](docs/DATA.md).

## Setup

```bash
# needs uv (https://docs.astral.sh/uv/) and, on macOS, OpenMP for XGBoost:
brew install uv libomp
uv sync                      # Python 3.12 venv + dependencies
uv run pytest                # tests

uv run gridline nflverse pull --seasons 1999-2026   # ~1 min, ~0.3 GB
uv run gridline events build --seasons 1999-2026    # ~1 min
uv run gridline kalshi probe                        # must run where Kalshi is reachable
uv run gridline kalshi pull --season 2025           # resumable; ~30-60 min
uv run gridline inventory --season 2025             # Phase 0 exit gate

uv run gridline wp cv                               # Phase 1 gate: nflfastR's protocol, ~10 min
uv run gridline wp test --train 2000-2024 --season 2025   # forward test vs nflverse's model

uv run gridline drives build --seasons 2001-2026    # drive outcomes for every snap, ~1 min
uv run gridline sim train --last 2024               # drive model + transitions, ~5 min
uv run gridline sim gate --season 2025              # Phase 2 gate: every 2025 snap simulated
uv run gridline sim price --game 2025_22_SEA_NE     # fair prices for one game state

uv run gridline kalshi consolidate --season 2025    # one table each: markets, candles, trades
uv run gridline replay run --season 2025            # Phase 3: price every play of 2025, ~50 min on 2 cores
uv run gridline market benchmark --season 2025      # the report: gridline against Kalshi
```

Data lands in `data/` (git-ignored). Set `GRIDLINE_DATA_DIR` to put it elsewhere.

## Layout

```
src/gridline/
  data/       nflverse ingest, read-only Kalshi client, game <-> ticker catalog, pull
  state/      play events (cleaned wall-clock timing), GameState
  models/     win probability; drive data, drive model, transitions, simulator (Phases 1-2)
  pricing/    simulated scores -> fair prices for every contract       (Phase 2)
  engine/     event source -> state -> model -> pricer -> price log; the prior (Phase 3)
  eval/       metrics, game bootstrap, distribution scores, simulator gate, market series and
              the benchmark against Kalshi; integrity to come          (Phases 1-4)
  inventory.py, cli.py
tests/        unit tests, real Kalshi response shapes as fixtures
docs/         PIPELINE.md (illustrated walkthrough), RESULTS.md (numbers), DESIGN.md (decisions),
              DATA.md (field notes)
scripts/      make_figures.py (rebuilds the data charts in docs/img)
```

## Data and attribution

Play-by-play and schedules come from [nflverse](https://github.com/nflverse) (CC-BY 4.0).
Market data comes from Kalshi's public API; raw Kalshi data is never committed. Only
derived results are published.
