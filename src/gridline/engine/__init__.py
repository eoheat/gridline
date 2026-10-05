"""Event-driven pricing engine (Phase 3).

One code path for replay and live:

    EventSource -> GameState -> simulator -> prices -> price log

  sources.py  events (pregame, play, snap, final) at the moment each became public.
              ReplaySource builds them from the play events table; a live feed would
              emit the same Event objects.
  prior.py    the prior: Kalshi's last prices before kickoff.
  engine.py   GamePricer (knobs solved at kickoff, a Quote at every event) and replay().
  sinks.py    the price log: one parquet file per game, one row per event.

The engine sees one market price per game, the prior. Everything after kickoff is joined
only in evaluation (eval/benchmark.py), so the model cannot learn from the market it is
judged against.
"""
