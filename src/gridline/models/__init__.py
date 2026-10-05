"""Models of the game (Phases 1-2). Nothing here yet.

Planned modules:
  features.py  whitelisted pre-snap features; a test fails if any feature is
               derived from a post-play column (total_*_score, *_post, wpa, result...).
  wp.py        direct win probability (XGBoost, nflfastR feature parity).
               Phase 1 gate: calibration error <= 0.0055 under nflfastR's
               leave-one-season-out protocol.
  drives.py    "probability of key actions": for the drive in progress and every
               future drive, P(outcome | down, distance, field position, clock,
               score, timeouts, era) with outcomes TD / FG / missed FG / punt /
               turnover / turnover on downs / safety / end of half, plus clock
               used and the next drive's starting field position.
  sim.py       vectorised Monte Carlo from any GameState to the final whistle
               (overtime rules included) -> joint final-score distribution.
               Two per-game knobs (strength gap, scoring rate) are solved at
               kickoff so the simulator reproduces the pregame prior.
"""
