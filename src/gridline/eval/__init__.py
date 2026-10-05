"""Evaluation (Phases 1, 3, 4). Nothing here yet.

Planned modules:
  metrics.py  calibration error exactly as nflfastR defines it, Brier, log loss,
              CRPS for score distributions, reliability tables.
  market.py   joins the price log to Kalshi with no lookahead (market state at t =
              last candle closed <= t, last trade <= t) and reports:
                - per-game paired Brier/log-loss differences, CIs by resampling games
                - information test: does (model - market) at t predict the
                  market's move over the next k minutes?
                - lead-lag after scoring plays (how fast does the price move?)
                - every result as a sweep over the publication delay delta
  integrity.py (Phase 4) residual z-scores / change points on market-vs-model
              moves that no game event explains; evaluated on injected anomalies.
"""
