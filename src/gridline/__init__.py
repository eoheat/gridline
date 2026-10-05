"""gridline: a live NFL in-play pricing engine, benchmarked against Kalshi.

Pipeline: play event -> GameState -> model -> final-score distribution
          -> fair price for every contract -> price log -> evaluation vs Kalshi.

See docs/DESIGN.md for the reasoning behind the structure.
"""

__version__ = "0.1.0"
