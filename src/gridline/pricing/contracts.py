"""Fair prices for Kalshi's NFL game contracts, read off simulated final scores
(DESIGN.md decision 4). One set of simulated games prices every contract, so the
prices are consistent with each other by construction.

  winner  YES pays if the team wins; a tie settles at 50/50, so fair YES = P(win) + P(tie) / 2
  spread  "<team> wins by over N.5 points": YES = P(team's margin > N.5)
  total   "over N.5 points scored": YES = P(total > N.5)

Every strike is a half point, so no contract can push.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Side = Literal["home", "away"]


@dataclass(frozen=True)
class ScoreDistribution:
    home: np.ndarray  # simulated final scores
    away: np.ndarray

    @property
    def margin(self) -> np.ndarray:
        return self.home - self.away

    @property
    def total(self) -> np.ndarray:
        return self.home + self.away

    def winner(self, side: Side = "home") -> float:
        m = self.margin if side == "home" else -self.margin
        return float(np.mean(m > 0) + 0.5 * np.mean(m == 0))

    def spread(self, side: Side, strike: float) -> float:
        """P(side wins by more than strike)."""
        m = self.margin if side == "home" else -self.margin
        return float(np.mean(m > strike))

    def over(self, strike: float) -> float:
        return float(np.mean(self.total > strike))

    def price(self, kind: str, side: Side | None, strike: float | None) -> float:
        """Fair YES price of a market as classified by data.catalog.classify_market."""
        if kind == "winner":
            return self.winner(side or "home")
        if kind == "spread":
            return self.spread(side or "home", float(strike))
        if kind == "total":
            return self.over(float(strike))
        raise ValueError(f"unknown contract kind {kind!r}")

    def ladder(self, kind: Literal["spread", "total"], strikes, side: Side = "home") -> np.ndarray:
        return np.array([self.spread(side, k) if kind == "spread" else self.over(k) for k in strikes])
