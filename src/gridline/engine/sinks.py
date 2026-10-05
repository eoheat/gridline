"""Price log: an append-only record of every price the engine published.

A replay run is a folder, data/logs/<run>/, holding one parquet file per game
(prices/<game_id>.parquet). Each file is written once, when its game is done, through a
temporary file and a rename, so an interrupted run resumes where it stopped and never
leaves half a game behind.

One row per event (engine/sources.py): when it happened, the state the engine priced,
and the prices. The distribution is stored whole, as the CDF of the final margin and
total on integer grids, so any contract can be priced afterwards (engine.Quote.price).
The prior and the knobs solved from it repeat on every row of a game.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from gridline import config


def run_dir(run: str) -> Path:
    return config.LOGS_DIR / run


class PriceLog:
    def __init__(self, run: str):
        self.dir = run_dir(run) / "prices"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, game_id: str) -> Path:
        return self.dir / f"{game_id}.parquet"

    def done(self) -> set[str]:
        return {p.stem for p in self.dir.glob("*.parquet")}

    def write(self, game_id: str, rows: pl.DataFrame) -> Path:
        path = self.path(game_id)
        tmp = path.with_suffix(".parquet.tmp")
        rows.write_parquet(tmp)
        os.replace(tmp, path)
        return path


def load(run: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Every game of a run, sorted by game and time."""
    files = sorted((run_dir(run) / "prices").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no price log for run {run!r}; run `gridline replay run`")
    lf = pl.scan_parquet(files)
    if columns:
        lf = lf.select(list(dict.fromkeys(["game_id", "order", *columns])))
    return lf.collect().sort("game_id", "order")
