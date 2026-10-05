"""nflverse ingest: play-by-play and schedules, cached as parquet under data/raw/nflverse.

nflverse data is CC-BY 4.0 (attribute "nflverse" when publishing results).
The raw cache stores exactly what nflreadpy returns; cleaning happens in
gridline.state.events so it is versioned with the code, not baked into files.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import polars as pl

from gridline import config

Log = Callable[[str], None]


def pbp_path(season: int) -> Path:
    return config.NFLVERSE_DIR / "pbp" / f"pbp_{season}.parquet"


def schedules_path() -> Path:
    return config.NFLVERSE_DIR / "schedules.parquet"


def current_season() -> int:
    import nflreadpy as nfl

    return int(nfl.get_current_season())


def _atomic_write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def pull_pbp(seasons: Iterable[int], force: bool = False, log: Log = print) -> list[Path]:
    """Download play-by-play for `seasons`. Past seasons are cached; the current
    season is always refreshed because nflverse adds games every week."""
    import nflreadpy as nfl

    cur = current_season()
    paths = []
    for season in seasons:
        path = pbp_path(season)
        if path.exists() and not force and season != cur:
            log(f"pbp {season}: cached")
        else:
            df = nfl.load_pbp([season])
            _atomic_write(df, path)
            log(f"pbp {season}: {df.height:,} rows, {df['game_id'].n_unique()} games")
        paths.append(path)
    return paths


def pull_schedules(log: Log = print) -> Path:
    """Download the full schedule (all seasons, with closing spread/total lines).
    Small, so always refreshed."""
    import nflreadpy as nfl

    df = nfl.load_schedules()
    _atomic_write(df, schedules_path())
    log(f"schedules: {df.height:,} games, seasons {df['season'].min()}-{df['season'].max()}")
    return schedules_path()


def load_pbp(seasons: Sequence[int], columns: Sequence[str] | None = None) -> pl.DataFrame:
    missing = [s for s in seasons if not pbp_path(s).exists()]
    if missing:
        raise FileNotFoundError(
            f"no cached play-by-play for seasons {missing}; run `gridline nflverse pull`"
        )
    frames = [
        pl.read_parquet(pbp_path(s), columns=list(columns) if columns else None) for s in seasons
    ]
    # column dtypes drift across seasons (f64 vs i32 etc.); relax to supertypes
    return pl.concat(frames, how="diagonal_relaxed")


def load_schedules(seasons: Sequence[int] | None = None) -> pl.DataFrame:
    if not schedules_path().exists():
        raise FileNotFoundError("no cached schedules; run `gridline nflverse pull`")
    df = pl.read_parquet(schedules_path())
    if seasons is not None:
        df = df.filter(pl.col("season").is_in(list(seasons)))
    return df


def kickoff_utc(gameday: str, gametime: str | None) -> dt.datetime:
    """nflverse gameday/gametime are US Eastern wall-clock; convert to UTC.
    A missing gametime falls back to 13:00 ET (the most common slot)."""
    hh, mm = (gametime or "13:00").split(":")[:2]
    local = dt.datetime.fromisoformat(gameday).replace(
        hour=int(hh), minute=int(mm), tzinfo=ZoneInfo(config.EASTERN_TZ)
    )
    return local.astimezone(dt.timezone.utc)
