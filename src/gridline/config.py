"""Paths and project-wide constants.

Every path hangs off DATA_DIR so the whole data tree can be moved by setting
GRIDLINE_DATA_DIR (e.g. to an external drive).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRIDLINE_DATA_DIR", REPO_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"          # exactly what the sources returned (immutable)
DERIVED_DIR = DATA_DIR / "derived"  # rebuilt by code from raw/
LOGS_DIR = DATA_DIR / "logs"        # engine price logs, one folder per replay run
NFLVERSE_DIR = RAW_DIR / "nflverse"
KALSHI_DIR = RAW_DIR / "kalshi"

FIRST_PBP_SEASON = 1999

# Kalshi NFL series, verified 2026-09-26 (docs/DATA.md).
GAME_SERIES = "KXNFLGAME"      # "<team> wins?" - one binary contract per team
SPREAD_SERIES = "KXNFLSPREAD"  # "<team> wins by over N.5 points?" ladder
TOTAL_SERIES = "KXNFLTOTAL"    # "Over N.5 points scored?" ladder
KALSHI_SERIES = (GAME_SERIES, SPREAD_SERIES, TOTAL_SERIES)

# nflverse schedule times (gameday + gametime) are US Eastern.
EASTERN_TZ = "America/New_York"
