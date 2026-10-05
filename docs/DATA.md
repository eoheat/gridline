# Data notes

Everything below was checked against the live sources on 2026-09-26. Unverified
points are marked as such.

## nflverse play-by-play

- **Access:** `nflreadpy.load_pbp(season)`. Seasons 1999-2026 are available.
  The current season updates weekly (33 games of 2026 as of week 3's Thursday
  game). About 48k rows and 372 columns per season; 308 MB of parquet for all
  seasons.
- **Licence:** CC-BY 4.0. Attribute nflverse when publishing. FTN charting is
  CC-BY-SA and is not used.

### Pre-play vs post-play columns (the leakage trap)

| column | meaning |
|---|---|
| `total_home_score`, `total_away_score` | **post**-play: they include the row's own points (a TD row shows 6) |
| `posteam_score`, `defteam_score`, `score_differential` | pre-play |
| `posteam_score_post`, `score_differential_post` | post-play |
| `home_timeouts_remaining`, `away_timeouts_remaining` | at the snap (on a timeout row: after that timeout) |
| `game_seconds_remaining` | at the snap; in overtime it counts the OT clock (600 → 0) |
| `result`, `total` | final margin (home − away) and final total: **labels only** |

Events expose these as `home_score_pre`, `home_score_post`, `final_margin_home`
and `final_total`.

**Score repair.** Timeout rows are logged after plays that happened after them,
and those rows carry the score as of their true, earlier time. A plain lag of
`total_home_score` therefore briefly "un-scores" a touchdown, which happens on
about 250 plays a season in 2022-25. Scores never decrease, so post scores are
repaired with a running max within each game. Checks:
- **2025:** the repaired pre-play scores match nflverse's own `posteam_score` /
  `defteam_score` on every play row (tested), and the final score equals the
  official result in every game (tested).
- **1999-2026:** they match on 99.96% of play rows. The final score differs from
  the official result in 16 of 7,306 games, all from 2001-2002.

### Wall-clock timestamps

| column | meaning | format by era |
|---|---|---|
| `time_of_day` | snap time, UTC | 2003-2004: bare `HH:MM:SS`, unusable (no date or zone); 2005+: ISO `2025-09-07T17:02:38.787Z`, milliseconds on about 85% of rows since 2022 |
| `end_clock_time` | end of the play, UTC | only from 2022 (ISO). Before that it is absent or the *game* clock (`MM:SS`) |

Coverage over real plays (rows with a `play_type`), from `timing_summary`:

| seasons | snap from raw time | end from raw time |
|---|---|---|
| 1999-2002 | 0% | 0% |
| 2003-2021 | 96-100% | ~0% |
| 2022-2026 | 99.8-100% | 77-87% |

Play length, snap to end (2025-26, rows with both raw times):

| play type | median | 95th percentile |
|---|---|---|
| pass | 5.1 s | 8.4 s |
| run | 4.4 s | 7.4 s |
| kickoff | 8.4 s | 11.3 s |
| punt | 8.6 s | 13.3 s |
| field goal | 3.7 s | 5.1 s |

Snap to snap: median 43 s, 90th percentile 177 s. A game takes a median of
182 minutes of wall-clock time.

Quirks, and how `state/events.py` handles them:
- About 4.5% of play rows (2022+) have a snap earlier than an earlier row's
  snap: timeouts logged out of `play_id` order. Their timestamp is right and
  their position is wrong. They are placed no earlier than the previous row,
  which is conservative.
- A handful of snaps per season are hours off. They are detected against the
  median of neighbouring snaps (45 min threshold) and imputed.
- Administrative rows (game start, quarter end, two-minute warning) have no
  time. They get the previous row's end.

### Drives (Phase 2)

- `fixed_drive` numbers drives and `fixed_drive_result` names the outcome: Touchdown,
  Field goal, Missed field goal, Punt, Turnover, Turnover on downs, Safety, Opp
  touchdown (a defensive or return touchdown during the drive) and End of half.
- A kickoff belongs to the **receiving** team's drive, and the extra point to the
  scoring drive. A kickoff returned for a touchdown is a drive of its own with no
  scrimmage snap; `models/drives.py` books its points to the gap between the drives
  around it.
- The labels are checked by rebuilding every final score from them alone (points
  implied by each drive's outcome, plus the points in each gap). Since 2006 that works
  exactly for 95-99% of games per season. The rest have a drive whose points don't fit
  its outcome, 2-15 drives per season: blocked extra points returned for two, a drive
  booked across a change of possession. Those drives are left out of training.
- Eras show clearly in the outcomes. Punts fell from 41% of drives (2006) to 34%
  (2025) and turnovers on downs rose from 3.4% to 6.3%. Failed extra points (touchdown
  plus 0) jumped in 2015, when the kick moved back. Kickoffs were reshaped repeatedly:
  touchbacks came out at the 20 (2011-15), the 25 (2016-23), the 30 (2024) and the 35
  (2025, when most kicks started being returned).
- Timeouts matter late: with 2-5 minutes left in a half, the defence uses 0.64 of a
  timeout per drive on average and the offence 0.30.

### Schedules

- `nflreadpy.load_schedules()` covers 1999-2026.
- `gameday` and `gametime` are US Eastern.
- `spread_line` is positive when the **home team is favoured**. `total_line` is
  the closing total. Moneylines are also present.
- Neutral-site games (international, Super Bowl) keep a designated home team.
  2025 had 8 of them.
- 2025: 272 regular-season games + 13 postseason. 2026: 272 scheduled.

## Kalshi

- **Access:** public v2 REST at `https://api.elections.kalshi.com/trade-api/v2`,
  no authentication needed for market data. The cloud sandbox cannot reach it;
  the Mac can.

### Series and tickers

| series | contract | ticker example |
|---|---|---|
| `KXNFLGAME` | "<team> wins?", one per team | `KXNFLGAME-25SEP07NYGWAS-WAS` |
| `KXNFLSPREAD` | "<team> wins by over N.5 points?" | `KXNFLSPREAD-25SEP07NYGWAS-WAS3` (floor_strike 3.5) |
| `KXNFLTOTAL` | "Over N.5 points scored?" | `KXNFLTOTAL-25SEP07NYGWAS-45` (floor_strike 45.5) |

- The event suffix is `<YYMONDD><AWAY><HOME>`, with an Eastern date. The TNF
  game of 2026-09-24 (8:15 pm ET) is `...-26SEP24ATLGB`.
- Team codes match nflverse except **Jacksonville = `JAC`** (nflverse `JAX`) and,
  from 2026, **the Rams = `LAR`** (`LA` in 2025 tickers and in nflverse).
- `KXNFLGAME` also lists preseason games (e.g. `26AUG28...`). They have no nflverse
  schedule row, so the catalog ignores them.
- **Some spread tickers spell out the team, and they are dead duplicates.**
  `KXNFLSPREAD-25SEP28NOBUF-BUFFALO22` and `KXNFLSPREAD-25NOV16DETPHI-PHILADELPHIA17`
  sit at the same strikes as the real rungs `…-BUF22` and `…-PHI17`. Each opened a
  few seconds before its twin and never traded: 0 contracts, against 45,981 and
  59,524. The catalog reads the spelled-out name, or failing that the market's title.
  Anything that builds a ladder must keep one contract per side and strike: the one
  that traded.
- **Conference championships and the Super Bowl trade as "champion" markets**,
  one per team:
  - `KXNFLAFCCHAMP-25` / `KXNFLNFCCHAMP-25` for the 2025 season's title games, named
    for the season year;
  - `KXSB-26` for the Super Bowl played in February 2026, named for the game's
    calendar year.

  The NFC title game also had a `KXNFLGAME` event, but with zero volume. The AFC
  title game and the Super Bowl have none. Their spread and total ladders still use
  the usual suffix.
- Other NFL series exist and could be priced by the simulator later:
  `KXNFL2HWINNER`, `KXNFL3QWINNER`, `KXNFLBOTH` (both teams score) and player
  props.

### Rules that matter to the pricer

- Winner markets: "If cancelled and not played, rescheduled, or **declared a
  tie**, the market resolves to 50/50 for all teams." A postponed game stays
  open until it is played.
- All ladder strikes are N.5, so nothing pushes.
- **Ladders are wide and thin.** NYG @ WAS 2025 had 51 total rungs (31.5-80.5).
  The busiest, 45.5, traded 16,840 contracts, against 2.75M on the WAS winner
  contract; 34 rungs never traded. So the winner market is the main benchmark,
  and ladders are used only where they are quoted (the inventory measures this).

- **A tie settles as `result = "scalar"` at 0.50 on both sides.** Real example:
  GB @ DAL, 40–40, 2025 week 4.
- **`close_time` is not reliable.** On that tied game, Kalshi's `close_time` reads
  8.3 hours *before* kickoff, yet 1-minute candles exist for the whole game. The
  pull therefore uses a fixed window, kickoff − 3 h to kickoff + 8 h, and never
  cuts at `close_time`.
- **Books go one-sided at the extremes.** In a decided game there is often a 99¢
  bid with no ask, or a 1¢ ask with no bid. In 2025, 9% of in-game minutes looked
  like this (2,946 with no ask, 1,760 with no bid, of 51,615), and none were
  empty. These minutes mean "nearly certain", not missing data. The inventory
  counts either side as quoted and reports two-sided coverage separately
  (median 96% of in-game minutes).
- **Per game** (2025, winner markets): median 28,600 trades and 25 traded
  ladder rungs.

### One market's lifecycle (NYG @ WAS, 2025-09-07)

| | |
|---|---|
| opened | 2025-05-20 |
| last play ended | 20:13:39Z |
| closed | 20:15:39Z |
| settled | 20:21:57Z |
| volume | WAS 2.75M contracts, NYG 2.55M |
| mid at kickoff | WAS 0.715 (WAS won) |

### Two storage tiers

`GET /historical/cutoff` returned `market_settled_ts = trades_created_ts =
2026-07-28T00:00:00Z`. **The whole 2025 season is in the historical tier; 2026
is in the live tier.**

| data | live tier | historical tier |
|---|---|---|
| markets | `/markets?event_ticker=` | `/historical/markets?event_ticker=` |
| candles | `/series/{s}/markets/{t}/candlesticks` | `/historical/markets/{t}/candlesticks` |
| trades | `/markets/trades?ticker=` | `/historical/trades?ticker=` |
| candle price fields | `yes_bid.close_dollars` | `yes_bid.close` |
| volume / OI fields | `volume_fp`, `open_interest_fp` | `volume`, `open_interest` |

- Asking the live tier for a historical event returns `200` with an empty
  `markets` list, not an error. The client then retries on the historical tier.
- Trades come back newest first, paginated by `cursor`.
- Candles: `period_interval=1` gives 1-minute bars with bid, ask and last-trade
  OHLC. A bar covers `(end_period_ts − 60, end_period_ts]` and is only
  **known at its end**. An empty side reads bid `0.0000` / ask `1.0000`; that
  means no quote, not a price.
- Trades: `created_time` has microsecond resolution (trailing zeros are
  trimmed: `.20097Z`). Also `yes_price_dollars`, `count_fp` (fractional
  contracts exist), `taker_side` and `is_block_trade`.
- Events list newest first. The first page reached back to Nov 2025, and week 1
  of 2025 (Sep 7) has markets and 1-minute candles.

### The 2025 markets in play (Phase 3)

Measured from the consolidated tables (`gridline kalshi consolidate`: 10,994 markets,
2.15 million one-minute candles, 12.7 million trades).

- **The winner books are tight.** At kickoff both were quoted in all 285 games, 1¢
  wide at the median and 4¢ at most. The home and away mids summed to 0.99–1.02. In
  play, the median width is 1¢ and the 90th percentile 2¢.
- **The total ladder is usable at kickoff in most games.** In 266 of 285, two rungs no
  wider than 10¢ straddle 50%. The median total read off them agrees with the
  sportsbook total on average (mean gap −0.01 points, SD 0.55): two thirds of games are
  within half a point, and the largest gap is 1.7.
- **Two games have no ladders.** The pull found winner markets but no spread or total
  contracts for the two San Francisco–Seattle games (weeks 18 and 20).
- **In play, the ladders are thin.** A rung counts as liquid in a minute when its book
  is two-sided and at most 5¢ wide. 4,240 of the 5,820 spread contracts were liquid in
  at least one in-game minute, and 3,500 of the 4,604 total contracts.
- **Trades are dense enough for seconds.** On the 3,598 plays the lead-lag study uses,
  one of the two winner contracts traded within a second after the play's end in 87% of
  cases (median gap 0.2 s).

### Not yet verified (the Kalshi pull will show)

- Whether every 2025 game, including the postseason, has all three series.
- Rate limits. The client defaults to 10 requests/s with backoff on 429.
- Whether any 2024-season games (Jan-Feb 2025 playoffs) are listed.
