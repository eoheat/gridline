# Design

## The question

During an NFL game, does a model that sees each play the moment it is public
update win, spread and total probabilities **faster or better** than Kalshi's
market?

The question is about in-game updating, not pregame handicapping. Model and
market start from the same pregame price, and the benchmark measures what each
does with the plays that follow.

## Pipeline

```
 nflverse pbp ──► play events ──► EventSource ──► GameState ──► model ──► final-score ──► fair price for ──► price log
 (+ timestamps)   (cleaned,       (replay or      (home view)             distribution    every contract     (parquet)
                   timed)          live feed)                                                                    │
                                                                                                                 ▼
 Kalshi candles + trades ───────────────────────────────────────────────────────────────────────────────►  evaluation
```

The engine reads Kalshi's prices once per game: the last quotes before kickoff (both
winner books, and the total ladder when it is liquid), which are its prior
(decision 3). Every later price is joined only afterwards, so the model cannot learn
from the market it is judged against.

## Decisions, and why

### 1. One code path for replay and live
The engine receives events only through an `EventSource`. Replay emits
historical plays at the moment each became public, and a live feed would emit
the same event type. Whatever is backtested is exactly what would run live. The
Kalshi bot's September audit found bugs that existed only in the live path, and
this rules out that class of bug.

### 2. Two models of the same game
- **Direct win probability (Phase 1).** XGBoost on nflfastR's features, with its
  monotonic constraints. It is fast, well studied, and has a published bar:
  under leave-one-season-out validation (2000-2019), a calibration error of 0.0055
  without the betting spread and 0.0066 with it. Reproducing those numbers first
  proves the data pipeline and the evaluation code are right before anything new
  is built. (Done: 0.0054 and 0.0067; see RESULTS.md.)
- **Drive simulator (Phase 2).** This is the "probability of key actions"
  approach. For the drive in progress and every later drive it models:
  - the outcome: TD (with the conversion folded in: TD+0, +1, +2), FG, missed FG,
    punt, turnover, turnover on downs, safety, defensive TD or end of half;
  - the clock the drive uses;
  - where the next drive starts.

  Outcome and clock come from one softmax over (outcome, clock bin) pairs,
  conditioned on down, distance, field position, clock, score, timeouts, era and
  the pregame line. It is a small neural network evaluated in numpy, because the
  simulator calls it once per drive for thousands of simulated games: for 4,000
  games it gives outcome and clock in 19 ms, where boosted trees take 78 ms for
  the outcome alone, and a boosted-tree reference was no more accurate at drive
  starts. Where the next
  drive starts (kicks, returns, onside kicks, changes of possession) and the
  timeouts used are drawn from real drives in the same situation. Monte Carlo
  then plays out the rest of the game, overtime rules included, and returns the
  **joint** distribution of final scores. One run prices the winner, every spread
  rung and every total rung, so the prices are consistent with each other by
  construction.
- The two models estimate the same win probability independently. When they
  disagree, one of them is wrong, which makes a cheap standing bug detector.
- **Why drive-level and not play-level:** a play-level simulator has to model
  play calling, clock management and penalties. That is roughly 10x the work,
  and the errors compound. The drive model conditions on the clock and score,
  so it picks up two-minute-drill and clock-killing behavior from the data. If
  its errors turn out to concentrate late in games, a play-level model for the
  final minutes is the upgrade.

### 3. The pregame line is the prior
At kickoff the simulator has two free settings per game: a **strength gap**
(which offense scores more per drive) and a **scoring rate** (how many points
both teams score). They are solved so that the simulated pregame margin and
total match the prior. Soccer in-play models are anchored the same way, with
supremacy and goal expectancy taken from pregame lines.

As built, the two knobs are the spread and total fed to the drive model, which
learned what a line means for each drive. Training on the line also keeps team
strength from leaking into the score features: a team that leads is usually the
better team, and the model sees both. The solver (`simulator.solve_knobs`) can
also match a winner price instead of a margin, which is what the Kalshi prior
needs.

- Headline prior: **Kalshi's own last mid before kickoff** (winner market), plus
  the ladders when they are liquid. Model and market then start from the same
  place, and the benchmark isolates in-game updating.
  As built (`engine/prior.py`), the prior comes from the last one-minute candles
  that closed at or before kickoff:
  - Win price: the average of the home contract's mid and one minus the away
    contract's.
  - Total: the median total, read where the total ladder's P(over) crosses 50%.
    Only rungs quoted no more than 10¢ wide count; without two such rungs either
    side of 50%, the sportsbook total is used instead.

  The knobs are solved with 10,000 simulated games, aiming for 0.002 on the win
  price and 0.05 points on the median. In 2025, 232 of 285 games got there; the
  rest keep the closest of 12 tries (at most 0.009 and 0.27 points off). Each event
  is then priced with 2,000 simulated games (decision 8), so the published pregame
  price carries the same Monte Carlo noise as every later price, about a point.
- Alternative prior: the sportsbook closing line in nflverse (`spread_line`,
  `total_line`), which is what nflfastR's spread-adjusted model uses. Phase 1
  uses this one for parity.

### 4. Kalshi's contract rules, mirrored exactly
- Winner (`KXNFLGAME`): **a tie settles 50/50**, so fair YES = P(win) + 0.5·P(tie).
  Regular-season games can end tied; postseason games cannot.
- Spread (`KXNFLSPREAD`): "<team> wins by over N.5", so YES = P(margin > N.5).
- Total (`KXNFLTOTAL`): "over N.5 points", so YES = P(total > N.5).
- Every strike is a half point, so no contract can push.

### 5. No lookahead, anywhere
- A play's result is treated as public at `t_known = play end + δ`. The delay δ
  is unknown (broadcast delay plus data feed), so every market result is
  reported as a sweep over δ = 0-20 s rather than a single guessed value.
- The market's state at time t is the last candle that **closed** at or before t
  (a candle is only known at its `end_ts`), or the last trade at or before t.
- Timestamps are cleaned so information can arrive late but never early. For
  example, an out-of-order row is moved later, never earlier (`state/events.py`).
- After play *i*, the engine knows the play's score change and the next
  possession, down, distance and field position. It does **not** know the clock
  runoff before the next snap: until that snap, the clock is play *i*'s clock
  minus the play's own duration. The exact pre-snap state (runoff, any timeout)
  becomes known at the next snap.
- As built (`engine/sources.py`), a play's result counts as public at its end
  timestamp unless one of these later times applies:
  - **No end timestamp** (11% of 2025 plays, nearly every extra point): the snap
    plus the 99th-percentile length of that type of play in the three seasons
    before.
  - **A penalty**: when the ruling is announced, taken as a minute after the play
    or the next snap, whichever comes first.
  - **A replay review or challenge**: the next snap.
  - **The clock ran out after the play**: when it reached zero. Until then nobody
    knows that no further play is coming.
  - **Overtime**: its coin toss counts as known only at the overtime kickoff.
  - **Snap events**: in the 4th quarter, overtime and the last 5 minutes of the
    2nd quarter, each snap is an event too, carrying the exact clock.
- The minutes while a review is pending are left out of the benchmark for both
  sides: the replay keeps the price from before the play there, where a live
  engine would price the call on the field.

### 6. Leakage guards
- nflverse mixes pre-play and post-play columns. For example,
  `total_home_score` already includes the row's points. Events expose only
  `*_pre` and `*_post` scores. Outcomes are renamed `final_*` so that using one
  as a feature looks obviously wrong in review.
- Features may read only a whitelist of columns (`FEATURE_INPUTS` in
  `models/features.py`). A test scrambles every other column, outcomes and post-play
  scores included, and fails if any feature changes.

### 7. Honest statistics
- A season has about 285 games. Snapshots within a game are highly correlated,
  so the effective sample size is closer to the number of games than to the
  number of plays. (The KXBTC work hit the same trap: 538 snapshots came from
  only about 16 independent hours.)
- Every interval resamples whole games, never plays (`eval/bootstrap.py`).
- **Primary outcome test:** per-game paired Brier and log-loss differences
  (model vs market at matched times), with confidence intervals from resampling
  whole games.
- **Information test:** regress the market's move over the next *k* minutes on
  (model − market) at t. A positive, significant slope means the model knows
  something the price does not yet reflect. This uses thousands of weakly
  dependent observations and asks the question that matters.
- **Lead-lag:** an event study around scoring plays and turnovers measuring how
  many seconds after `t_known` the market has covered half of the model's
  price move.
- Splits:
  - Phase 1 parity: leave-one-season-out.
  - Market benchmark: forward only. Train on 2024 and earlier, test on 2025.
    Then train on 2025 and earlier and test each 2026 week as it happens, which
    is truly out-of-sample.

### 8. Monte Carlo noise budget
With 10,000 simulations, a 50% price carries about ±0.5 points of standard
error. That is as large as the effects being measured. So:
- Consecutive states in a game reuse the same random numbers, so a price
  changes because the game changed, not because the dice did.
- N is chosen from a stated noise budget, not by habit.
- The direct win-probability model has no Monte Carlo noise, which is one more
  reason to keep it.
- As built: 2,000 simulated games per state for evaluation (the gate averages over
  tens of thousands of states, so the noise washes out) and 20,000 for a single
  quoted price (about ±0.35 points at 50%).
- In the Phase 3 benchmark the noise is accounted for, not ignored:
  - **Brier score.** The noise adds its variance to the model's Brier score, so
    the corrected score subtracts that variance. The variance is known exactly
    from the simulated price.
  - **Information test.** For the market's move toward the model, noise in
    (model − market) only biases the slope toward zero, which is conservative. For
    the model's move toward the market it can inflate the slope, by up to about 0.05.
  - **Log loss.** It is reported without correction. The noise costs the model
    about 0.0003.

### 9. The game changes over time
The data spans rule changes that matter to the simulator:
- the 2024 kickoff redesign and the 2025 touchback moved to the 35, which shift
  starting field position;
- the 2024 regular-season overtime rule, under which both teams possess;
- a steady rise in fourth-down aggressiveness.

The drive model gets an era feature and recency weights (a season six years
older counts half as much), and later seasons are fed in as the last training
season so the trend is never extrapolated. Kickoffs are drawn from the latest
season only. The forward tests (train through 2023, test on 2024; through 2024,
test on 2025) are the drift checks that matter for the benchmark. A cross-era
check (train on 2015-2019, test on 2023-2025) has not been run yet.

### 10. Read-only, and where things run
- No keys, no order endpoints. Everything uses public market data. If quoting
  is ever wanted, it belongs in the separate Kalshi bot, consuming this
  engine's price log.
- Kalshi pulls must run on the Mac, the only machine with Kalshi access.
  Everything else (nflverse, training, replay, evaluation) runs anywhere.
- Raw Kalshi data is never committed. Published results are derived tables and
  charts.

### 11. Performance targets
- Live: price one state (all contracts) in under 100 ms. Plays are about 40 s
  apart, so this is generous.
- Replay: a full season in minutes. The simulator is vectorized across
  simulations; model calls are batched across all simulations at each drive
  step. If that is too slow, the drive model is compiled into lookup tables:
  fit with ML, serve as a table.
- Measured in Phase 2. In the 2-core cloud container, a state takes about 140 ms
  on one core at 2,000 simulated games, so every 2025 snap takes about 50 minutes
  on two cores; replay is fine. On the Mac's 4-core Linux VM a state takes 36 ms
  at 2,000 simulated games and 315 ms at 20,000. A live price therefore fits
  under 100 ms at a few thousand simulated games, not at 20,000.
- Measured in Phase 3: the 2025 replay priced 62,009 events in 48.5 minutes on
  the container's two cores, 68 ms of one core per event at 2,000 simulated games.

## Data layers

```
data/raw/        exactly what sources returned. Immutable, re-pullable.
  nflverse/pbp/pbp_<season>.parquet, schedules.parquet
  kalshi/<season>/catalog.parquet, games/<game_id>/{markets,candles,trades}.parquet
data/derived/    rebuilt by code from raw/
  events/events_<season>.parquet      (state/events.py)
  kalshi/{markets,candles,trades}_<season>.parquet   one table each (kalshi consolidate)
  benchmark/                          the Phase 3 report's numbers and curves
data/logs/       engine price logs, one folder per replay run (Phase 3)
  replay_<season>/prices/<game_id>.parquet   one row per event: state, prices, prior
```

The **events table** is the contract between data and everything else. It has
one row per pbp row, in play_id order:
- timing: `t_snap`, `t_end` (monotone), plus flags `snap_imputed`, `snap_wild`,
  `snap_out_of_order`, `end_imputed`;
- pre-snap state: clock, possession, down, distance, field position, timeouts,
  `*_score_pre`;
- the play's result: `*_score_post`;
- priors: `pregame_spread`, `pregame_total`;
- labels: `final_margin_home`, `final_total`.

`with_known_times(events, δ)` adds `t_known`. `GameState.pre_snap(row)` turns a
row into the object models consume.

## Phases and exit gates

| Phase | Build | Exit gate |
|---|---|---|
| 0 | nflverse ingest + timing, Kalshi client + catalog + pull, inventory | ≥ 95% of 2025 games timed and quoted (`gridline inventory`) |
| 1 | features + leakage test, direct WP, nflfastR's calibration metric | calibration error within 0.001 of nflfastR's for both models (0.0055 / 0.0066, LOSO 2000-2019); forward test on 2025 level with nflverse's published model. **Passed.** |
| 2 | drive model, simulator, OT, prior-matching knobs, pricing of all three families | on 2025, with models trained through 2024 and fixed before the run: the final margin and total land inside the simulator's central 50% and 90% intervals within 3 points of 50% and 90% (randomized PIT); the simulator's CRPS beats a normal baseline for both; simulated win probability is within 0.03 of the Phase 1 model on average and its log loss at most 0.01 worse **Passed.** |
| 3 | replay engine, price log, market benchmark | a report of where the model leads or lags the market, by how much, with CIs. **Done** (RESULTS.md): level with the market on the winner and total contracts, slightly behind on the spread ladder; the market makes half of a big play's move within about 5 s of the NFL's timestamp, so the model leads only with a faster feed |
| 4 | integrity monitor, dashboard, write-up | precision/recall on injected anomalies; the repo runs end to end |

## Known approximations and open questions

- Timeout rows are placed at or after the previous row, never at their earlier
  true time. This is conservative, and the next snap carries the right timeout
  count anyway.
- Spread and total ladders are compared only where a rung's book is two-sided and
  at most 5¢ wide: 356,394 spread and 242,191 total rung-minutes in 2025, on 4,240
  and 3,500 contracts.
- Kalshi prices may carry a fee-driven favorite-longshot bias, so a mid is not a
  pure probability. It is measured by calibrating mids against outcomes, not
  assumed away. In 2025 it did not show: long shots were, if anything, underpriced.
- Between a play and the next snap, the state counts the clock as stopped at the
  play's end. The next snap corrects it, and late in games the snap is an event of
  its own, but the price in between overstates the time left when the clock runs.
- The knobs are fixed at kickoff: gridline does not revise the teams' strength or
  scoring rate as a game goes on. The market does.
- Live mode needs a permitted real-time feed. Replay is the core deliverable,
  and the missing live feed is exactly the moat official-data companies own.
