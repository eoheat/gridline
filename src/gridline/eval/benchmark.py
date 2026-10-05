"""Phase 3 benchmark: where the model leads or lags Kalshi, by how much, with intervals.

The model's price at time t is its last quote known by t: the latest event with
t_event + delta <= t (DESIGN.md decision 5). delta, the time between a play ending and
its result reaching the engine, is unknown, so everything is run for DELTAS. The
market's price at t is the close of the candle ending at t (or, for the event study, the
last trade at or before t). Every interval resamples whole games (eval/bootstrap.py).

1. Accuracy (winner). On a one-minute grid from kickoff to the final whistle: Brier score
   and log loss of model and market against how the contract settled, and their paired
   difference. The model's prices carry Monte Carlo noise (2,000 simulated games), which
   adds its variance to the Brier score; the corrected score subtracts it.
2. Accuracy (ladders). The same on spread and total rungs, where and when the rung's
   book is two-sided and at most 5 cents wide.
3. Information. Regress the market's move over the next k minutes on (model - market)
   now, controlling for the market's own last-minute move (quotes that bounce revert,
   which alone would give a positive slope). A positive slope means the market moves
   toward the model: the model knew something the price did not yet show. The reverse
   regression asks the same of the model.
4. Lead-lag. Around every play that moved the model's price by at least MOVE, the
   market's trade price relative to its price at the snap, as a share of the model's
   move, second by second. From it: when the market has made half the move, how much
   of the move is left at t_end + delta, and what trading the model's direction at that
   moment would have made at settlement, before and after Kalshi's taker fee (not the
   bid-ask spread: the price is the last trade's).
5. Calibration of market mids and model prices (nflfastR's calibration error, Phase 1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from gridline import config
from gridline.engine import engine as eng
from gridline.eval import bootstrap, market, metrics

Log = Callable[[str], None]

DELTAS = (0, 5, 10, 20)  # seconds from a play's end until the engine knows its result
K_MINUTES = (1, 2, 5)
MOVE = 0.05  # event study: plays that moved the model's home price by at least this
TAUS = np.arange(-60, 181)  # event study: seconds around the end of the play
CLIP = 0.01  # prices are clipped to [0.01, 0.99] for log loss (Kalshi's tick is 1 cent)
MAX_SPREAD = 0.05  # quotes wider than this are left out
TAKER_FEE = 0.07  # Kalshi's taker fee per contract: 0.07 * P * (1 - P) (fee schedule, NFL multiplier 1)


def out_dir() -> Path:
    return config.DERIVED_DIR / "benchmark"


def load_prices(run: str) -> pl.DataFrame:
    from gridline.engine import sinks

    return sinks.load(run).with_row_index("q")


def spans(prices: pl.DataFrame) -> pl.DataFrame:
    """Per game: kickoff (the pregame event), the final whistle, and how the home
    winner contract settled (1, 0, or 0.5 for a tie)."""
    f = prices.filter(pl.col("kind") == "final").select(
        "game_id", t_final="t", y=pl.when(pl.col("home_score") > pl.col("away_score")).then(1.0)
        .when(pl.col("home_score") < pl.col("away_score")).then(0.0).otherwise(0.5),
        final_margin=pl.col("home_score") - pl.col("away_score"),
        final_total=pl.col("home_score") + pl.col("away_score"))
    k = prices.filter(pl.col("kind") == "pregame").select("game_id", kickoff="t")
    return k.join(f, on="game_id")


def model_asof(prices: pl.DataFrame, times: pl.DataFrame, delta: float) -> pl.DataFrame:
    """times (game_id, t, ...) with the model's last quote known by t: q (its row in
    prices), p_home, p_tie, and the event's kind (event_kind), quarter and clock."""
    # the pregame price is known before kickoff (it is Kalshi's own); every later event
    # reaches the engine `delta` seconds after it became public
    lag = pl.when(pl.col("event_kind") == "pregame").then(pl.duration(microseconds=0)).otherwise(
        pl.duration(microseconds=int(round(delta * 1e6))))
    ev = (prices.select("q", "game_id", "t", "order", "p_home", "p_tie", "qtr", "game_seconds_remaining",
                        "n_sims", event_kind="kind")
          .with_columns(t_known=pl.col("t") + lag)
          .drop("t")
          .sort("game_id", "order")
          .with_columns(t_known=pl.col("t_known").cum_max().over("game_id"))
          .unique(subset=["game_id", "t_known"], keep="last", maintain_order=True)
          .sort("t_known"))
    out = (times.sort("t").join_asof(ev, left_on="t", right_on="t_known", by="game_id",
                                     strategy="backward", check_sortedness=False)
           .sort("game_id", "t"))
    return out


def outside_reviews(prices: pl.DataFrame, times: pl.DataFrame, delta: float) -> pl.DataFrame:
    """times without the moments when a replay review was pending: from the reviewed
    play's end until the engine learns the ruling (its event time plus delta). The replay
    holds the price from before the play there, where a live engine would have priced the
    call on the field, so neither model nor market is scored."""
    w = (prices.filter(pl.col("timing") == "review")
         .select("game_id", start="t_play_end",
                 stop=pl.col("t") + pl.duration(microseconds=int(round(delta * 1e6))))
         .sort("start"))
    if w.height == 0:
        return times
    j = times.sort("t").join_asof(w, left_on="t", right_on="start", by="game_id", strategy="backward",
                                  check_sortedness=False)
    return j.filter(pl.col("stop").is_null() | (pl.col("t") >= pl.col("stop"))).drop("start", "stop").sort("game_id", "t")


def mc_var(p: np.ndarray, p_tie: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Monte Carlo variance of a simulated win price (ties count half) from n simulated
    games: each pays 1, 0.5 or 0, with variance p(1 - p) - p_tie / 4. Zero where n is 0
    (the final whistle, which is not simulated)."""
    v = np.clip(p * (1 - p) - 0.25 * p_tie, 0, None)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


# ------------------------------------------------------------------ accuracy


def _in_game(prices: pl.DataFrame, minutes: pl.DataFrame) -> pl.DataFrame:
    """Candle closes from kickoff to the final whistle with a quote on the winner books."""
    return (minutes.join(spans(prices), on="game_id")
            .filter((pl.col("t") >= pl.col("kickoff")) & (pl.col("t") < pl.col("t_final"))
                    & pl.col("market").is_not_null()))


def winner_grid(prices: pl.DataFrame, minutes: pl.DataFrame, delta: float) -> pl.DataFrame:
    """One row per game and in-game minute where the market is quoted: market, model,
    y and the model's Monte Carlo variance."""
    g = _in_game(prices, minutes).filter(pl.col("spread") <= MAX_SPREAD + 1e-9)
    g = model_asof(prices, outside_reviews(prices, g, delta), delta).rename({"p_home": "model"})
    assert g["model"].null_count() == 0, "grid minutes with no model price"
    return g.with_columns(var_mc=pl.Series(mc_var(g["model"].to_numpy(), g["p_tie"].to_numpy(),
                                                  g["n_sims"].to_numpy().astype(float))))


def _log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP, 1 - CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(model: np.ndarray, mkt: np.ndarray, y: np.ndarray, games: np.ndarray,
           var_mc: np.ndarray | None = None) -> dict:
    """Brier and log loss of model and market on the same rows, and model minus market
    with a 90% interval from resampling games. brier_model_corrected subtracts the
    Monte Carlo variance from the model's squared errors."""
    b_mod, b_mkt = (model - y) ** 2, (mkt - y) ** 2
    l_mod, l_mkt = _log_loss(model, y), _log_loss(mkt, y)
    b_cor = b_mod - (var_mc if var_mc is not None else 0.0)
    out = {"rows": int(len(y)), "games": int(len(np.unique(games))),
           "brier_model": float(b_mod.mean()), "brier_model_corrected": float(b_cor.mean()),
           "brier_market": float(b_mkt.mean()), "log_loss_model": float(l_mod.mean()),
           "log_loss_market": float(l_mkt.mean())}
    diffs = {"brier": b_mod - b_mkt, "brier_corrected": b_cor - b_mkt, "log_loss": l_mod - l_mkt}
    draws = {k: [] for k in diffs}
    for w in bootstrap.game_weights(games):
        for k, d in diffs.items():
            draws[k].append(np.average(d, weights=w))
    for k, d in diffs.items():
        out[f"{k}_diff"] = float(d.mean())
        out[f"{k}_diff_p05"], out[f"{k}_diff_p95"] = (float(x) for x in np.quantile(draws[k], [0.05, 0.95]))
    return out


def stage(qtr: pl.Expr, gsr: pl.Expr) -> pl.Expr:
    return (pl.when(qtr >= 5).then(pl.lit("overtime")).when(gsr <= 300).then(pl.lit("last 5 min"))
            .otherwise(pl.format("Q{}", qtr)))


def winner_accuracy(g: pl.DataFrame) -> dict:
    res = paired(g["model"].to_numpy(), g["market"].to_numpy(), g["y"].to_numpy(),
                 g["game_id"].to_numpy(), g["var_mc"].to_numpy())
    res["by_stage"] = {}
    for (st,), part in g.with_columns(stage=stage(pl.col("qtr"), pl.col("game_seconds_remaining"))).group_by("stage"):
        res["by_stage"][st] = paired(part["model"].to_numpy(), part["market"].to_numpy(), part["y"].to_numpy(),
                                     part["game_id"].to_numpy(), part["var_mc"].to_numpy())
    return res


def direct_accuracy(g: pl.DataFrame, prices: pl.DataFrame) -> dict:
    """The Phase 1 model against the market, on the minutes where its last quote is a
    state it can price (a scrimmage snap comes next, in regulation), with the simulator
    on the same minutes for reference. The Phase 1 model is anchored to the sportsbook
    spread, not to Kalshi's price at kickoff."""
    d = g.join(prices.select("q", "wp_direct_home"), on="q").filter(pl.col("wp_direct_home").is_not_null())
    games = d["game_id"].to_numpy()
    y, mkt = d["y"].to_numpy(), d["market"].to_numpy()
    out = {"direct": paired(d["wp_direct_home"].to_numpy(), mkt, y, games),
           "simulator": paired(d["model"].to_numpy(), mkt, y, games, d["var_mc"].to_numpy())}
    late = d.filter(pl.col("game_seconds_remaining") <= 300)
    if late.height:
        lg = late["game_id"].to_numpy()
        out["direct_last5"] = paired(late["wp_direct_home"].to_numpy(), late["market"].to_numpy(), late["y"].to_numpy(), lg)
        out["simulator_last5"] = paired(late["model"].to_numpy(), late["market"].to_numpy(), late["y"].to_numpy(), lg,
                                        late["var_mc"].to_numpy())
    return out


def rung_grid(prices: pl.DataFrame, rungs: pl.DataFrame, delta: float) -> pl.DataFrame:
    """Liquid spread and total rungs at every in-game minute, with the model's price
    read off its distribution of the final margin or total."""
    sp = spans(prices).select("game_id", "kickoff", "t_final")
    r = (rungs.join(sp, on="game_id")
         .filter((pl.col("t") >= pl.col("kickoff")) & (pl.col("t") < pl.col("t_final")) & pl.col("y").is_not_null()))
    r = model_asof(prices, outside_reviews(prices, r, delta), delta)
    assert r["q"].null_count() == 0, "rung minutes with no model price"
    q = r["q"].to_numpy()
    k = np.floor(r["strike"].to_numpy()).astype(np.int64)
    kind, side = r["kind"].to_numpy(), r["side"].to_numpy()
    cm = np.stack(prices["cdf_margin"].to_numpy())
    ct = np.stack(prices["cdf_total"].to_numpy())

    def at(cdf, grid, rows, z):
        return np.where(z < grid[0], 0.0, np.where(z > grid[-1], 1.0, cdf[rows, np.clip(z - grid[0], 0, grid.size - 1)]))

    home = 1.0 - at(cm, eng.MARGIN_GRID, q, k)
    away = at(cm, eng.MARGIN_GRID, q, -k - 1)
    over = 1.0 - at(ct, eng.TOTAL_GRID, q, k)
    model = np.where(kind == "total", over, np.where(side == "home", home, away))
    n = r["n_sims"].to_numpy().astype(float)
    var = np.divide(model * (1 - model), n, out=np.zeros_like(model), where=n > 0)
    return r.with_columns(model=pl.Series(model), var_mc=pl.Series(var))


def rung_accuracy(r: pl.DataFrame) -> dict:
    out = {}
    for (kind,), part in r.group_by("kind"):
        out[kind] = paired(part["model"].to_numpy(), part["market"].to_numpy(), part["y"].to_numpy(),
                           part["game_id"].to_numpy(), part["var_mc"].to_numpy())
        out[kind]["contracts"] = int(part["ticker"].n_unique())
    return out


# ------------------------------------------------------------------ information


def _wls(y: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    sw = np.sqrt(w) if w is not None else np.ones(len(y))
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def info_test(g: pl.DataFrame, k: int) -> dict:
    """Does the market move toward the model over the next k minutes (and the model
    toward the market)? g is winner_grid's output."""
    g = g.sort("game_id", "t")
    mk = pl.duration(minutes=k)
    fut = g.select("game_id", "t", m_fut="market", p_fut="model").with_columns(t=pl.col("t") - mk)
    past = g.select("game_id", "t", m_past="market", p_past="model").with_columns(
        t=pl.col("t") + pl.duration(minutes=1))
    d = g.join(fut, on=["game_id", "t"]).join(past, on=["game_id", "t"])
    d = d.filter(pl.col("t") + mk < pl.col("t_final"))
    games = d["game_id"].to_numpy()
    m, p = d["market"].to_numpy(), d["model"].to_numpy()
    one = np.ones(len(m))
    res = {"rows": int(len(m)), "k_minutes": k}
    for name, y, x, c in (("market_toward_model", d["m_fut"].to_numpy() - m, p - m, d["m_past"].to_numpy() - m),
                          ("model_toward_market", d["p_fut"].to_numpy() - p, m - p, d["p_past"].to_numpy() - p)):
        X = np.column_stack([one, x, c])
        beta = _wls(y, X)
        draws = np.array([_wls(y, X, w)[1] for w in bootstrap.game_weights(games, n_boot=500)])
        res[name] = {"slope": float(beta[1]), "p05": float(np.quantile(draws, 0.05)),
                     "p95": float(np.quantile(draws, 0.95)), "reversal": float(beta[2])}
    res["gap_sd"] = float(np.std(p - m))
    return res


# ------------------------------------------------------------------ lead-lag


def plays_that_moved(prices: pl.DataFrame, events: pl.DataFrame, move: float = MOVE) -> pl.DataFrame:
    """Play (and final-whistle) events that moved the model's home price by at least
    `move`, timed by the play's own end timestamp from the feed (not imputed, and not
    waiting on a ruling or the clock), with the play's snap time."""
    p = (prices.sort("game_id", "order")
         .with_columns(prev=pl.col("p_home").shift(1).over("game_id"))
         .with_columns(dp=pl.col("p_home") - pl.col("prev"))
         .filter(pl.col("kind").is_in(["play", "final"]) & (pl.col("timing") == "end") & (pl.col("dp").abs() >= move)))
    snaps = events.select("game_id", "seq", "t_snap", "play_type", "desc")
    return p.join(snaps, on=["game_id", "seq"], how="inner").filter(pl.col("t_snap").is_not_null())


def market_paths(moves: pl.DataFrame, trades: pl.DataFrame, taus: np.ndarray = TAUS) -> dict[str, np.ndarray]:
    """For each move: the last trade price at the snap (base) and at t_end + tau."""
    base = np.full(moves.height, np.nan)
    path = np.full((moves.height, taus.size), np.nan)
    tr_by_game = {gid: (g["ts"].dt.epoch("us").to_numpy(), g["price"].to_numpy())
                  for (gid,), g in trades.filter(pl.col("game_id").is_in(moves["game_id"].unique().implode()))
                  .group_by("game_id")}
    t_end = moves["t"].dt.epoch("us").to_numpy()
    t_snap = moves["t_snap"].dt.epoch("us").to_numpy()
    for i, gid in enumerate(moves["game_id"].to_list()):
        if gid not in tr_by_game:
            continue
        ts, px = tr_by_game[gid]
        j = np.searchsorted(ts, t_snap[i], side="right") - 1
        if j >= 0:
            base[i] = px[j]
        jj = np.searchsorted(ts, t_end[i] + taus * 1_000_000, side="right") - 1
        ok = jj >= 0
        path[i, ok] = px[jj[ok]]
    return {"base": base, "path": path}


def _traded_within(moves: pl.DataFrame, trades: pl.DataFrame, seconds: float) -> float:
    """Share of the moves after whose end either winner contract traded within `seconds`."""
    by = {gid: g["ts"].dt.epoch("us").to_numpy() for (gid,), g in
          trades.filter(pl.col("game_id").is_in(moves["game_id"].unique().implode())).group_by("game_id")}
    hits = []
    for gid, t in zip(moves["game_id"].to_list(), moves["t"].dt.epoch("us").to_list()):
        ts = by.get(gid)
        if ts is None:
            continue
        j = np.searchsorted(ts, t)
        hits.append(j < ts.size and ts[j] - t <= seconds * 1e6)
    return float(np.mean(hits)) if hits else float("nan")


def event_study(prices: pl.DataFrame, events: pl.DataFrame, trades: pl.DataFrame, spans_: pl.DataFrame,
                deltas=DELTAS) -> tuple[dict, pl.DataFrame]:
    moves = plays_that_moved(prices, events).join(spans_.select("game_id", "y"), on="game_id")
    mp = market_paths(moves, trades)
    traded = _traded_within(moves, trades, 1.0)
    dp = moves["dp"].to_numpy()
    share = (mp["path"] - mp["base"][:, None]) / dp[:, None]  # NaN before a game's first trade
    ok = np.isfinite(mp["base"])
    moves, share, dp, path = moves.filter(pl.Series(ok)), share[ok], dp[ok], mp["path"][ok]
    games = moves["game_id"].to_numpy()
    y = moves["y"].to_numpy()

    def wmean(x, w=None):  # column means over events, skipping NaN
        w = np.ones(x.shape[0]) if w is None else w
        f = np.isfinite(x)
        return (np.where(f, x, 0) * w[:, None]).sum(0) / np.maximum((f * w[:, None]).sum(0), 1e-12)

    mean_share = wmean(share)
    draws = np.array([wmean(share, w) for w in bootstrap.game_weights(games, n_boot=500)])
    lo, hi = np.quantile(draws, 0.05, axis=0), np.quantile(draws, 0.95, axis=0)

    def first_cross(curve, level):
        idx = np.flatnonzero(curve >= level)
        return float(TAUS[idx[0]]) if idx.size else np.nan

    half = np.array([first_cross(c, 0.5) for c in draws])
    res = {"moves": int(len(dp)), "games": int(len(np.unique(games))),
           "final_whistles": int((moves["kind"] == "final").sum()),
           "traded_within_1s": traded,
           "median_abs_move": float(np.median(np.abs(dp))),
           "half_move_s": first_cross(mean_share, 0.5),
           "half_move_s_p05": float(np.nanquantile(half, 0.05)), "half_move_s_p95": float(np.nanquantile(half, 0.95)),
           "share_at_end": float(mean_share[TAUS == 0][0]),
           "share_at_120s": float(mean_share[TAUS == 120][0])}
    by_delta = {}
    sign = np.sign(dp)
    for d in (*deltas, 30, 60):
        col = int(np.flatnonzero(TAUS == d)[0])
        left = (1 - share[:, col]) * np.abs(dp)  # the model's move not yet in the price
        pnl = sign * (y - path[:, col])  # trade the model's direction at the market, hold to settlement
        px = path[:, col]
        net = pnl - TAKER_FEE * px * (1 - px)  # after Kalshi's taker fee
        bl, bp, bn = [], [], []
        for w in bootstrap.game_weights(games, n_boot=500):
            bl.append(np.average(left, weights=w))
            bp.append(np.average(pnl, weights=w))
            bn.append(np.average(net, weights=w))
        by_delta[str(d)] = {"share": float(np.nanmean(share[:, col])), "left_cents": float(left.mean() * 100),
                            "left_p05": float(np.quantile(bl, 0.05) * 100), "left_p95": float(np.quantile(bl, 0.95) * 100),
                            "pnl_cents": float(pnl.mean() * 100), "pnl_p05": float(np.quantile(bp, 0.05) * 100),
                            "pnl_p95": float(np.quantile(bp, 0.95) * 100),
                            "net_cents": float(net.mean() * 100), "net_p05": float(np.quantile(bn, 0.05) * 100),
                            "net_p95": float(np.quantile(bn, 0.95) * 100)}
    res["by_delta"] = by_delta
    curve = pl.DataFrame({"tau": TAUS, "share": mean_share, "p05": lo, "p95": hi})
    return res, curve


# ------------------------------------------------------------------ calibration


def long_shots(g: pl.DataFrame, below: float = 0.10) -> dict:
    """Minutes when the market priced either team under `below`, from that team's side:
    its average price, how often it won, gridline's average price, and how many games had
    such a team and how many of those it went on to win. Ties are left out."""
    g = g.filter(pl.col("y") != 0.5)
    home = g.filter(pl.col("market") < below).select("game_id", price="market", won="y", model="model")
    away = g.filter(pl.col("market") > 1 - below).select(
        "game_id", price=1 - pl.col("market"), won=1 - pl.col("y"), model=1 - pl.col("model"))
    d = pl.concat([home, away])
    per_game = d.group_by("game_id").agg(pl.col("won").max())
    return {"minutes": d.height, "price": float(d["price"].mean()), "won": float(d["won"].mean()),
            "model": float(d["model"].mean()), "games": per_game.height, "games_won": int(per_game["won"].sum())}


def calibration(g: pl.DataFrame) -> dict:
    """nflfastR's calibration error (Phase 1) of market and model on the grid, ties left
    out; quarters as in Phase 1 (overtime counted with the 4th)."""
    g = g.filter(pl.col("y") != 0.5)
    q = np.minimum(g["qtr"].to_numpy(), 4)
    y = g["y"].to_numpy()
    out = {}
    for who in ("market", "model"):
        p = g[who].to_numpy()
        out[who] = metrics.calibration_error(p, y, q)
    t = metrics.calibration_table(g["market"].to_numpy(), y, np.zeros(len(y)))
    out["market_table"] = t.select("bin", "n_plays", "actual").to_dicts()
    tm = metrics.calibration_table(g["model"].to_numpy(), y, np.zeros(len(y)))
    out["model_table"] = tm.select("bin", "n_plays", "actual").to_dicts()
    d = bootstrap.spread(lambda w: metrics.calibration_error(g["market"].to_numpy(), y, q, w)
                         - metrics.calibration_error(g["model"].to_numpy(), y, q, w), g["game_id"].to_numpy(), n_boot=300)
    out["market_minus_model_p05"], out["market_minus_model_p95"] = d["p05"], d["p95"]
    return out


# ------------------------------------------------------------------ the whole report


def run(season: int, run_name: str, log: Log = print) -> dict:
    from gridline.inventory import load_events

    prices = load_prices(run_name)
    sp = spans(prices)
    log(f"price log {run_name}: {prices.height:,} quotes, {sp.height} games")
    minutes = market.winner_minutes(season)
    rungs = market.rung_minutes(season, MAX_SPREAD)
    res: dict = {"season": season, "run": run_name, "deltas": list(DELTAS), "n_sims": int(prices["n_sims"].max())}
    res["prior"] = {
        "games": sp.height,
        "win_from_kalshi": int(prices.filter(pl.col("kind") == "pregame")["prior_win_source"].eq("kalshi").sum()),
        "total_from_kalshi": int(prices.filter(pl.col("kind") == "pregame")["prior_total_source"].eq("kalshi").sum()),
        "knob_win_gap_mean": float(prices.filter(pl.col("kind") == "pregame")
                                   .select((pl.col("knob_home_win") - pl.col("prior_home_win")).abs()
                                           .fill_nan(None).mean()).item()),
        "pregame_gap_sd": float(prices.filter(pl.col("kind") == "pregame")
                                .select((pl.col("p_home") - pl.col("prior_home_win")).fill_nan(None).std()).item()),
        "knobs_converged": int(prices.filter(pl.col("kind") == "pregame")["knob_converged"].sum()),
    }
    res["winner"], res["rungs"], res["info"] = {}, {}, {}
    grids = {}
    for d in DELTAS:
        g = winner_grid(prices, minutes, d)
        grids[d] = g
        res["winner"][str(d)] = winner_accuracy(g)
        w = res["winner"][str(d)]
        log(f"delta {d:>2}s winner: {w['rows']:,} minutes; Brier model {w['brier_model_corrected']:.4f} "
            f"(raw {w['brier_model']:.4f}) vs market {w['brier_market']:.4f}: {w['brier_corrected_diff']:+.4f} "
            f"[{w['brier_corrected_diff_p05']:+.4f}, {w['brier_corrected_diff_p95']:+.4f}]; log loss "
            f"{w['log_loss_diff']:+.4f} [{w['log_loss_diff_p05']:+.4f}, {w['log_loss_diff_p95']:+.4f}]")
        r = rung_grid(prices, rungs, d)
        res["rungs"][str(d)] = rung_accuracy(r)
        for kind, v in res["rungs"][str(d)].items():
            log(f"            {kind}: {v['rows']:,} rung-minutes, {v['contracts']} contracts; Brier diff "
                f"{v['brier_corrected_diff']:+.4f} [{v['brier_corrected_diff_p05']:+.4f}, {v['brier_corrected_diff_p95']:+.4f}]")
        res["info"][str(d)] = {str(k): info_test(g, k) for k in K_MINUTES}
        for k in K_MINUTES:
            it = res["info"][str(d)][str(k)]
            a, b = it["market_toward_model"], it["model_toward_market"]
            log(f"            {k} min: market toward model {a['slope']:+.3f} [{a['p05']:+.3f}, {a['p95']:+.3f}], "
                f"model toward market {b['slope']:+.3f} [{b['p05']:+.3f}, {b['p95']:+.3f}]")
    events_by = prices.group_by("kind").len()
    timed = prices.filter(pl.col("kind").is_in(["play", "final"])).group_by("timing").len()
    res["replay"] = {"events": dict(zip(events_by["kind"].to_list(), events_by["len"].to_list())),
                     "timing": dict(zip(timed["timing"].to_list(), timed["len"].to_list())),
                     "ms_per_event": float(prices.filter(pl.col("kind") != "final")["seconds"].mean() * 1000)}
    res["winner_minutes_too_wide"] = float(
        1 - grids[0].height / outside_reviews(prices, _in_game(prices, minutes), 0).height)
    res["calibration"] = calibration(grids[0])
    res["long_shots"] = long_shots(grids[0])
    res["direct"] = direct_accuracy(grids[0], prices)
    dd = res["direct"]
    log(f"Phase 1 model vs market (delta 0, {dd['direct']['rows']:,} minutes): Brier diff "
        f"{dd['direct']['brier_diff']:+.4f} [{dd['direct']['brier_diff_p05']:+.4f}, {dd['direct']['brier_diff_p95']:+.4f}]; "
        f"simulator on the same minutes {dd['simulator']['brier_corrected_diff']:+.4f}")
    events = load_events(season)
    trades = market.winner_trades(season)
    res["event_study"], curve = event_study(prices, events, trades, sp)
    e = res["event_study"]
    log(f"event study: {e['moves']} plays moved the model >= {MOVE:.2f}; the market made half the move "
        f"{e['half_move_s']}s after the play ended (90%: {e['half_move_s_p05']} to {e['half_move_s_p95']})")
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    curve.write_parquet(out / f"event_curve_{season}.parquet")
    grids[0].drop("q").write_parquet(out / f"winner_grid_{season}.parquet")
    (out / f"benchmark_{season}.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"  saved {out}")
    return res
