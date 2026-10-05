"""Phase 3 benchmark statistics on synthetic model and market series with known answers."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from gridline.eval import benchmark as bm

T0 = dt.datetime(2025, 9, 7, 17, 0, tzinfo=dt.timezone.utc)


def at(seconds):
    return T0 + dt.timedelta(seconds=float(seconds))


def price_log(game_id, quotes, final=(24, 17)):
    """quotes: (seconds, kind, seq, p_home), in order; a pregame quote first."""
    rows = [{"game_id": game_id, "order": i, "seq": seq, "kind": kind, "t": at(s), "timing": "end",
             "t_play_end": at(s),
             "p_home": p, "p_tie": 0.0, "qtr": 1, "game_seconds_remaining": 3000.0,
             "n_sims": 0 if kind == "final" else 2000,
             "home_score": final[0] if kind == "final" else 0, "away_score": final[1] if kind == "final" else 0}
            for i, (s, kind, seq, p) in enumerate(quotes)]
    return pl.DataFrame(rows)


def test_model_price_is_the_last_quote_known_by_then():
    prices = price_log("G", [(0, "pregame", -1, 0.5), (10, "play", 1, 0.6), (10, "snap", 2, 0.65),
                             (20, "play", 2, 0.7), (100, "final", 3, 1.0)]).with_row_index("q")
    times = pl.DataFrame({"game_id": ["G"] * 5, "t": [at(s) for s in (5, 10, 14, 16, 25)]})
    got = bm.model_asof(prices, times, 0)["p_home"].to_list()
    assert got == [0.5, 0.65, 0.65, 0.65, 0.7]  # two quotes at 10 s: the later one counts
    got = bm.model_asof(prices, times, 5)["p_home"].to_list()
    assert got == [0.5, 0.5, 0.5, 0.65, 0.7]
    # the pregame price is Kalshi's own, known at kickoff whatever the delay
    early = pl.DataFrame({"game_id": ["G"] * 2, "t": [at(0), at(12)]})
    assert bm.model_asof(prices, early, 20)["p_home"].to_list() == [0.5, 0.5]


def test_minutes_with_a_review_pending_are_left_out():
    prices = price_log("G", [(0, "pregame", -1, 0.5), (100, "play", 1, 0.6), (400, "play", 2, 0.9),
                             (500, "final", 3, 1.0)]).with_row_index("q")
    prices = prices.with_columns(  # play 2 ended at 190 s and its review ended at 400 s
        timing=pl.when(pl.col("seq") == 2).then(pl.lit("review")).otherwise(pl.col("timing")),
        t_play_end=pl.when(pl.col("seq") == 2).then(pl.lit(at(190))).otherwise(pl.col("t_play_end")))
    times = pl.DataFrame({"game_id": ["G"] * 5, "t": [at(s) for s in (120, 180, 240, 400, 420)]})
    kept = bm.outside_reviews(prices, times, 10)["t"].to_list()
    assert kept == [at(120), at(180), at(420)]


def test_paired_scores_and_monte_carlo_correction():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(float)
    p = rng.uniform(0.2, 0.8, 400)
    games = np.repeat(np.arange(40), 10)
    same = bm.paired(p, p, y, games)
    assert same["brier_diff"] == 0 and same["log_loss_diff"] == 0 and same["brier_diff_p05"] == 0
    var = np.full(400, 1e-3)
    cor = bm.paired(p, p, y, games, var)
    assert cor["brier_corrected_diff"] == pytest.approx(-1e-3)
    # a noisy copy of a forecast scores worse by about its variance, which the correction removes
    noisy = np.clip(p + rng.normal(0, 0.05, 400), 0.01, 0.99)
    r = bm.paired(noisy, p, y, games, np.full(400, 0.05 ** 2))
    assert abs(r["brier_corrected_diff"]) < abs(r["brier_diff"]) + 1e-9


def grid_frame(model, market, game_len=200, games=30):
    rows = []
    for g in range(games):
        for i in range(game_len):
            rows.append({"game_id": f"G{g}", "t": at(60 * i), "t_final": at(60 * (game_len + 5)),
                         "model": model[g, i], "market": market[g, i]})
    return pl.DataFrame(rows)


def test_information_test_finds_a_market_that_lags_the_model():
    rng = np.random.default_rng(1)
    steps = rng.normal(0, 0.02, (30, 201))
    model = 0.5 + np.cumsum(steps, axis=1)
    lagging = np.concatenate([np.full((30, 1), 0.5), model[:, :-1]], axis=1)  # one minute behind
    r = bm.info_test(grid_frame(model[:, :200], lagging[:, :200]), 1)
    assert r["market_toward_model"]["slope"] == pytest.approx(1.0, abs=0.05)
    assert r["model_toward_market"]["slope"] == pytest.approx(0.0, abs=0.1)
    unrelated = 0.5 + np.cumsum(rng.normal(0, 0.02, (30, 200)), axis=1)
    r = bm.info_test(grid_frame(model[:, :200], unrelated), 1)
    assert abs(r["market_toward_model"]["slope"]) < 0.05
    assert r["market_toward_model"]["p05"] < 0 < r["market_toward_model"]["p95"]


def test_event_study_times_the_market_response():
    prices, events, trades = [], [], []
    for g in range(12):
        gid = f"G{g}"
        base = 1000 * g
        up = g % 2 == 0  # half the plays help home, half hurt
        p1 = 0.6 if up else 0.4
        prices.append(price_log(gid, [(base, "pregame", -1, 0.5), (base + 100, "play", 7, p1),
                                      (base + 900, "final", 9, 1.0 if up else 0.0)],
                                final=(21, 14) if up else (14, 21)))
        events.append({"game_id": gid, "seq": 7, "t_snap": at(base + 92), "play_type": "pass", "desc": ""})
        # the market sits at 0.50, gets halfway 4 s after the play ends and all the way at 12 s
        for s, px in ((base + 50, 0.5), (base + 104, 0.55 if up else 0.45), (base + 112, p1)):
            trades.append({"game_id": gid, "ts": at(s), "price": px})
    prices = pl.concat(prices).with_row_index("q")
    res, curve = bm.event_study(prices, pl.DataFrame(events), pl.DataFrame(trades).sort("game_id", "ts"),
                                bm.spans(prices))
    assert res["moves"] == 12 and res["half_move_s"] == 4.0
    assert res["by_delta"]["0"]["left_cents"] == pytest.approx(10.0)
    assert res["by_delta"]["5"]["left_cents"] == pytest.approx(5.0)
    assert res["by_delta"]["20"]["left_cents"] == pytest.approx(0.0)
    # trading the model's direction at the market price at t_end + 5 s: bought at 55,
    # settled at 100 (and the mirror image)
    assert res["by_delta"]["5"]["pnl_cents"] == pytest.approx(45.0)
    # less Kalshi's taker fee, 0.07 * P * (1 - P) a contract at 55 (or 45) cents
    assert res["by_delta"]["5"]["net_cents"] == pytest.approx(45.0 - 7 * 0.55 * 0.45)
    assert curve.filter(pl.col("tau") == 12)["share"][0] == pytest.approx(1.0)


def test_long_shots_count_either_side_of_the_book():
    g = pl.DataFrame({"game_id": ["A", "A", "B", "C"], "market": [0.05, 0.08, 0.97, 0.50],
                      "model": [0.04, 0.06, 0.98, 0.5], "y": [1.0, 1.0, 1.0, 0.0]})
    r = bm.long_shots(g)
    # A's home team at 5 and 8 cents came back to win; B's away team at 3 cents lost
    assert (r["minutes"], r["games"], r["games_won"]) == (3, 2, 1)
    assert r["price"] == pytest.approx((0.05 + 0.08 + 0.03) / 3)
    assert r["won"] == pytest.approx(2 / 3) and r["model"] == pytest.approx((0.04 + 0.06 + 0.02) / 3)
