import numpy as np
import pytest

from gridline.eval import bootstrap

GAMES = np.repeat(["g1", "g2", "g3", "g4", "g5"], [3, 1, 4, 2, 5])


def test_a_resample_draws_whole_games():
    for w in bootstrap.game_weights(GAMES, n_boot=50, seed=1):
        for g in np.unique(GAMES):
            assert len(set(w[GAMES == g])) == 1  # all rows of a game move together
        per_game = [w[GAMES == g][0] for g in np.unique(GAMES)]
        assert sum(per_game) == 5  # as many draws as there are games


def test_resamples_are_reproducible():
    a = list(bootstrap.game_weights(GAMES, n_boot=5, seed=7))
    b = list(bootstrap.game_weights(GAMES, n_boot=5, seed=7))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_paired_difference_of_a_model_with_itself_is_zero():
    rng = np.random.default_rng(0)
    games = np.repeat(np.arange(50), 8)
    pred = rng.random(games.size)
    label = np.repeat(rng.random(50) < 0.5, 8).astype(int)
    qtr = np.tile(np.repeat([1, 2, 3, 4], 2), 50)
    out = bootstrap.paired(pred, pred, label, qtr, games, n_boot=20)
    assert set(out) == set(bootstrap.PAIRED_METRICS)
    assert all(d == {"diff": 0.0, "p05": 0.0, "p95": 0.0} for d in out.values())


def test_paired_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    games = np.repeat(np.arange(300), 20)
    truth = np.repeat(rng.random(300), 20)
    label = (np.repeat(rng.random(300), 20) < truth).astype(int)
    qtr = np.tile(np.repeat([1, 2, 3, 4], 5), 300)
    out = bootstrap.paired(truth, np.full(truth.size, 0.5), label, qtr, games, n_boot=200)
    assert out["log_loss"]["p95"] < 0 and out["brier"]["p95"] < 0


def test_spread_of_a_statistic():
    rng = np.random.default_rng(2)
    games = np.repeat(np.arange(200), 10)
    x = np.repeat(rng.normal(size=200), 10)
    s = bootstrap.spread(lambda w: float(np.average(x, weights=w)), games, n_boot=400)
    # SD of a mean of 200 independent games is about 1 / sqrt(200)
    assert s["sd"] == pytest.approx(1 / np.sqrt(200), rel=0.2)
    assert s["p05"] < s["mean"] < s["p95"]
