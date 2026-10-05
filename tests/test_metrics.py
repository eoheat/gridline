import numpy as np
import pytest

from gridline.eval import metrics


def test_calibration_error_by_hand():
    # quarter 1: 4 plays at 0.30 (bin 0.30), 1 win -> actual 0.25 -> |diff| 0.05, 1 win
    # quarter 2: 2 plays at 0.72 (bin 0.70), 2 wins -> actual 1.00 -> |diff| 0.30, 2 wins
    pred = [0.3] * 4 + [0.72] * 2
    label = [1, 0, 0, 0, 1, 1]
    qtr = [1] * 4 + [2] * 2
    # per-quarter errors weighted by wins in the quarter: (0.05*1 + 0.30*2) / 3
    assert metrics.calibration_error(pred, label, qtr) == pytest.approx((0.05 + 0.60) / 3)
    table = metrics.calibration_table(pred, label, qtr)
    assert table.select("qtr", "n_plays", "n_wins").rows() == [(1, 4, 1), (2, 2, 2)]
    assert table["bin"].to_list() == pytest.approx([0.30, 0.70])
    assert table["actual"].to_list() == pytest.approx([0.25, 1.0])


def test_perfect_calibration_is_zero():
    rng = np.random.default_rng(1)
    pred = np.repeat([0.1, 0.5, 0.9], 20_000)
    label = rng.random(pred.size) < pred
    assert metrics.calibration_error(pred, label, np.ones(pred.size)) < 0.01


def test_weights_count_rows_repeatedly():
    # a weight of k must give exactly what k copies of the row give (the bootstrap relies on it)
    rng = np.random.default_rng(3)
    pred, qtr = rng.random(400), rng.integers(1, 5, 400)
    label = (rng.random(400) < pred).astype(int)
    w = rng.integers(0, 4, 400)
    rep = np.repeat(np.arange(400), w)
    weighted = metrics.summary(pred, label, qtr, w)
    repeated = metrics.summary(pred[rep], label[rep], qtr[rep])
    for k in ("calibration_error", "log_loss", "brier", "error_rate"):
        assert weighted[k] == pytest.approx(repeated[k], rel=1e-12), k


def test_other_metrics():
    assert metrics.brier([1.0, 0.0], [1, 0]) == 0.0
    assert metrics.error_rate([0.9, 0.2, 0.6], [1, 1, 0]) == pytest.approx(2 / 3)
    assert metrics.log_loss([0.5], [1]) == pytest.approx(np.log(2))
