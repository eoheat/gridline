import numpy as np
import pytest

from gridline.eval import distribution as dist


def test_crps_of_a_point_forecast_is_the_absolute_error():
    cdf = dist.cdf_from_samples(np.full(10, 3), dist.MARGIN_GRID)
    assert dist.crps(cdf, dist.MARGIN_GRID, 10) == pytest.approx(7)
    assert dist.crps(cdf, dist.MARGIN_GRID, 3) == 0


def test_pit_is_uniform_for_a_calibrated_integer_forecast():
    rng = np.random.default_rng(0)
    samples = rng.poisson(20, 5000)
    cdf = dist.cdf_from_samples(samples, dist.TOTAL_GRID)
    pits = [dist.pit(cdf, dist.TOTAL_GRID, y, v) for y, v in zip(rng.poisson(20, 4000), rng.random(4000))]
    assert np.histogram(pits, bins=10, range=(0, 1))[0].min() > 330  # about 400 per bin
    assert dist.coverage(pits, 0.5) == pytest.approx(0.5, abs=0.03)
    assert dist.coverage(pits, 0.9) == pytest.approx(0.9, abs=0.02)


def test_pit_flags_a_forecast_that_is_too_narrow():
    rng = np.random.default_rng(1)
    cdf = dist.cdf_normal(0, 5, dist.MARGIN_GRID)  # truth has sd 13
    pits = [dist.pit(cdf, dist.MARGIN_GRID, y, v)
            for y, v in zip(np.round(rng.normal(0, 13, 3000)), rng.random(3000))]
    assert dist.coverage(pits, 0.5) < 0.3


def test_normal_cdf_is_rounded_to_integers():
    cdf = dist.cdf_normal(0.0, 1.0, np.array([-1, 0, 1]))
    assert cdf[1] == pytest.approx(0.6915, abs=1e-4)  # P(X < 0.5)
