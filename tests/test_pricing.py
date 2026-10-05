import numpy as np
import pytest

from gridline.models import transitions as tr
from gridline.pricing.contracts import ScoreDistribution


def test_contract_prices_follow_kalshi_rules():
    # four equally likely finals: home wins by 7, by 3, a 20-20 tie, away wins by 1
    d = ScoreDistribution(np.array([27.0, 23, 20, 16]), np.array([20.0, 20, 20, 17]))
    assert d.winner("home") == pytest.approx(0.5 + 0.125)  # a tie settles 50/50
    assert d.winner("home") + d.winner("away") == pytest.approx(1)
    assert d.spread("home", 3.5) == 0.25 and d.spread("home", 2.5) == 0.5
    assert d.spread("away", 0.5) == 0.25
    assert d.over(40.5) == 0.5 and d.over(33.5) == 0.75
    assert d.price("spread", "home", 6.5) == 0.25 and d.price("total", None, 46.5) == 0.25
    assert list(d.ladder("total", [30.5, 50.5])) == [1.0, 0.0]


def test_pool_draws_within_each_bucket():
    pool = tr.Pool(start=np.array([0, 2]), count=np.array([2, 3]), same=np.zeros(5, bool),
                   yardline=np.arange(5.0), seconds=np.zeros(5), points_a=np.zeros(5), points_b=np.zeros(5))
    rows = pool.draw(np.array([0, 0, 1, 1, 1]), np.array([0.0, 0.99, 0.0, 0.5, 0.999]))
    assert list(rows) == [0, 1, 2, 3, 4]


def test_possession_buckets_are_distinct_and_in_range():
    b = tr.possession_bucket(np.array([0, 3, 3]), np.array([1, 99, 99]), np.array([1, 4, 1]))
    assert len(set(b)) == 3 and b.max() < tr.N_POSSESSION_BUCKETS and b.min() >= 0
