"""Smoke checks on the real M5 files. Skipped when the raw data is not available locally."""

import numpy as np
import pytest

from forecasting.m5 import load_m5_weekly, m5_dir

pytestmark = pytest.mark.skipif(
    not (m5_dir() / "sales_train_evaluation.csv").exists(), reason="M5 raw data not found"
)


@pytest.fixture(scope="module")
def m5():
    return load_m5_weekly()


def test_shape_and_week_alignment(m5):
    # 30,490 item-store series; 1,941 days = 277 full Walmart weeks + 2 dropped partial days
    assert m5.sales.shape == (30490, 277)
    assert m5.meta.series_id.is_unique
    assert np.all(np.diff(m5.week_ids) > 0)
    assert m5.sales.min() >= 0


def test_weekly_total_matches_daily_total_minus_partial_week(m5):
    import pandas as pd

    daily = pd.read_csv(m5_dir() / "sales_train_evaluation.csv").filter(regex=r"^d_").to_numpy()
    assert daily.shape[1] == 1941
    # the dropped partial week is d_1940 and d_1941
    assert m5.sales.sum() == daily[:, :1939].sum()


def test_no_sales_before_first_listed_week(m5):
    """Validates the price-based listing start: sales before it must be zero."""
    cols = np.arange(m5.sales.shape[1])
    before = cols[None, :] < m5.meta.first_week.to_numpy()[:, None]
    assert m5.sales[before].sum() == 0
    # a large share of series is listed late, so the trim matters for ADI / CV2
    assert (m5.meta.first_week > 0).mean() > 0.3
    assert m5.meta.first_week.max() <= m5.sales.shape[1]
