"""M5 daily -> weekly aggregation and listing-start detection (synthetic inputs, no raw data)."""

import numpy as np
import pandas as pd
import pytest

from forecasting.m5 import daily_to_weekly, first_listed_week


def test_daily_to_weekly_sums_complete_weeks():
    daily = np.arange(1, 22).reshape(1, 21)  # 3 full weeks of 7 days
    day_week = np.repeat([100, 101, 102], 7)
    weekly, week_ids = daily_to_weekly(daily, day_week)
    assert week_ids.tolist() == [100, 101, 102]
    assert weekly.tolist() == [[28, 77, 126]]


def test_daily_to_weekly_drops_partial_weeks():
    # 3 days of week 100 (partial), full week 101, 2 days of week 102 (partial)
    day_week = np.array([100] * 3 + [101] * 7 + [102] * 2)
    daily = np.ones((2, day_week.size), dtype=int)
    weekly, week_ids = daily_to_weekly(daily, day_week)
    assert week_ids.tolist() == [101]
    assert weekly.tolist() == [[7], [7]]


def test_daily_to_weekly_conserves_units_over_kept_weeks():
    rng = np.random.default_rng(0)
    day_week = np.repeat(np.arange(50, 60), 7)
    daily = rng.poisson(3, size=(5, day_week.size))
    weekly, _ = daily_to_weekly(daily, day_week)
    assert weekly.sum() == daily.sum()


def test_daily_to_weekly_rejects_unsorted_weeks():
    with pytest.raises(ValueError):
        daily_to_weekly(np.ones((1, 14), dtype=int), np.repeat([101, 100], 7))


def test_daily_to_weekly_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        daily_to_weekly(np.ones((1, 10), dtype=int), np.repeat([100, 101], 7))


def test_first_listed_week_uses_first_price_row():
    meta = pd.DataFrame({"item_id": ["A", "A", "B", "C"], "store_id": ["S1", "S2", "S1", "S1"]})
    week_ids = np.array([100, 101, 102, 103, 105])  # wm_yr_wk can skip numbers
    prices = pd.DataFrame(
        {
            "item_id": ["A", "A", "A", "B", "B"],
            "store_id": ["S1", "S1", "S2", "S1", "S1"],
            "wm_yr_wk": [101, 102, 99, 103, 105],
        }
    )
    got = first_listed_week(meta, prices, week_ids)
    # A@S1 -> idx 1; A@S2 priced before the window -> 0; B@S1 -> idx 3; C never priced -> n_weeks
    assert got.tolist() == [1, 0, 3, 5]


def test_first_listed_week_price_in_skipped_week_maps_to_next_kept_week():
    meta = pd.DataFrame({"item_id": ["A"], "store_id": ["S1"]})
    week_ids = np.array([100, 102])  # week 101 was a dropped partial week
    prices = pd.DataFrame({"item_id": ["A"], "store_id": ["S1"], "wm_yr_wk": [101]})
    assert first_listed_week(meta, prices, week_ids).tolist() == [1]
