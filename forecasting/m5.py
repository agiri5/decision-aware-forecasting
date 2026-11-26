"""M5 (Walmart) loading and daily -> weekly aggregation.

Weeks are Walmart weeks (`wm_yr_wk`, Saturday to Friday). Only complete 7-day weeks are kept, so
the last two days of the evaluation file (d_1940, d_1941) are dropped. Observed sales are censored
demand: a zero may be a stockout (see docs/simulator_spec.md, section 10).

Leading zeros before an item is listed at a store are not demand. `first_week` marks the first
kept week in which the item has a sell price; weeks before it must be ignored downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR_ENV = "DAF_RAW_DATA_DIR"
M5_SUBDIR = "m5-forecasting-accuracy"
# The evaluation file is the validation file plus the 28 days that follow it.
SALES_FILE = "sales_train_evaluation.csv"
DAYS_PER_WEEK = 7

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def m5_dir() -> Path:
    """`$DAF_RAW_DATA_DIR/m5-forecasting-accuracy`; default raw dir is `../Data` from the repo."""
    root = Path(os.environ.get(RAW_DIR_ENV, Path(__file__).resolve().parents[2] / "Data"))
    return root / M5_SUBDIR


@dataclass(frozen=True)
class M5Weekly:
    """Weekly sales for every item-store series.

    meta:     one row per series: series_id, the M5 id columns, first_week
    sales:    (n_series, n_weeks) units per Walmart week, rows aligned with `meta`
    week_ids: (n_weeks,) `wm_yr_wk` of each column, increasing
    """

    meta: pd.DataFrame
    sales: np.ndarray
    week_ids: np.ndarray


def daily_to_weekly(daily: np.ndarray, day_week: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum daily columns into weeks, keeping only weeks with exactly 7 days.

    daily: (n_series, n_days); day_week: (n_days,) week id of each day, non-decreasing.
    Returns (weekly, week_ids).
    """
    daily = np.asarray(daily)
    day_week = np.asarray(day_week)
    if daily.ndim != 2 or day_week.shape != (daily.shape[1],):
        raise ValueError("daily must be (n_series, n_days) and day_week (n_days,)")
    if np.any(np.diff(day_week) < 0):
        raise ValueError("day_week must be non-decreasing")
    week_ids, starts, counts = np.unique(day_week, return_index=True, return_counts=True)
    full = counts == DAYS_PER_WEEK
    weekly = np.add.reduceat(daily, starts, axis=1)[:, full]
    return weekly, week_ids[full]


def first_listed_week(meta: pd.DataFrame, prices: pd.DataFrame, week_ids: np.ndarray) -> np.ndarray:
    """Index into `week_ids` of the first kept week on or after each series' first price row.

    0 if listed before the window starts; `len(week_ids)` if the series never has a price.
    meta needs item_id, store_id; prices needs item_id, store_id, wm_yr_wk.
    """
    first_price = prices.groupby(["item_id", "store_id"], observed=True)["wm_yr_wk"].min()
    key = pd.MultiIndex.from_frame(meta[["item_id", "store_id"]])
    listed = first_price.reindex(key).to_numpy()
    idx = np.searchsorted(week_ids, np.nan_to_num(listed, nan=np.inf), side="left")
    return idx.astype(int)


def load_m5_weekly(raw_dir: str | Path | None = None) -> M5Weekly:
    """Read the M5 files and return weekly sales for all 30,490 item-store series."""
    root = Path(raw_dir) if raw_dir is not None else m5_dir()
    header = pd.read_csv(root / SALES_FILE, nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]
    dtypes = {c: np.int32 for c in day_cols}
    sales = pd.read_csv(root / SALES_FILE, dtype=dtypes)

    calendar = pd.read_csv(root / "calendar.csv", usecols=["d", "wm_yr_wk"]).set_index("d")
    day_week = calendar.loc[day_cols, "wm_yr_wk"].to_numpy()
    weekly, week_ids = daily_to_weekly(sales[day_cols].to_numpy(), day_week)

    meta = sales[ID_COLS].copy()
    meta.insert(0, "series_id", meta["item_id"] + "_" + meta["store_id"])
    prices = pd.read_csv(
        root / "sell_prices.csv",
        usecols=["store_id", "item_id", "wm_yr_wk"],
        dtype={"store_id": "category", "item_id": "category"},
    )
    meta["first_week"] = first_listed_week(meta, prices, week_ids)
    return M5Weekly(meta=meta, sales=weekly.astype(np.int32), week_ids=week_ids)
