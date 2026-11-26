"""Stratified subsample by category x store, and the M5 sample builder on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from forecasting.m5 import M5Weekly
from forecasting.m5_sample import build_m5_sample
from forecasting.subsample import stratified_sample


def _meta(sizes: dict[tuple[str, str], int]) -> pd.DataFrame:
    rows = []
    for (cat, store), k in sizes.items():
        rows += [(f"{cat}_{i}_{store}", cat, store) for i in range(k)]
    return pd.DataFrame(rows, columns=["series_id", "cat_id", "store_id"])


SIZES = {
    ("FOODS", "S1"): 600,
    ("FOODS", "S2"): 600,
    ("HOBBIES", "S1"): 200,
    ("HOBBIES", "S2"): 200,
    ("HOUSEHOLD", "S1"): 400,
    ("HOUSEHOLD", "S2"): 400,
}


def test_returns_exactly_n_unique_rows():
    meta = _meta(SIZES)
    idx = stratified_sample(meta, n=300, seed=0)
    assert len(idx) == 300 and len(set(idx.tolist())) == 300


def test_allocation_is_proportional_to_stratum_size():
    meta = _meta(SIZES)  # 2,400 rows; n = 240 -> exactly 10% of each stratum
    idx = stratified_sample(meta, n=240, seed=0)
    got = meta.iloc[idx].groupby(["cat_id", "store_id"]).size().to_dict()
    assert got == {k: v // 10 for k, v in SIZES.items()}


def test_every_stratum_is_represented():
    meta = _meta(SIZES)
    idx = stratified_sample(meta, n=60, seed=3)
    assert meta.iloc[idx].groupby(["cat_id", "store_id"]).ngroups == len(SIZES)


def test_allocation_with_remainders_sums_to_n_and_stays_within_one_of_target():
    meta = _meta({("A", "S1"): 333, ("A", "S2"): 333, ("B", "S1"): 334})
    idx = stratified_sample(meta, n=100, seed=0)
    got = meta.iloc[idx].groupby(["cat_id", "store_id"]).size()
    assert got.sum() == 100
    assert (got - 100 * pd.Series([333, 333, 334], index=got.index) / 1000).abs().max() < 1


def test_same_seed_reproduces_and_different_seed_differs():
    meta = _meta(SIZES)
    a = stratified_sample(meta, n=300, seed=7)
    b = stratified_sample(meta, n=300, seed=7)
    c = stratified_sample(meta, n=300, seed=8)
    assert a.tolist() == b.tolist()
    assert set(a.tolist()) != set(c.tolist())


def test_selection_does_not_depend_on_row_order():
    meta = _meta(SIZES)
    shuffled = meta.sample(frac=1.0, random_state=5).reset_index(drop=True)
    a = set(meta.iloc[stratified_sample(meta, n=300, seed=7)]["series_id"])
    b = set(shuffled.iloc[stratified_sample(shuffled, n=300, seed=7)]["series_id"])
    assert a == b


def test_rejects_n_larger_than_population_or_fewer_than_strata():
    meta = _meta(SIZES)
    with pytest.raises(ValueError):
        stratified_sample(meta, n=len(meta) + 1, seed=0)
    with pytest.raises(ValueError):
        stratified_sample(meta, n=len(SIZES) - 1, seed=0)


def _fake_m5(n_series=120, n_weeks=200, seed=0) -> M5Weekly:
    rng = np.random.default_rng(seed)
    meta = _meta({("FOODS", "S1"): 50, ("FOODS", "S2"): 50, ("HOBBIES", "S1"): 20})
    meta["first_week"] = 0
    sales = rng.poisson(rng.uniform(0.1, 15, size=(n_series, 1)), size=(n_series, n_weeks))
    return M5Weekly(meta=meta, sales=sales.astype(np.int32), week_ids=np.arange(n_weeks) + 1000)


def test_build_sample_excludes_short_history_and_no_demand_series():
    data = _fake_m5()
    data.meta.loc[0, "first_week"] = 190  # only 10 weeks before the holdout cutoff -> too short
    data.sales[1, :148] = 0  # no demand in the classification window (200 - 52 = 148)
    res = build_m5_sample(data, n=60, seed=0, holdout_weeks=52, min_history_weeks=104)
    ids = set(res.sample["series_id"])
    assert data.meta.series_id[0] not in ids and data.meta.series_id[1] not in ids
    assert res.n_excluded == {"short_history": 1, "no_demand": 1}


def test_build_sample_classifies_on_pre_holdout_window_only():
    data = _fake_m5()
    data.sales[2, :148] = 10  # smooth before the holdout
    data.sales[2, 148:] = np.where(np.arange(52) % 2 == 0, 0, 900)  # wild inside the holdout
    res = build_m5_sample(data, n=117, seed=0, holdout_weeks=52, min_history_weeks=104)
    row = res.population.set_index("series_id").loc[data.meta.series_id[2]]
    assert row["segment"] == "smooth" and row["adi"] == 1.0
    assert row["window_end"] == 148


def test_build_sample_counts_are_consistent():
    data = _fake_m5()
    res = build_m5_sample(data, n=60, seed=0)
    assert len(res.sample) == 60
    assert res.sample_counts().sum() == 60
    assert res.population_counts().sum() == len(res.population)
    assert set(res.sample["series_id"]) <= set(res.population["series_id"])
