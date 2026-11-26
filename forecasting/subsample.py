"""Stratified subsampling of series by category x store (proportional allocation)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def stratified_sample(
    meta: pd.DataFrame,
    n: int,
    strata_cols: Sequence[str] = ("cat_id", "store_id"),
    seed: int = 0,
) -> np.ndarray:
    """Positional row indices (sorted) of an n-row sample, proportional to stratum size.

    Per-stratum counts use largest-remainder rounding. A stratum that would get no draws raises,
    rather than silently dropping out. Draws depend on `series_id` and the seed, not on row order.
    """
    if not 0 < n <= len(meta):
        raise ValueError(f"n must be in [1, {len(meta)}], got {n}")
    if not meta["series_id"].is_unique:
        raise ValueError("series_id must be unique")

    groups = meta.groupby(list(strata_cols), sort=True, observed=True).indices
    keys = list(groups)
    sizes = np.array([len(groups[k]) for k in keys])
    target = n * sizes / sizes.sum()
    counts = np.floor(target).astype(int)
    short = n - counts.sum()
    # stable sort: ties in the remainder go to the earlier (sorted) stratum
    counts[np.argsort(-(target - counts), kind="stable")[:short]] += 1
    if np.any(counts == 0):
        empty = [keys[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"n={n} is too small: strata with no draws: {empty}")

    ids = meta["series_id"].to_numpy()
    chosen = []
    for i, key in enumerate(keys):
        rows = groups[key]
        rows = rows[np.argsort(ids[rows])]
        rng = np.random.default_rng([seed, i])
        chosen.append(rng.choice(rows, size=counts[i], replace=False))
    return np.sort(np.concatenate(chosen))
