"""Build the M5 evaluation sample: eligibility, SB segmentation, stratified subsample.

    python -m forecasting.m5_sample [--n 4000] [--seed 0]

Segments are computed on each series' listed history before the last `holdout_weeks` weeks
(default 52 = `simulation.eval_weeks`), so labels never use the evaluation year. Segment is not a
stratification variable: sample segment counts are an outcome, reported next to the population's.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from forecasting.m5 import M5Weekly, load_m5_weekly
from forecasting.segmentation import SEGMENTS, segment_windows
from forecasting.subsample import stratified_sample

DEFAULT_N = 4000  # file 03: 3,000-5,000 series
DEFAULT_HOLDOUT_WEEKS = 52
# Room for 8-week horizons, rolling origins and >= 26 residual origins (bootstrap_min_origins).
DEFAULT_MIN_HISTORY_WEEKS = 104
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "m5_sample_series.csv"


@dataclass(frozen=True)
class M5Sample:
    population: pd.DataFrame  # eligible series with segment columns
    sample: pd.DataFrame  # the stratified subsample (subset of population rows)
    n_excluded: dict[str, int] = field(default_factory=dict)

    def population_counts(self) -> pd.Series:
        return _counts(self.population)

    def sample_counts(self) -> pd.Series:
        return _counts(self.sample)


def _counts(df: pd.DataFrame) -> pd.Series:
    return df["segment"].value_counts().reindex(list(SEGMENTS), fill_value=0)


def build_m5_sample(
    data: M5Weekly,
    n: int = DEFAULT_N,
    seed: int = 0,
    holdout_weeks: int = DEFAULT_HOLDOUT_WEEKS,
    min_history_weeks: int = DEFAULT_MIN_HISTORY_WEEKS,
) -> M5Sample:
    window_end = data.sales.shape[1] - holdout_weeks
    seg = segment_windows(data.sales, start=data.meta["first_week"].to_numpy(), end=window_end)
    table = data.meta.reset_index(drop=True).join(seg)
    table["window_end"] = window_end

    short = table["n_weeks"] < min_history_weeks
    no_demand = ~short & (table["n_nonzero"] == 0)
    population = table[~short & ~no_demand].reset_index(drop=True)
    excluded = {"short_history": int(short.sum()), "no_demand": int(no_demand.sum())}

    idx = stratified_sample(population, n=n, seed=seed)
    return M5Sample(population, population.iloc[idx].reset_index(drop=True), excluded)


def _report(label: str, res: M5Sample) -> None:
    pop, smp = res.population_counts(), res.sample_counts()
    print(f"\n{label}")
    print(
        pd.DataFrame(
            {"population": pop, "sample": smp, "sample_%": (100 * smp / smp.sum()).round(1)}
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_m5_weekly()
    print(f"loaded {data.sales.shape[0]:,} series x {data.sales.shape[1]} weeks")
    res = build_m5_sample(data, n=args.n, seed=args.seed)
    print(f"excluded from the {len(data.meta):,}: {res.n_excluded}")
    _report(f"segments on weeks before the last {DEFAULT_HOLDOUT_WEEKS} (primary)", res)

    full = build_m5_sample(data, n=args.n, seed=args.seed, holdout_weeks=0)
    _report("sensitivity: segments on the full history", full)

    strata = res.sample.groupby(["cat_id", "store_id"]).size().unstack()
    print("\nsample size by category x store\n", strata)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.sample.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} (gitignored)")


if __name__ == "__main__":
    main()
