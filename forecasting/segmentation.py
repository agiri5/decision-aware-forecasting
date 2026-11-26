"""Syntetos-Boylan demand segmentation on weekly series.

ADI = periods in the window / periods with demand. CV2 = squared coefficient of variation of the
non-zero demand sizes (population variance). Cutoffs 1.32 and 0.49 are the published SB values,
derived for forecast-method choice, not tuned here. A boundary value belongs to the higher class
(ADI >= 1.32 -> intermittent/lumpy; CV2 >= 0.49 -> erratic/lumpy). Weekly aggregation moves many
daily-intermittent series into smooth/erratic; the cutoffs are applied to weekly data as is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49
SEGMENTS = ("smooth", "erratic", "intermittent", "lumpy")
NO_DEMAND = "no_demand"


def classify(adi: float, cv2: float) -> str:
    """SB quadrant for one series; `no_demand` when there is no non-zero week."""
    if np.isnan(adi) or np.isnan(cv2):
        return NO_DEMAND
    if adi < ADI_CUTOFF:
        return "smooth" if cv2 < CV2_CUTOFF else "erratic"
    return "intermittent" if cv2 < CV2_CUTOFF else "lumpy"


def segment_windows(sales: np.ndarray, start: np.ndarray, end: int) -> pd.DataFrame:
    """ADI, CV2 and segment per row of `sales` over weeks [start_i, end).

    sales: (n_series, n_weeks); start: (n_series,) first week of each series' window.
    Returns a frame with n_weeks, n_nonzero, adi, cv2, segment (row-aligned with `sales`).
    """
    sales = np.asarray(sales, dtype=float)
    start = np.clip(np.asarray(start), 0, end)
    cols = np.arange(sales.shape[1])
    in_window = (cols[None, :] >= start[:, None]) & (cols[None, :] < end)
    demand = in_window & (sales > 0)

    n_weeks = end - start
    n_nonzero = demand.sum(axis=1)
    has_demand = n_nonzero > 0
    safe_n = np.where(has_demand, n_nonzero, 1)
    mean = (sales * demand).sum(axis=1) / safe_n
    var = (((sales - mean[:, None]) ** 2) * demand).sum(axis=1) / safe_n

    with np.errstate(invalid="ignore", divide="ignore"):
        adi = np.where(has_demand, n_weeks / safe_n, np.nan)
        cv2 = np.where(has_demand, var / mean**2, np.nan)
    segment = [classify(a, c) for a, c in zip(adi, cv2, strict=True)]
    return pd.DataFrame(
        {"n_weeks": n_weeks, "n_nonzero": n_nonzero, "adi": adi, "cv2": cv2, "segment": segment}
    )


def adi_cv2(x: np.ndarray) -> tuple[float, float]:
    """(ADI, CV2) of one series; (nan, nan) if it has no non-zero value."""
    x = np.asarray(x)[None, :]
    out = segment_windows(x, start=np.zeros(1, dtype=int), end=x.shape[1])
    return float(out["adi"].iloc[0]), float(out["cv2"].iloc[0])
