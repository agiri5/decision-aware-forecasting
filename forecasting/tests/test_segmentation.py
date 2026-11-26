"""Syntetos-Boylan segmentation: ADI / CV2 definitions, cutoffs and edge cases."""

import numpy as np
import pytest

from forecasting.segmentation import (
    ADI_CUTOFF,
    CV2_CUTOFF,
    SEGMENTS,
    adi_cv2,
    classify,
    segment_windows,
)


def test_cutoffs_are_the_documented_ones():
    assert (ADI_CUTOFF, CV2_CUTOFF) == (1.32, 0.49)
    assert set(SEGMENTS) == {"smooth", "erratic", "intermittent", "lumpy"}


def test_adi_is_periods_per_nonzero_demand():
    adi, _ = adi_cv2(np.array([1, 0, 1, 0]))
    assert adi == 2.0
    adi, _ = adi_cv2(np.array([3, 4, 5]))
    assert adi == 1.0


def test_cv2_uses_nonzero_sizes_only_population_variance():
    # nonzero sizes 2 and 6: mean 4, population var 4 -> CV2 = 4 / 16
    _, cv2 = adi_cv2(np.array([2, 0, 0, 6]))
    assert cv2 == pytest.approx(0.25)


def test_constant_demand_size_has_zero_cv2():
    _, cv2 = adi_cv2(np.array([5, 0, 5, 0, 5]))
    assert cv2 == 0.0


def test_all_zero_series_is_nan_and_unclassified():
    adi, cv2 = adi_cv2(np.zeros(10, dtype=int))
    assert np.isnan(adi) and np.isnan(cv2)
    assert classify(adi, cv2) == "no_demand"


@pytest.mark.parametrize(
    "adi, cv2, expected",
    [
        (1.0, 0.1, "smooth"),
        (1.0, 1.0, "erratic"),
        (3.0, 0.1, "intermittent"),
        (3.0, 1.0, "lumpy"),
        # boundaries: ADI >= 1.32 is intermittent/lumpy; CV2 >= 0.49 is erratic/lumpy
        (1.32, 0.0, "intermittent"),
        (1.3199, 0.0, "smooth"),
        (1.0, 0.49, "erratic"),
        (1.0, 0.4899, "smooth"),
        (1.32, 0.49, "lumpy"),
    ],
)
def test_classify_quadrants_and_boundaries(adi, cv2, expected):
    assert classify(adi, cv2) == expected


def _four_archetypes(n=104):
    t = np.arange(n)
    smooth = 20 + (t % 3)  # every week, low variability
    erratic = np.where(t % 2 == 0, 1, 40)  # every week, high variability
    intermittent = np.where(t % 4 == 0, 5, 0)  # every 4th week, constant size
    lumpy = np.where(t % 4 == 0, np.where(t % 8 == 0, 1, 40), 0)
    return np.vstack([smooth, erratic, intermittent, lumpy])


def test_segment_windows_labels_archetypes():
    sales = _four_archetypes()
    n = sales.shape[1]
    out = segment_windows(sales, start=np.zeros(4, dtype=int), end=n)
    assert out["segment"].tolist() == ["smooth", "erratic", "intermittent", "lumpy"]
    assert out["n_weeks"].tolist() == [n] * 4
    assert out["n_nonzero"].tolist() == [n, n, n // 4, n // 4]


def test_segment_windows_ignores_weeks_before_start_and_after_end():
    n = 104
    t = np.arange(n)
    # zeros before listing and a wild tail after `end` must not change the label
    x = np.where(t < 20, 0, 20 + (t % 3))
    x[90:] = np.where(t[90:] % 2 == 0, 0, 500)
    out = segment_windows(x[None, :], start=np.array([20]), end=90)
    assert out["segment"].tolist() == ["smooth"]
    assert out["n_weeks"].tolist() == [70]
    assert out["adi"].iloc[0] == 1.0


def test_segment_windows_matches_scalar_adi_cv2():
    rng = np.random.default_rng(1)
    sales = rng.poisson(0.7, size=(30, 120)) * rng.integers(1, 6, size=(30, 120))
    out = segment_windows(sales, start=np.full(30, 10), end=110)
    for i in range(30):
        adi, cv2 = adi_cv2(sales[i, 10:110])
        assert out["adi"].iloc[i] == pytest.approx(adi, nan_ok=True)
        assert out["cv2"].iloc[i] == pytest.approx(cv2, nan_ok=True)
        assert out["segment"].iloc[i] == classify(adi, cv2)


def test_segment_windows_empty_window_is_no_demand():
    out = segment_windows(np.ones((1, 10), dtype=int), start=np.array([10]), end=10)
    assert out["segment"].tolist() == ["no_demand"]
    assert out["n_weeks"].tolist() == [0]
