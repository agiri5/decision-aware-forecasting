"""State shown to a policy at the Observe step (spec section 3, step 2 and section 6.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class State:
    """Read-only snapshot at week t, after receipts, before ordering and before d_t.

    History arrays are copies covering weeks 0..t-1 only (len == t); the policy never sees d_t.
    `seed` and `series_id` are carried so policies can derive their planning RNG stream
    (spec section 5.1) without being constructed per run; they are not informative about demand.
    """

    t: int
    on_hand: int  # OH_t^avail (net inventory in backorder mode; may be negative)
    pipeline: tuple[tuple[int, int], ...]  # (arrival_week, qty), sorted by arrival
    inventory_position: int
    sales_history: np.ndarray  # s_0..s_{t-1} (censored in lost-sales mode)
    demand_history: np.ndarray  # d_0..d_{t-1}
    lead_time_pmf: dict[int, float]  # effective lead-time pmf (spec section 4)
    seed: int
    series_id: str
