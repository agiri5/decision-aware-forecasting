"""Weekly single-item inventory simulator. Implements docs/simulator_spec.md v1.0.0.

Event order each week t (normative, changed only with a SIM_VERSION bump):
  1. Receive orders arriving at t.
  2. Observe state (history through t-1 only).
  3. Policy orders q_t on review weeks (t % R == 0); arrives at A_t >= t + 1.
  4. Demand d_t is realized.
  5. Serve: lost-sales -> s = min(OH, d), lost = d - s; backorder (validation only) -> NI -= d.
  6. Cost: holding on ending on-hand, stockout on lost units (or backlog unit-weeks), order cost.

All costs and lead times are SYNTHETIC assumptions from params.yaml.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from simulator.leadtime import effective_pmf, lead_time_schedule, pmf_hash
from simulator.params import SIM_VERSION, SimParams
from simulator.policies import Policy
from simulator.state import State


@dataclass
class SimResult:
    """Per-week arrays (length T) plus run metadata.

    Backorder mode (validation only): on_hand_* are net inventory (may be negative), sold == demand,
    lost == 0, and `backlog_end` holds max(-net inventory, 0).
    `arrival_week` is the supplier schedule A_t for every week, whether or not an order was placed.
    `order_up_to` is the policy's S_t on review weeks (NaN elsewhere / for policies without S).
    `pipeline_end[t]` is units in transit after week t's receipts and order.
    """

    on_hand_start: np.ndarray
    received: np.ndarray
    on_hand_avail: np.ndarray
    ordered: np.ndarray
    arrival_week: np.ndarray
    lead_draws: np.ndarray
    demand: np.ndarray
    sold: np.ndarray
    lost: np.ndarray
    on_hand_end: np.ndarray
    backlog_end: np.ndarray
    pipeline_end: np.ndarray
    holding_cost: np.ndarray
    stockout_cost: np.ndarray
    order_cost: np.ndarray
    order_up_to: np.ndarray
    initial_on_hand: int
    h: float
    p: float
    sim_version: str
    params_hash: str
    seed: int
    series_id: str
    mode: str
    tau: float
    quantile_source: str
    n_paths: int
    pmf_hash: str
    warmup_weeks: int


def _validate_demand(demand: np.ndarray) -> np.ndarray:
    arr = np.asarray(demand)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("demand must be a non-empty 1-D array")
    if np.any(arr < 0) or not np.all(arr == np.round(arr)):
        raise ValueError("demand must be non-negative integers")
    return arr.astype(np.int64)


def simulate(
    demand: np.ndarray,
    policy: Policy,
    params: SimParams,
    seed: int,
    series_id: str,
    initial_on_hand: int | None = None,
    unit_price: float | None = None,
) -> SimResult:
    """Run one series for len(demand) weeks. See module docstring for the event order.

    `initial_on_hand=None` falls back to params.simulation.initial_on_hand; 'steady_state' asks
    the policy for `initial_level(state0)` (start with that much stock, empty pipeline).
    `unit_price` (optional, per-series) scales h and p; it does not change base-stock decisions.
    """
    demand = _validate_demand(demand)
    horizon_weeks = len(demand)
    review = params.simulation.review_period_weeks
    backorder = params.simulation.mode == "backorder"
    h = params.holding_cost(unit_price)
    p = params.stockout_cost(unit_price)
    k_fixed = params.costs.fixed_order_cost

    lead_draws, arrivals = lead_time_schedule(params.lead_time, horizon_weeks, seed, series_id)
    pmf = effective_pmf(params.lead_time)

    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset()

    def make_state(t: int, on_hand: int, pipe: tuple[tuple[int, int], ...], sales: np.ndarray):
        return State(
            t=t,
            on_hand=on_hand,
            pipeline=pipe,
            inventory_position=on_hand + sum(q for _, q in pipe),
            sales_history=sales[:t].copy(),
            demand_history=demand[:t].copy(),
            lead_time_pmf=pmf,
            seed=seed,
            series_id=series_id,
        )

    sold = np.zeros(horizon_weeks, dtype=np.int64)
    if initial_on_hand is None:
        initial_on_hand = params.simulation.initial_on_hand
    if initial_on_hand == "steady_state":
        hook = getattr(policy, "initial_level", None)
        if hook is None:
            raise ValueError(
                f"{type(policy).__name__} has no initial_level(); pass initial_on_hand explicitly"
            )
        initial_on_hand = int(hook(make_state(0, 0, (), sold)))
    initial_on_hand = int(initial_on_hand)
    if initial_on_hand < 0:
        raise ValueError("initial_on_hand must be >= 0")

    zeros = lambda: np.zeros(horizon_weeks, dtype=np.int64)  # noqa: E731
    on_hand_start, received, on_hand_avail = zeros(), zeros(), zeros()
    ordered, lost, on_hand_end, backlog_end, pipeline_end = (
        zeros(),
        zeros(),
        zeros(),
        zeros(),
        zeros(),
    )
    holding_cost = np.zeros(horizon_weeks)
    stockout_cost = np.zeros(horizon_weeks)
    order_cost = np.zeros(horizon_weeks)

    pipeline: deque[tuple[int, int]] = deque()
    in_transit = 0
    on_hand = initial_on_hand

    for t in range(horizon_weeks):
        on_hand_start[t] = on_hand

        # 1. Receive. Arrivals are non-decreasing (no overtaking), so receipts are FIFO.
        r = 0
        while pipeline and pipeline[0][0] == t:
            r += pipeline.popleft()[1]
        if pipeline and pipeline[0][0] < t:
            raise AssertionError("order missed its arrival week")  # cannot happen: A_t >= t + 1
        received[t] = r
        in_transit -= r
        avail = on_hand + r
        on_hand_avail[t] = avail

        # 2-3. Observe and order (review weeks only).
        q = 0
        if t % review == 0:
            q_raw = policy.order(make_state(t, avail, tuple(pipeline), sold))
            q = int(q_raw)
            if q != q_raw or q < 0:
                raise ValueError(f"policy returned invalid order quantity {q_raw!r} at t={t}")
        if q > 0:
            pipeline.append((int(arrivals[t]), q))
            in_transit += q
        ordered[t] = q
        pipeline_end[t] = in_transit

        # 4-5. Demand realized, served.
        d = int(demand[t])
        if backorder:
            sold[t] = d
            on_hand = avail - d
            backlog_end[t] = max(-on_hand, 0)
            stockout_cost[t] = p * backlog_end[t]
        else:
            sold[t] = min(avail, d)
            lost[t] = d - sold[t]
            on_hand = avail - int(sold[t])
            stockout_cost[t] = p * lost[t]
        on_hand_end[t] = on_hand

        # 6. Cost.
        holding_cost[t] = h * max(on_hand, 0)
        order_cost[t] = k_fixed if q > 0 else 0.0

    levels = getattr(policy, "levels", None)
    order_up_to = np.full(horizon_weeks, np.nan)
    if isinstance(levels, dict):
        for t, s_t in levels.items():
            if 0 <= t < horizon_weeks:
                order_up_to[t] = s_t

    return SimResult(
        on_hand_start=on_hand_start,
        received=received,
        on_hand_avail=on_hand_avail,
        ordered=ordered,
        arrival_week=arrivals,
        lead_draws=lead_draws,
        demand=demand,
        sold=sold,
        lost=lost,
        on_hand_end=on_hand_end,
        backlog_end=backlog_end,
        pipeline_end=pipeline_end,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        order_cost=order_cost,
        order_up_to=order_up_to,
        initial_on_hand=initial_on_hand,
        h=h,
        p=p,
        sim_version=SIM_VERSION,
        params_hash=params.params_hash,
        seed=seed,
        series_id=series_id,
        mode=params.simulation.mode,
        tau=params.tau,
        quantile_source=params.policy.quantile_source,
        n_paths=params.policy.n_paths,
        pmf_hash=pmf_hash(pmf),
        warmup_weeks=params.simulation.warmup_weeks,
    )
