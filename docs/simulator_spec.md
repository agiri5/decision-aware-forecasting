# Simulator Spec (v1.0.0, draft for G1)

Weekly, single-item, single-location, periodic-review inventory simulator with lost sales.
Plain Python/NumPy. This document is the contract for `simulator/`; the code must match it, and
any change to the event order or cost accounting bumps `SIM_VERSION`.

> **All costs, lead times and service targets are SYNTHETIC assumptions** (see section 7 and
> `params.yaml`). They are not observed values for any company or product.

Reference implementation only. One simulator design; conclusions are conditional on it.

---

## 1. Scope

In scope (v1): one series at a time; integer units; weekly time steps; lost sales (with a
backorder mode used **only** for validation); stochastic lead times without overtaking; periodic
review every R weeks; a base-stock policy; a `Policy` interface the LLM agent will implement later.
(s,S) is an optional arm, outside the core scope (see `docs/assumptions.md`, D2).

Out of scope (v1): multi-echelon, shared capacity across series, perishability, purchase-price
effects, demand that responds to stockouts (substitution), feeding simulator stockouts back into
forecaster training.

## 2. Notation

| Symbol | Meaning | Units |
|---|---|---|
| t | week index, `0..T-1` (simulation window, includes warm-up) | week |
| d_t | demand in week t (historical weekly sales, treated as demand) | units |
| OH_t^start | on-hand at start of week t, **before** receipts | units |
| r_t | units received at start of week t | units |
| OH_t^avail | on-hand after receipts, available to serve demand | units |
| q_t | order quantity placed in week t (0 on non-review weeks) | units |
| A_t | arrival week of an order placed in week t | week |
| s_t | units sold (served) in week t | units |
| ℓ_t | lost units in week t (lost-sales mode) | units |
| OH_t^end | on-hand at end of week t | units |
| P_t | pipeline: set of (arrival week, qty) not yet received | units |
| IP_t | inventory position = OH_t^avail + Σ pipeline qty (+ net backlog sign in backorder mode) | units |
| h | holding cost per unit per week, charged on OH^end | currency/unit-week |
| p | stockout cost per lost unit (lost-sales) or per backlogged unit-week (backorder mode) | currency/unit |
| K | fixed cost per order placed (q_t > 0) | currency/order |
| R | review period | weeks |
| L_t | sampled lead time for week t's order slot | weeks |
| τ | target quantile for the order-up-to level | - |

## 3. Event order within week t (normative)

1. **Receive.** `r_t = Σ qty for (a, qty) in P where a == t`; remove them from P.
   `OH_t^avail = OH_t^start + r_t`.
2. **Observe.** The policy receives a read-only `State` (section 6) containing only information
   available at this point: on-hand after receipts, the pipeline, and demand/sales history
   **through week t-1**. It never sees d_t.
3. **Order.** If `t % R == 0`, the policy returns `q_t ≥ 0` (integer). Otherwise `q_t = 0`. If
   `q_t > 0`, append `(A_t, q_t)` to P. Orders are never received in the week they are placed
   (`A_t ≥ t + 1`).
4. **Demand.** d_t is realized.
5. **Serve.** Lost-sales mode: `s_t = min(OH_t^avail, d_t)`, `ℓ_t = d_t - s_t`,
   `OH_t^end = OH_t^avail - s_t`.
   Backorder mode (validation only): net inventory `NI_t^end = NI_t^avail - d_t` (can be negative;
   receipts in step 1 clear backlog first, implicitly via NI arithmetic).
6. **Cost.** `holding_t = h · max(OH_t^end, 0)`;
   `stockout_t = p · ℓ_t` (lost-sales) or `p · max(-NI_t^end, 0)` (backorder);
   `order_t = K · 1[q_t > 0]`. `OH_{t+1}^start = OH_t^end`.

Consequences (these are the conventions that are easy to get wrong):
- In-transit units incur **no** holding cost.
- An order placed in week t can first affect **ending** inventory in week A_t. With deterministic
  lead time L and R = 1, ending on-hand in week t+L equals `IP_t(after order) − Σ_{k=t}^{t+L} d_k`
  in backorder mode, i.e. the protection interval is **L + R** weeks of demand starting with d_t.
  This is the identity test (b) exercises exactly.
- Units received and sold within the same week incur no holding cost.

## 4. Lead times (common random numbers)

Lead times are **pre-drawn per week, not per order**, so every policy run with the same
(series, seed) faces the identical supplier: the paired per-series comparison then measures the
policy/forecast, not lead-time luck.

For each week t in the window, draw independently:
- `X_t ~ LogNormal(μ, σ)` with `σ² = ln(1 + cv²)`, `μ = ln(mean) − σ²/2` (mean, cv in weeks).
- `shock_t ~ Bernoulli(shock_prob)`; if 1, add `shock_weeks`.
- `L_t = clip(round(X_t) + shock_t · shock_weeks, min_weeks, max_weeks)`.

Arrival week, no overtaking (single sequential supplier):
`A_t = max(t + L_t, A_{t-1})` (with `A_{-1} = −∞`), computed for **every** week whether or not an
order is placed, so the schedule is policy-independent. A_t is non-decreasing by construction,
and orders are received in the order they were placed. There is no option to allow crossing.

Because of rounding, clipping, shocks and the cumulative max, the **effective** lead time
`E_t = A_t − t` does not have the nominal distribution. It is serially correlated, and its
marginal pmf is shifted upward relative to L_t. **Effective pmf:** generate one long sequence
(`lead_time.pmf_estimation_weeks`, default 200,000 weeks, fixed `pmf_estimation_seed`) with the
same generator and cumulative max, discard the first 100 weeks, and take the empirical marginal
pmf of E_t. Since E_t ≤ max_weeks, the support is within `[min_weeks, max_weeks]`. The pmf
depends only on the lead-time params, is cached per params hash, and is what policies plan with
(section 5.1). Every run logs the realized mean/max of E_t.

Known approximation: planning draws E from the marginal pmf independently per sample path. It
ignores serial correlation of E_t, which the cumulative max introduces. Stated, not corrected.

**Deterministic config** (tests (a), (b), (e)): `lead_time.distribution: deterministic` with
`fixed_weeks: L`. Then `L_t = L`, `A_t = t + L`, and the effective pmf is a point mass at L.

Constraint: `max_weeks + R ≤ forecast max horizon (8)`. The params loader rejects violations.
This applies to the deterministic config too (`fixed_weeks + R ≤ 8`).

Orders still in transit when the window ends are reported as ending pipeline; they incur no cost.

Stress hook (S2 supplier delay, Phase 6): an optional `delay_events` list of
`(start_week, end_week, extra_weeks)` that shifts A_t for orders in flight during the event. Not
implemented in v1; its addition is a version bump.

## 5. Policies

### 5.1 Base-stock (order-up-to S), Monte Carlo protection-interval demand
At review week t:
1. Get `N = policy.n_paths` (default 2,000) demand sample paths `Y ∈ ℝ^{N×H}`, H = 8, for weeks
   t..t+H−1, using data through t−1 (section 6.2).
2. For each path n, draw an effective lead time `E_n` from the effective pmf (section 4), so the
   protection interval is `k_n = E_n + R`.
3. Protection-interval demand `X_n = Σ_{j=0}^{k_n−1} Y[n, j]`.
4. `S_t = ceil(empirical τ-quantile of {X_n})` (numpy `method="inverted_cdf"`, i.e. the smallest
   sample value with empirical CDF ≥ τ).
5. `q_t = max(0, S_t − IP_t)`.

τ is set by `policy.quantile_source`:
- `critical_ratio` (**core**): `τ = p / (p + h)`. Service level is reported only, never enforced.
- `service_target` (**sensitivity arm only**): `τ = service.target`, read as a cycle-service
  quantile. Achieved fill rate is reported next to it; CSL and fill rate are different measures,
  and the write-up must not conflate them.

Planning randomness: path generation and E_n draws use a **planning RNG** seeded from
`SeedSequence([seed, crc32(series_id), t, 1])`. This is separate from the lead-time RNG, so CRN of
the supply schedule (section 4) is unaffected. S_t is reproducible, and the same (seed, series, t)
gives the same paths for every τ (needed for test (f)(i)). Each forecaster gets identical E_n
draws at the same (seed, series, t).

Monte Carlo error: with N = 2,000 and τ = 0.99, about 20 paths lie above the quantile. Test (j)
(section 9) bounds the MC error. Raise N for the medtech arm if that test shows S_t moving by
more than 1 unit between planning seeds.

Known simplification, to be stated in the write-ups: under lost sales this policy is a
heuristic, not the optimum. It uses the backorder critical ratio and ignores serial correlation
of lead times. Huh et al. (2009) show base-stock is asymptotically optimal for lost sales as p/h
grows, which supports using it as a benchmark. It is the **same heuristic for every forecaster**,
so forecaster comparisons are fair. Absolute service levels are conditional on it.

### 5.2 (s,S): optional arm, not in core scope
Removed from core scope (K = 0 makes it identical to base-stock; `docs/assumptions.md`, D2).
If run later in a K > 0 arm: `s_t` = base-stock level from 5.1;
`S_t = s_t + Q`, `Q = max(1, round(sqrt(2·K·μ̂_t / h)))`, where `μ̂_t` is the mean weekly demand
over the sample paths. Order `S_t − IP_t` if `IP_t ≤ s_t`, else 0. Not built in v1.

### 5.3 Test-only policies
`FixedOrderUpTo(S)` (constant S, no forecast), `RandomOrders(rng)` (for the conservation test),
`ZeroPolicy`.

## 6. Interfaces

### 6.1 Simulator
```python
SIM_VERSION = "1.0.0"

def simulate(
    demand: np.ndarray,           # int, shape (T,), weekly demand for the window
    policy: Policy,
    params: SimParams,            # parsed from params.yaml, validated
    seed: int,
    series_id: str,
    initial_on_hand: int | None = None,  # None -> policy's S at t=0 (steady-state start)
) -> SimResult: ...
```
`SimResult` holds per-week arrays (`on_hand_start, received, on_hand_avail, ordered,
arrival_week, demand, sold, lost, on_hand_end, pipeline_end, holding_cost, stockout_cost,
order_cost`), the lead-time draws, and metadata (`sim_version`, `params_hash`, `seed`,
`series_id`, `mode`, `tau`, `quantile_source`, `n_paths`, effective-pmf hash, per-week `S_t`).

RNG streams (never Python's `hash()`, which is salted per process):
- lead times: `SeedSequence([seed, zlib.crc32(series_id.encode()), 0])`
- planning (policy side): `SeedSequence([seed, crc32(series_id), t, 1])`
- residual bootstrap (forecast side): `SeedSequence([seed, crc32(series_id), t, 2])`

The simulator's own randomness is lead times only. Demand is historical (or supplied by tests).

### 6.2 Policy and forecast provider
```python
@dataclass(frozen=True)
class State:
    t: int
    on_hand: int                  # after receipts (OH_t^avail)
    pipeline: tuple[tuple[int, int], ...]  # (arrival_week, qty)
    inventory_position: int
    sales_history: np.ndarray     # s_0..s_{t-1} from this simulation (censored)
    demand_history: np.ndarray    # d_0..d_{t-1} (copy, not a view of the full array)
    lead_time_pmf: dict[int, float]

class Policy(Protocol):
    def order(self, state: State) -> int: ...

class ForecastProvider(Protocol):
    # Joint demand sample paths for weeks origin..origin+H-1, using data through origin-1.
    # Shape (n_paths, H), non-negative floats. Joint = each row is one coherent future.
    def sample_paths(
        self, series_id: str, origin: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray: ...
```
Forecasts are computed **offline** and cached (file 03, C). The simulator does not retrain
models, and simulated stockouts do not feed back into forecaster inputs.

**Rule: protection-interval demand is always computed by summing joint sample paths (section
5.1). It is never computed by summing per-horizon quantiles,** because the quantile of a sum is
not the sum of quantiles. The interface deliberately exposes no per-horizon quantile method.

How each model produces paths:
- **Point forecasters** (seasonal naive, Croston/TSB, and any model used through a point
  forecast) go through **one shared residual-bootstrap method** (`forecasting/bootstrap.py`),
  identical for every such model:
  - Residual vectors: from rolling-origin backtests on the training window only, for each origin
    o, store the 8-vector `e_o = (y_{o..o+7} − ŷ_{o..o+7|o−1})`. A whole vector is resampled so
    the correlation across horizons is kept.
  - Path n: `Y[n] = max(0, ŷ_{t..t+7|t−1} + e_{o_n})`, with `o_n` drawn uniformly from the stored
    origins. Additive errors with clipping at 0; clipping raises the path mean on intermittent
    series, which is stated.
  - Residuals come from the series itself. If it has fewer than `min_origins` (default 26) usable
    origins, they are pooled within its demand segment. The fallback is logged.
- **Native sample-path models** (e.g. Chronos) use their own samples.
- Models whose decision inputs are quantiles only (e.g. LightGBM quantile heads, TimesFM
  quantiles) cannot produce joint paths. They go through the shared residual bootstrap on their
  median/point forecast. Their quantile heads are still used for the pinball-loss accuracy tables.
- Robustness check: also run native-path models through the shared bootstrap, so a cost gap can
  be attributed to the point forecast rather than the uncertainty method.

Leakage rules: a forecast for origin t uses data through week t−1 only. Bootstrap residuals come
only from origins whose targets lie inside the training window, never the evaluation window.

## 7. Cost accounting and synthetic parameters

All values below are **SYNTHETIC** and live only in `params.yaml`.

| Parameter | Base value | Rationale | Sweep (analysis D) |
|---|---|---|---|
| holding rate | 25%/year of unit price → `h = 0.25/52 · price` | textbook carrying-cost range | 15%, 40% |
| unit price | median M5 sell price over the training window, per series; 1.0 where no prices (auto-parts) | sets cost scale only | - |
| p/h ratio | 9 (τ = 0.90) | generic retail | 1, 4, 9, 19, 49, 99 |
| medtech analog p/h | 20, 50, 100 (τ = 0.952, 0.980, 0.990) | brief RQ2 | - |
| fixed order cost K | 0 | isolates forecast value | none in core; optional (s,S) arm only |
| review period R | 1 week | weekly planning | 1, 2 |
| lead time mean / cv | 2 weeks / 0.3 | short replenishment | mean 1, 2, 4; cv 0, 0.3, 0.6 |
| delay shock | prob 0.05, +2 weeks | occasional late shipment | prob 0, 0.05, 0.15 |
| lead time bounds | 1..6 weeks | max + R ≤ 8 | - |
| service target | 0.95 (0.99 in medtech analog) | reported against in core; used as τ only in the `service_target` arm | `service_target` arm |
| planning paths N | 2,000 | MC quantile of protection-interval demand | raise if test (j) fails |
| warm-up | 8 weeks, excluded from metrics | flush initial conditions | - |
| evaluation window | 52 weeks | one seasonal cycle | - |
| seeds | 0, 1, 2 | lead-time draws only | - |

Notes:
- Policies depend only on p/h, R and the lead-time distribution. h sets the cost **scale**, which
  matters only when costs are aggregated across series. The analysis should use per-series
  relative cost (vs. the baseline forecaster, same seed). Pooled currency totals are dominated
  by high-volume series.
- No unit purchase cost: lost margin is folded into p. Stated as a simplification.

## 8. Metrics (evaluation window only, warm-up excluded)

| Metric | Definition | Edge case |
|---|---|---|
| total cost | Σ (holding + stockout + order) | - |
| cost components | each sum separately | - |
| fill rate | Σ s_t / Σ d_t | NaN if Σ d_t = 0; series reported, excluded from fill-rate averages, count logged |
| stockout frequency | #weeks with ℓ_t > 0 / #weeks with d_t > 0 | NaN if no demand weeks |
| cycle service level | 1 − stockout frequency | same |
| avg inventory | mean OH_t^end | - |
| orders placed | #weeks with q_t > 0 | - |
| realized lead time | mean/max of A_t − t over placed orders | - |

## 9. Required tests (file 03, a-f) and what each proves

| Test | Setup | Assertion | Proves | Does **not** prove |
|---|---|---|---|---|
| (a) newsvendor | T = 1, no ordering, `initial_on_hand = S`, discrete demand pmf (Poisson(10), uniform{0..20}). Run once per demand value and weight by the pmf (exact, no Monte Carlo) | argmin_S expected cost = smallest S with F(S) ≥ p/(p+h); min cost = closed form `h·E(S−D)+ + p·E(D−S)+` to 1e-9 | single-period cost accounting and serve logic | anything about lead times or dynamics |
| (b) zero variance | constant d, deterministic-lead-time config L ∈ {1,2,4}, R ∈ {1,2}, oracle provider returning identical constant paths (so the MC quantile is exact), start at steady state | at S* = d·(L+R): zero lost sales and zero holding cost after warm-up; at S*+1: zero lost, holding = h every week (both modes); at S*−1: backorder mode backlogs exactly 1 unit every week, lost-sales mode loses exactly 1 unit every L+R weeks (the lost unit is not re-ordered against; verified by a throwaway script for R = 1 only, derive R = 2 counts when writing the test) | protection-interval timing (no off-by-one in receive/order/serve order) | stochastic behaviour |
| (c) unit conservation | random demand, stochastic L (incl. shocks), `RandomOrders`, both modes | every week: `OH^avail = OH^start + r`, `OH^end = OH^avail − s`, `s + ℓ = d`, `pipeline_end = pipeline_start + q − r`; cumulative: `OH_0 + Σq = OH_T^end + pipeline_T + Σs` | no units created/destroyed | that costs are priced correctly |
| (d) reproducibility | same (seed, series_id) twice; different seeds; two different policies with same seed | identical arrays; lead-time draws differ across seeds; **identical A_t schedule across policies** (CRN) | determinism and fair pairing | statistical quality of RNG |
| (e) base-stock cross-check | backorder mode, deterministic-lead-time config, R = 1, iid Poisson(λ), FixedOrderUpTo(S), T = 200,000 | long-run cost/week within 3 MC std errors of `h·E(S−D_{L+1})+ + p·E(D_{L+1}−S)+`, `D_{L+1} ~ Poisson((L+1)λ)`; optimal S matches critical-ratio quantile. Optional: same against Stockpyl (MIT; check before adding) | multi-period pipeline, IP, timing against theory | lost-sales optimality (no closed form). Lost-sales mode is covered only by (a)-(d) and (f) |
| (f) monotonicity | (i) base-stock S_t over p ∈ {1..100}·h at fixed (seed, series, t), so paths and E_n draws are identical across p; (ii) lost-sales sim, grid of FixedOrderUpTo(S), CRN, iid demand | (i) S_t non-decreasing in p; (ii) lost units non-increasing in S, and the largest argmin_S cost non-decreasing in p | policy/cost respond in the right direction | that the levels are optimal |

Additional cheap tests (recommended; each guards a downstream-invalidating bug):
- (g) **no look-ahead**: a policy that records `State` asserts `len(demand_history) == t` and
  that mutating it does not change the simulation.
- (h) **params validation**: loader rejects `max_weeks + R > 8` (and `fixed_weeks + R > 8`),
  negative costs, R < 1.
- (i) **arrivals never decrease** (required, decision D4): for stochastic configs with shocks on,
  over many seeds, `A_t` is non-decreasing in t, `A_t ≥ t + 1`, and receipts are FIFO (the
  pipeline is always sorted by arrival week, and no order is received before an earlier-placed one).
- (j) **planning MC**: (1) the effective pmf sums to 1, its support is within
  `[min_weeks, max_weeks]`, and its mean is ≥ the mean of the clipped L_t; (2) with the
  deterministic config, the pmf is a point mass; (3) for τ = 0.99 and a Poisson path provider,
  S_t varies by at most 1 unit across 5 planning seeds at the default N.
- (k) **no summed quantiles**: with iid Poisson(10) paths, deterministic L = 2, R = 1, S_t
  matches the exact quantile of Poisson(30) within 1 unit (37 at τ = 0.9, 43 at τ = 0.99). It
  must be well below the sum of per-week quantiles (42 and 54). This guards against regression to the forbidden method.

## 10. Known limitations (carry into README and write-ups)
- Historical sales are censored demand. The simulator treats them as true demand, so simulated
  lost sales are relative to observed sales, not true demand.
- Simulated stockouts do not change future demand or forecaster inputs.
- Base-stock is a heuristic under lost sales and stochastic lead time (5.1). Planning ignores
  serial correlation of effective lead times.
- Uncertainty for point forecasters comes from one shared residual bootstrap. A different
  uncertainty method could change cost rankings; the robustness check in 6.2 tests part of this.
- All cost and lead-time parameters are synthetic. Only relative comparisons are meaningful.
- Single item, single location, no capacity limits.

## 11. Decisions
Resolved 2026-10-12 (logged in `docs/assumptions.md`, D1-D4):
1. Order at τ = p/(p+h); report service level only; service target is a separate sensitivity
   arm. Protection-interval demand comes from joint sample paths, never summed quantiles. All
   point forecasters use one shared residual bootstrap.
2. K = 0. (s,S) is removed from core scope and kept as an optional arm.
3. Planning uses Monte Carlo over the params.yaml lead-time distribution (effective pmf).
   A deterministic-lead-time config is kept for tests.
4. No overtaking: `A_t = max(t + L_t, A_{t−1})`. The effective pmf is estimated by simulation
   and used in planning. Test (i) checks arrivals never decrease.

Still open:
5. **What the agent sees** (for G3, not v1): censored sales history vs. demand history. `State`
   carries both so the choice stays open.
