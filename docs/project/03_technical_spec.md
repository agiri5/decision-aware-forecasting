# 03 Technical Spec

## A. Simulator (plain Python/NumPy, weekly)
Event order each week t (document it; change only with a version bump):
1. Receive orders arriving at t. 2. Observe state (on-hand, in-transit, history). 3. Policy places order q_t (arrives at t+L, L sampled per order). 4. Demand d_t is realized. 5. Serve min(on-hand, d_t); unmet demand is lost sales. 6. Record holding cost on ending on-hand, stockout cost on lost units, and order cost if any.
Parameters (params.yaml, labeled SYNTHETIC): holding cost h per unit-week, stockout cost p per lost unit, lead-time distribution (lognormal + occasional delay shock), review period R, service target.
Policies: base-stock with order-up-to S from a forecast quantile at critical ratio p/(p+h) over (L+R); (s,S); LLM agent (Phase 5).
Metrics: fill rate, stockout frequency, average inventory, total cost, cost per series.
Required tests:
(a) newsvendor single-period case matches the closed-form optimal quantile and cost
(b) zero-variance demand gives zero stockouts and minimal cost
(c) unit conservation: the inventory identity holds every week
(d) seeded reproducibility
(e) base-stock results cross-checked against Stockpyl or a closed form
(f) monotonicity: raising p never lowers the optimal stock level
Optional libraries (check maintenance and license first): SimPy (overkill here), OR-Gym (for reference), Stockpyl (validation).

## B. Data and segmentation
- M5: subsample 3,000-5,000 item-store series stratified by category and store. Aggregate daily to weekly. State the censored-demand limitation.
- Segment with Syntetos-Boylan thresholds (ADI 1.32, CV2 0.49) into smooth, erratic, intermittent, lumpy. Document cutoffs.
- Favorita: small stratified subsample. Auto-parts: all series.

## C. Forecasting
Models: seasonal naive; Croston/TSB; global LightGBM with quantile objectives; Chronos or TimesFM zero-shot (GPU spot instance, stop when idle, billing alarm).
Backtest: rolling origin, multiple origins, horizons 1-8 weeks, identical folds for all models.
Metrics: M5 WRMSSE (where applicable), pinball loss at the relevant quantiles, MASE, bias; by segment, bootstrap CIs over series.
Cache all forecasts to parquet so the simulator stage is reproducible.

## D. Core analysis
- Per model x segment: simulated total cost and fill rate.
- Cost saved per point of accuracy gained: paired comparison versus baseline per series, bootstrap CIs; avoid causal over-precision.
- Medtech analog: p/h in {20, 50, 100}, service target 99%+, on auto-parts data. Call it an analog only.
- Sensitivity sweep: h, p/h ratio, lead-time mean and variance, service target. Report which conclusions survive.
- Always state: synthetic costs, censored demand, grocery-leaning data, one simulator design.

## E. Agent study
Interface: weekly state in (on-hand, in-transit, last N weeks of demand, promo/price flag, lead-time info, optional note); JSON out {order_qty, rationale}. Strict schema validation; invalid output counts as a failure; one retry max.
Tools (cap 3 calls per decision): forecast_lookup, safety_stock_calc, simulate_whatif.
Conditions: C0 classical (best forecast + base-stock, the benchmark); C1 LLM no tools; C2 LLM with tools; C3 LLM with tools + guardrails (order bounds, sanity checks, escalation flag).
Design: 48 series (16 each smooth/intermittent/lumpy) x 52 weeks x 3 LLM conditions x 3 seeds on a small cheap model (Haiku-class) via Bedrock/API. Strong-model subset: 12 series, best condition, 3 seeds. Temperature fixed and recorded.
Cost control:
- Pilot first: 5 series x 10 weeks, measure tokens/decision, extrapolate.
- Disk cache keyed by hash(prompt + state + model + params).
- Prompt caching on static system prompt and tool definitions.
- Hard spending limits; freeze simulator/prompts/tools before the main run; log every rerun.
- Add seeds later only if results are noisy; cut series count before baselines or stress tests.
- Decisions ~ series x weeks x conditions x seeds (~22,000 in the base design); check provider pricing yourself.

### Pre-registration (write before any paid run; commit with timestamp to docs/preregistration.md)
- Primary metric: paired total cost relative to C0 per series, with bootstrap CI.
- Secondary: fill rate, invalid-output rate, tokens and dollars per decision, run-to-run variance.
- Hypotheses stated up front (for example: C1 is worse than C0 on intermittent series; C3 reduces worst-case failures).
- Failure taxonomy: arithmetic error; anchoring on mean demand; ignoring lead time; over-ordering after a stockout; under-ordering under high stockout cost; schema/format failure; prompt-injection compliance.
- Stress tests: S1 demand shock; S2 supplier delay; S3 stale or conflicting note; S4 prompt injection hidden in a "supplier email"; S5 asymmetric medtech costs; S6 promotion spike.
- Analysis rules: no dropping runs without logging why; report all conditions, including those that lose.
- Limitation to state: one model family (Claude) unless a non-Anthropic model is added for generalization.

## F. Demo and docs
Streamlit on a free host (Community Cloud or Hugging Face Spaces): user sets costs and service target, sees which model wins by segment, with a visible "synthetic assumptions" banner. README with one-command reproduction of main tables. Model/risk card covering assumptions, limits, and failure modes. Every result table reproducible from a script with fixed seeds.