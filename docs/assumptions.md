# Assumptions and Design-Decision Log

All costs, lead times and service targets in this project are **SYNTHETIC** assumptions. They
live in `params.yaml`; values and rationale are in `docs/simulator_spec.md`, section 7.

This file logs decisions and every deviation from `docs/project/03_technical_spec.md`, with the
date and reason. Entries are append-only; a reversal is a new entry.

| ID | Date | Decision | Deviation from file 03? | Reason |
|---|---|---|---|---|
| D1 | 2026-10-12 | Base-stock orders at τ = p/(p+h). Service level is reported only. Service target is used as τ only in a separate sensitivity arm. Protection-interval demand over L+R is computed from joint sample paths, never by summing per-horizon quantiles. All point forecasters (and quantile-only models) get distributions from one shared residual bootstrap of whole 8-week error vectors. | Clarification: file 03 lists both a critical ratio and a service target. LightGBM quantile heads are used for accuracy metrics; decisions use the shared bootstrap on its point/median forecast. | A critical ratio and a service target cannot both set the same quantile. The quantile of a sum is not the sum of quantiles. A single uncertainty method keeps forecaster comparisons about the point forecast. |
| D2 | 2026-10-12 | Fixed order cost K = 0. (s,S) removed from core scope; kept as an optional K > 0 arm, not built in v1. | **Yes**: file 03 lists (s,S) as a policy. | With K = 0, (s,S) is identical to base-stock, and K = 0 isolates the value of the forecast. |
| D3 | 2026-10-12 | Base-stock planning draws the protection interval by Monte Carlo over the effective lead-time distribution (lognormal + shock, after no-overtaking). A deterministic-lead-time config is kept for tests (a), (b), (e). | No | Planning on a single lead time would turn the lead-time sweep into a test of policy misspecification. |
| D4 | 2026-10-12 | No order overtaking: A_t = max(t + L_t, A_{t−1}). Lead times are pre-drawn per week (common random numbers across policies). The effective lead-time pmf is estimated by simulation and used in planning. Test (i) checks arrivals never decrease. | Refinement: file 03 says "L sampled per order". | Single sequential supplier. Pre-drawn schedules make per-series paired comparisons measure the policy, not lead-time luck. |
| D5 | 2026-10-13 | The agent harness is a LangGraph graph (draft topology v0 in `agent/graph.py`, `langgraph==1.2.15` as the optional `agent` extra). Conditions C1-C3 are configurations of one graph. Provider client, cache and spend limits stay behind a thin interface. Traces are local; the runner refuses to start if hosted tracing env vars are set. | Refinement of file 03 section E; adds a frozen-topology item to the pre-registration. | Learning goal plus a graph that matches the specified loop. Keeps cost, cache and mocks independent of the framework. Fallback: plain Python loop. |
