# Decision-aware forecasting: working notes for Claude Code

Reference implementation (not a production system) studying when forecast accuracy reduces
inventory cost, plus an LLM-agent planner scored in the same simulator.
Specs: `docs/project/01_project_brief.md`, `02_execution_plan.md`, `03_technical_spec.md`
(source of truth for simulator, forecasting, analysis and agent design).
Optional local-only notes (gitignored): `docs/project/private/`.

## Rules
- All costs, lead times and service targets are SYNTHETIC assumptions. Label them as such in
  code comments, README, demo and write-ups. Keep them in one params file (`params.yaml`).
- Public data only. No client/employer material. Never commit raw data (`*.csv`, `*.parquet`
  are gitignored). Raw data lives outside the repo: `DAF_RAW_DATA_DIR` (default `../Data`).
- Do not claim production use. Report null and negative results.
- Personal, non-commercial work only. Flag anything near revenue, paid users or freelancing.
- Before any paid LLM run: frozen simulator/prompts/tools, pilot, cost estimate, disk cache,
  hard spending limits, and the user's sign-off on `docs/preregistration.md`.
- Don't over-engineer; if behind schedule apply the cut order in file 02.
- Be a critical reviewer: push back on weak methodology and overclaims.

## Setup / checks
```bash
source .venv/bin/activate   # Python 3.12 via uv
ruff check . && ruff format --check . && pytest
```

## Current state (2026-10-22)
- Phase 1 / Week 2 started. Simulator v1.1.0 built and pushed: spec `docs/simulator_spec.md`,
  `params.yaml`, `simulator/`, tests (a)-(k) passing, decision log D1-D11 in `docs/assumptions.md`.
  Draft LangGraph decision graph in `agent/` (topology not frozen until the pre-registration).
- G1 simulator review done and triaged (D6-D11): planning now uses the pooled lead-time
  estimator (N = 2,000 core, 50,000 medtech arm, `medtech_analog.n_paths`); lead-time streams
  split so schedules are prefix-stable in window length. Documented limitations: critical-ratio
  bias under lost sales (D6), the agent seeing arrival weeks of in-transit orders (D7).
- Open: dataset license checks (TODOs in `data/README.md`), auto-parts dataset not downloaded,
  billing alarms/spend limits, `failure_order_qty` and over-budget tool-request rule for the
  pre-registration, whether the agent sees arrival weeks (D7) and censored sales vs demand
  (spec section 11), whether to reword/untrack CLAUDE.md in git history, untested fixed order
  cost K and unit_price scaling.
- Next (Week 2): M5 loading, subsample and Syntetos-Boylan segmentation;
  `forecasting/bootstrap.py` (shared residual bootstrap, D1) with two tests written first: a
  provider-leakage test (paths for origin t identical on the full series and on the series
  truncated at t-1, same rng) and a check that it draws only from the `rng` it is given; then
  baselines (seasonal naive, Croston/TSB); LightGBM. Re-time planning cost with the real bootstrap.

## Conventions
- Git identity is repo-local (agiri5). Commit messages carry no tool attribution: no Co-Authored-By
  or "Generated with" trailers, no mention of AI tooling.
- Agent harness uses LangGraph (pin in `pyproject.toml` extra `agent`). No hosted tracing (LangSmith); no extra LLM calls beyond the `plan` node. See D5 in `docs/assumptions.md`.
- Simulator event order is documented and changed only with a version bump.
