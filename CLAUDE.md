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

## Current state (2026-10-19)
- Phase 1 / Week 1 ending. Simulator v1.0.0 built and pushed: spec `docs/simulator_spec.md`,
  `params.yaml`, `simulator/`, tests (a)-(k) passing, decision log D1-D5 in `docs/assumptions.md`.
  Draft LangGraph decision graph in `agent/` (topology not frozen until the pre-registration).
- G1 simulator review (Fable) due Oct 19: prompt in `docs/project/private/` (local only).
- Open: dataset license checks (TODOs in `data/README.md`), auto-parts dataset not downloaded,
  billing alarms/spend limits, `failure_order_qty` and over-budget tool-request rule for the
  pre-registration, whether to reword/untrack CLAUDE.md in git history.
- Next (Week 2): fix G1 findings; M5 loading, subsample and Syntetos-Boylan segmentation;
  `forecasting/bootstrap.py` (shared residual bootstrap, D1); baselines (seasonal naive,
  Croston/TSB); LightGBM.

## Conventions
- Git identity is repo-local (agiri5). Commit messages carry no tool attribution: no Co-Authored-By
  or "Generated with" trailers, no mention of AI tooling.
- Agent harness uses LangGraph (pin in `pyproject.toml` extra `agent`). No hosted tracing (LangSmith); no extra LLM calls beyond the `plan` node. See D5 in `docs/assumptions.md`.
- Simulator event order is documented and changed only with a version bump.
