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

## Current state (2026-10-12)
- Phase 0 / Week 1. Scaffold done and pushed (github.com/agiri5/decision-aware-forecasting).
- Open: dataset license checks (TODOs in `data/README.md`), auto-parts dataset not downloaded,
  billing alarms/spend limits.
- Next: simulator spec (event order, cost accounting, `params.yaml`), then build the weekly
  simulator and required tests (a)-(f) from file 03. G1 simulator review due by Oct 19.

## Conventions
- Git identity is repo-local (agiri5). Commit messages end with the Co-Authored-By line for
  the model used.
- Simulator event order is documented and changed only with a version bump.
