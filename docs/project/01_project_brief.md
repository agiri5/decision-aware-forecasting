# 01 Project Brief and Guardrails

## The project
Working title: "When does forecast accuracy actually pay? A decision-aware forecasting and inventory benchmark, plus an LLM-agent planner."

Why: forecasting work usually ends at accuracy metrics (MAPE, WRMSSE). The practical question is whether better accuracy lowers stockouts and holding cost, and for which kinds of demand.

Research questions
- RQ1: By demand segment (smooth, erratic, intermittent, lumpy), how much simulated inventory cost is saved per unit of forecast accuracy gained?
- RQ2: How do conclusions change under asymmetric, medtech-like costs (stockout 20-100x holding cost, 99%+ service targets)?
- RQ3: Can an LLM agent with tools match a classical forecast + base-stock policy in the same simulator, and where does it fail?
- Hypothesis (not a conclusion): LLM for exceptions and context, solver for the math.

## Data
- Primary: M5 (Walmart, ~3,000 products x 10 stores, ~1,900 days, prices, calendar, SNAP days).
- Generalization checks: Favorita (Ecuador grocery, promo flag) and the auto-parts dataset from Hyndman's expsmooth package (highly intermittent; closest public analog to low-volume critical consumables).
- Lead times, costs, service targets: synthetic, in one parameter file, labeled synthetic.
- Verify each dataset's license/terms before use.
- Known limits to state in the write-ups: demand is censored (zeros may be stockouts), grocery-leaning data, no real inventory/supplier data, medtech is an analog only.

## Deliverables
Public GitHub repo with reproducible tables and steady commit history; validated simulator; forecast and decision-cost results on 3 datasets; sensitivity analysis; LLM-agent study with pre-registration, stress tests, and failure taxonomy; Streamlit demo (free hosting); 2 write-ups; model/risk card.

## Guardrails
Integrity
- Public data only; no client or employer material of any kind.
- Costs and lead times are assumptions. Say so in code comments, README, demo, and write-ups.
- Call it a "reference implementation," not a production system.
- Report null results (for example, a foundation model that doesn't beat baselines).
- Don't reuse the earlier trading-floor project.
Scope
- Personal, non-commercial work only.

## Out of scope
Production claims, monetization, running the agent on all ~30,000 series, a model leaderboard, extra datasets beyond the three.