# Decision-aware forecasting

**When does forecast accuracy actually pay?** A decision-aware forecasting and inventory
benchmark, plus an LLM-agent planner evaluated in the same simulator.

This is a **reference implementation**, not a production system. It uses only public data
(M5, Favorita, Hyndman auto-parts).

> **Synthetic assumptions:** all costs, lead times, and service targets are synthetic
> assumptions defined in one parameter file. They are not observed values.

Project docs: [`docs/project/`](docs/project/).

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check . && pytest
```

## Layout

| Dir | Contents |
|---|---|
| `data/` | Download instructions/scripts only |
| `simulator/` | Weekly inventory loop, policies, tests |
| `forecasting/` | Baselines, LightGBM, foundation models, backtests |
| `analysis/` | Decision value, sensitivity, medtech analog |
| `agent/` | Prompts, tools, runner, cache, stress tests |
| `app/` | Streamlit demo |
| `docs/` | Write-ups, model/risk card, assumptions, pre-registration |
