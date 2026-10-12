# 02 Execution Plan (10 weeks, start Mon Oct 12, 2026)

## Calendar
| Wk | Dates | Focus | Gate / deadline |
|---|---|---|---|
| 1 | Oct 12-18 | Phase 0 setup + simulator build | G1 simulator review (by Oct 19) |
| 2 | Oct 19-25 | Simulator fixes + validation, M5 segmentation, baselines, LightGBM | |
| 3 | Oct 26-Nov 1 | Agent design, prompts, tools, pre-registration, stress scenarios | G3 pre-registration review by Nov 1 |
| 4 | Nov 2-8 | Mocked-LLM agent runner + tests; Bedrock cost test; foundation-model forecasts | Mocked runner ready (Nov 5) |
| 5 | Nov 9-15 | Finalize backtests; forecasts into simulator; core finding; medtech analog; Favorita | |
| 6 | Nov 16-22 | Sensitivity sweeps; Streamlit demo; write-up #1; freeze and tag release | G2 claim review, G4 write-up #1 review |
| 7 | Nov 23-29 | Agent pilot (5 series x 10 weeks), cost extrapolation, start main run (short holiday week) | Pilot go/no-go |
| 8 | Nov 30-Dec 6 | Main agent run finishes; strong-model subset | Spend check |
| 9 | Dec 7-13 | Stress tests, failure taxonomy, design rule | |
| 10 | Dec 14-20 | Write-up #2, README, model/risk card, reproducibility check | G5 final review |

## Phase 0: Setup (Week 1)
1. Verify LLM API budget and eligibility via a tiny test. Set billing alarms and hard spending limits.
2. Check dataset licenses.
3. Create the GitHub repo; commit from day one; set up environment, lint, and test scaffolding.
Exit: credit terms known, licenses checked, repo live.

## Phase 1: Data and simulator (Weeks 1-2)
Load and subsample M5, segment demand, build the simulator and two policies, write tests.
Exit: simulator passes all required tests (file 03), assumptions in one params file, G1 passed.

## Phase 2: Forecasting (Weeks 2-4)
Seasonal naive, Croston/TSB, LightGBM quantile, Chronos/TimesFM zero-shot, rolling-origin backtests, cached forecasts.
Exit: forecast tables by model x segment, backtest design documented, null results reported.

## Phase 3: Core finding (Week 5)
Forecast -> simulator -> cost by segment; cost-per-accuracy analysis; medtech analog on auto-parts data; Favorita replication.
Exit: one chart and one table answering RQ1/RQ2, replicated or documented failures to replicate.

## Phase 4: Robustness and publication (Week 6)
Sensitivity sweeps, Streamlit demo, write-up #1 (forecast and decision value), freeze simulator/params/metrics, tag release.
Exit: release tagged, write-up published, demo deployed.

## Phase 5: Agent (design Wk 3-4, runs Wk 7-8)
Design and pre-register before Nov 1; mocked-LLM runner by Nov 5; real pilot then main run on Bedrock/API.
Exit: results with CIs, spend within budget, rerun log.

## Phase 6: Stress tests (Week 9)
Run S1-S6 (file 03), categorize failures, derive a design rule only if the data supports it.
Exit: failure taxonomy with counts; best and worst decision examples.

## Phase 7: Documentation (Week 10)
Write-up #2 (agent vs. classical policy, limitations stated plainly), README with one-command reproduction, model/risk card.
Exit: a stranger can rerun the main tables from the README.

## Cut order if behind by more than one week
1. Fewer agent series (48 -> 30). 2. Fewer seeds (3 -> 2). 3. Drop Favorita replication. 4. Drop strong-model subset.
Never cut: simulator validation, baselines, sensitivity sweeps, stress tests, pre-registration.

## Overload risk
Week 3 carries agent design and pre-registration. If it slips, protect the Nov 1 pre-registration and push foundation-model forecasts to Weeks 4-5 (already scheduled for Week 4).

## Repo structure
/data (download scripts only, no raw data committed)
/simulator (loop, policies, tests/)
/forecasting (baselines, LightGBM, foundation model, backtests)
/analysis (decision-value, sensitivity, medtech analog)
/agent (prompts, tools, runner, cache, stress tests)
/app (Streamlit)
/docs (write-ups, model/risk card, assumptions.md, preregistration.md)