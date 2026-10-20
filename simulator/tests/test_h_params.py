"""(h) Params validation: the loader rejects values that would break the simulator's contract."""

import pytest

from simulator import load_params
from simulator.params import ParamsError, params_from_overrides


def test_repo_params_load():
    p = load_params()
    assert p.lead_time.max_weeks + p.simulation.review_period_weeks <= p.forecast.max_horizon_weeks
    assert 0.0 < p.tau < 1.0


BAD = {
    "max_weeks + R > horizon": {"lead_time": {"max_weeks": 8}},
    "deterministic fixed_weeks + R > horizon": {
        "lead_time": {"distribution": "deterministic", "fixed_weeks": 8}
    },
    "review period < 1": {"simulation": {"review_period_weeks": 0}},
    "negative holding rate": {"costs": {"holding_rate_annual": -0.1}},
    "negative fixed order cost": {"costs": {"fixed_order_cost": -1.0}},
    "non-positive ratio": {"costs": {"stockout_to_holding_ratio": 0}},
    "min lead time < 1": {"lead_time": {"min_weeks": 0}},
    "max < min lead time": {"lead_time": {"min_weeks": 3, "max_weeks": 2}},
    "bad shock prob": {"lead_time": {"shock_prob": 1.5}},
    "bad mode": {"simulation": {"mode": "backlog"}},
    "bad quantile source": {"policy": {"quantile_source": "median"}},
    "unknown key": {"lead_time": {"allow_crossing": True}},
    "wrong sim_version": {"sim_version": "0.9.0"},
}


@pytest.mark.parametrize("name", list(BAD))
def test_rejects(name):
    with pytest.raises(ParamsError):
        params_from_overrides(BAD[name])


def test_horizon_check_uses_review_period():
    # max_weeks 6 + R 3 = 9 > 8
    with pytest.raises(ParamsError):
        params_from_overrides({"simulation": {"review_period_weeks": 3}})
