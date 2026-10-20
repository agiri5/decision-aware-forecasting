"""Parameter loading and validation for the weekly inventory simulator.

All costs, lead times and service targets loaded here are SYNTHETIC assumptions defined in
`params.yaml` (see docs/simulator_spec.md, section 7). They are not observed values.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SIM_VERSION = "1.0.0"

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[1] / "params.yaml"

_MODES = ("lost_sales", "backorder")
_LEAD_DISTRIBUTIONS = ("lognormal_shock", "deterministic")
_QUANTILE_SOURCES = ("critical_ratio", "service_target")


class ParamsError(ValueError):
    """Raised when params.yaml (or an override dict) violates the spec's constraints."""


@dataclass(frozen=True)
class SimulationParams:
    mode: str
    review_period_weeks: int
    warmup_weeks: int
    eval_weeks: int
    initial_on_hand: int | str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class LeadTimeParams:
    distribution: str
    fixed_weeks: int
    mean_weeks: float
    cv: float
    min_weeks: int
    max_weeks: int
    shock_prob: float
    shock_weeks: int
    pmf_estimation_weeks: int
    pmf_estimation_seed: int


@dataclass(frozen=True)
class CostParams:
    holding_rate_annual: float
    default_unit_price: float
    stockout_to_holding_ratio: float
    fixed_order_cost: float


@dataclass(frozen=True)
class PolicyParams:
    name: str
    quantile_source: str
    n_paths: int


@dataclass(frozen=True)
class ServiceParams:
    target: float


@dataclass(frozen=True)
class ForecastParams:
    max_horizon_weeks: int
    bootstrap_min_origins: int


@dataclass(frozen=True)
class SimParams:
    sim_version: str
    simulation: SimulationParams
    lead_time: LeadTimeParams
    costs: CostParams
    policy: PolicyParams
    service: ServiceParams
    forecast: ForecastParams

    # -- derived economics (SYNTHETIC) -------------------------------------------------------
    def holding_cost(self, unit_price: float | None = None) -> float:
        """h: holding cost per unit-week = annual rate / 52 * unit price."""
        price = self.costs.default_unit_price if unit_price is None else unit_price
        return self.costs.holding_rate_annual / 52.0 * price

    def stockout_cost(self, unit_price: float | None = None) -> float:
        """p: stockout cost per lost unit = ratio * h."""
        return self.costs.stockout_to_holding_ratio * self.holding_cost(unit_price)

    @property
    def tau(self) -> float:
        """Order-up-to quantile: p/(p+h) = ratio/(ratio+1) (core) or the service target."""
        if self.policy.quantile_source == "service_target":
            return self.service.target
        ratio = self.costs.stockout_to_holding_ratio
        return ratio / (ratio + 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def params_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimParams:
        try:
            params = cls(
                sim_version=str(raw["sim_version"]),
                simulation=_build_simulation(raw["simulation"]),
                lead_time=LeadTimeParams(**_pick(raw["lead_time"], LeadTimeParams)),
                costs=CostParams(**_pick(raw["costs"], CostParams)),
                policy=PolicyParams(**_pick(raw["policy"], PolicyParams)),
                service=ServiceParams(**_pick(raw["service"], ServiceParams)),
                forecast=ForecastParams(**_pick(raw["forecast"], ForecastParams)),
            )
        except (KeyError, TypeError) as exc:
            raise ParamsError(f"malformed params: {exc!r}") from exc
        _validate(params)
        return params


def _pick(section: dict[str, Any], cls: type) -> dict[str, Any]:
    """Select exactly the dataclass fields; extra keys (e.g. comments-as-keys) are an error."""
    names = set(cls.__dataclass_fields__)
    extra = set(section) - names
    if extra:
        raise ParamsError(f"unknown keys for {cls.__name__}: {sorted(extra)}")
    return {k: section[k] for k in names if k in section}


def _build_simulation(section: dict[str, Any]) -> SimulationParams:
    picked = _pick(section, SimulationParams)
    picked["seeds"] = tuple(int(s) for s in picked["seeds"])
    return SimulationParams(**picked)


def _validate(p: SimParams) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ParamsError(msg)

    need(p.sim_version == SIM_VERSION, f"sim_version {p.sim_version!r} != code {SIM_VERSION!r}")

    s, lt, c = p.simulation, p.lead_time, p.costs
    need(s.mode in _MODES, f"simulation.mode must be one of {_MODES}")
    need(s.review_period_weeks >= 1, "review_period_weeks must be >= 1")
    need(s.warmup_weeks >= 0 and s.eval_weeks >= 1, "warmup_weeks >= 0 and eval_weeks >= 1")
    need(
        s.initial_on_hand == "steady_state"
        or (isinstance(s.initial_on_hand, int) and s.initial_on_hand >= 0),
        "initial_on_hand must be 'steady_state' or a non-negative integer",
    )

    need(lt.distribution in _LEAD_DISTRIBUTIONS, f"lead_time.distribution in {_LEAD_DISTRIBUTIONS}")
    need(lt.min_weeks >= 1, "lead_time.min_weeks must be >= 1 (orders never arrive same week)")
    need(lt.max_weeks >= lt.min_weeks, "lead_time.max_weeks must be >= min_weeks")
    need(lt.mean_weeks > 0 and lt.cv >= 0, "lead_time.mean_weeks > 0 and cv >= 0")
    need(0.0 <= lt.shock_prob <= 1.0, "lead_time.shock_prob must be in [0, 1]")
    need(lt.shock_weeks >= 0, "lead_time.shock_weeks must be >= 0")
    need(lt.pmf_estimation_weeks > 200, "lead_time.pmf_estimation_weeks must exceed burn-in (100)")

    horizon = p.forecast.max_horizon_weeks
    need(horizon >= 1, "forecast.max_horizon_weeks must be >= 1")
    need(
        lt.max_weeks + s.review_period_weeks <= horizon,
        f"lead_time.max_weeks + review_period_weeks must be <= forecast horizon ({horizon})",
    )
    if lt.distribution == "deterministic":
        need(lt.fixed_weeks >= 1, "lead_time.fixed_weeks must be >= 1")
        need(
            lt.fixed_weeks + s.review_period_weeks <= horizon,
            f"lead_time.fixed_weeks + review_period_weeks must be <= forecast horizon ({horizon})",
        )

    need(c.holding_rate_annual >= 0, "costs.holding_rate_annual must be >= 0")
    need(c.default_unit_price > 0, "costs.default_unit_price must be > 0")
    need(c.stockout_to_holding_ratio > 0, "costs.stockout_to_holding_ratio must be > 0")
    need(c.fixed_order_cost >= 0, "costs.fixed_order_cost must be >= 0")

    need(p.policy.quantile_source in _QUANTILE_SOURCES, f"quantile_source in {_QUANTILE_SOURCES}")
    need(p.policy.n_paths >= 1, "policy.n_paths must be >= 1")
    need(0.0 < p.service.target < 1.0, "service.target must be in (0, 1)")
    need(p.forecast.bootstrap_min_origins >= 1, "forecast.bootstrap_min_origins must be >= 1")


def load_raw(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> SimParams:
    """Load and validate params.yaml (all values SYNTHETIC)."""
    raw = load_raw(path)
    return SimParams.from_dict(_core_sections(raw))


def with_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `overrides` ({section: {key: value}}) onto a copy of `raw`."""
    out = copy.deepcopy(raw)
    for section, values in overrides.items():
        if isinstance(values, dict):
            out.setdefault(section, {}).update(values)
        else:
            out[section] = values
    return out


def params_from_overrides(
    overrides: dict[str, Any] | None = None, path: str | Path = DEFAULT_PARAMS_PATH
) -> SimParams:
    raw = with_overrides(load_raw(path), overrides or {})
    return SimParams.from_dict(_core_sections(raw))


_CORE = ("sim_version", "simulation", "lead_time", "costs", "policy", "service", "forecast")


def _core_sections(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop analysis-only sections (sensitivity grids, optional arms, medtech analog)."""
    return {k: raw[k] for k in _CORE if k in raw}
