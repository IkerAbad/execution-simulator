"""Sensitivity of cost and risk to the model parameters.

Every scenario rebuilds the execution problem with one parameter changed and, for
the Almgren-Chriss row, re-optimises the schedule.  That matters: comparing a fixed
schedule across parameters would conflate a change of model with a change of
optimal policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from execution_sim._typing import FloatArray
from execution_sim.core import ExecutionProblem, MarketModel
from execution_sim.engine import expected_execution_cost
from execution_sim.frontier import FrontierPoint, frontier
from execution_sim.impact import cost_sd, expected_cost, to_bps
from execution_sim.schedules import build_schedules
from execution_sim.volume import VolumeProfile


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    """One (scenario, algorithm) cell of a sensitivity table."""

    scenario: str
    algorithm: str
    expected_bps: float
    sd_bps: float


@dataclass(frozen=True, slots=True)
class FrontierCurve:
    """A frontier drawn for one value of a parameter."""

    label: str
    parameter: float
    points: list[FrontierPoint]


def _with_market(problem: ExecutionProblem, **changes: float) -> ExecutionProblem:
    market = replace(problem.market, **changes)
    return replace(problem, market=market)


def impact_scenarios(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    *,
    eta_multipliers: FloatArray,
    gamma_multipliers: FloatArray,
) -> list[ScenarioRow]:
    """Cost and risk of every algorithm under scaled temporary and permanent impact."""
    rows: list[ScenarioRow] = []
    base = problem.market
    for eta_multiplier in np.asarray(eta_multipliers, dtype=float):
        for gamma_multiplier in np.asarray(gamma_multipliers, dtype=float):
            market = MarketModel(
                price=base.price,
                sigma=base.sigma,
                gamma=base.gamma * float(gamma_multiplier),
                eta=base.eta * float(eta_multiplier),
                half_spread_bps=base.half_spread_bps,
            )
            scenario = problem.with_changes(market=market)
            label = f"eta x{eta_multiplier:g} / gamma x{gamma_multiplier:g}"
            for name, trades in build_schedules(scenario, profile).items():
                rows.append(
                    ScenarioRow(
                        scenario=label,
                        algorithm=name,
                        expected_bps=to_bps(scenario, expected_cost(scenario, trades)),
                        sd_bps=to_bps(scenario, cost_sd(scenario, trades)),
                    )
                )
    return rows


def volatility_curves(
    problem: ExecutionProblem,
    *,
    multipliers: FloatArray,
    n_points: int = 25,
) -> list[FrontierCurve]:
    """The whole cost-risk frontier for several volatility levels."""
    curves: list[FrontierCurve] = []
    for multiplier in np.asarray(multipliers, dtype=float):
        scenario = _with_market(problem, sigma=problem.market.sigma * float(multiplier))
        grid = np.logspace(-2.0, 2.0, n_points) * max(problem.risk_aversion, 1e-12)
        points = frontier(scenario, np.concatenate(([0.0], grid)))
        curves.append(
            FrontierCurve(
                label=f"sigma x{multiplier:g}",
                parameter=float(multiplier),
                points=points,
            )
        )
    return curves


def drift_scenarios(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    *,
    drifts_bps_per_day: FloatArray,
) -> list[ScenarioRow]:
    """Expected cost of every algorithm under a trending mid price.

    The Almgren-Chriss schedule in this package is derived for a driftless price, so
    with a drift it is a plug-in rule rather than the exact optimum.  The point of
    the table is to show how much the ranking moves when the assumption fails.
    """
    rows: list[ScenarioRow] = []
    for bps in np.asarray(drifts_bps_per_day, dtype=float):
        drift = problem.market.price * float(bps) * 1e-4
        for name, trades in build_schedules(problem, profile).items():
            expected = expected_execution_cost(problem, trades, drift=drift)
            rows.append(
                ScenarioRow(
                    scenario=f"drift {bps:+.1f} bp/day",
                    algorithm=name,
                    expected_bps=to_bps(problem, expected),
                    sd_bps=to_bps(problem, cost_sd(problem, trades)),
                )
            )
    return rows
