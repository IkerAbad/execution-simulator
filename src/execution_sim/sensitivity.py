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
from execution_sim.book import LevelBook, LiquidityModel
from execution_sim.core import ExecutionProblem, MarketModel
from execution_sim.engine import book_execution, expected_execution_cost
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


@dataclass(frozen=True, slots=True)
class RefillRow:
    """One (refill, algorithm) cell of the running-book experiment."""

    refill: float
    algorithm: str
    expected_bps: float
    objective_bps: float
    beyond_shares: float
    beyond_pct: float
    max_child_pct: float
    depth_min_pct: float
    exhausted_intervals: int


def refill_scenarios(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    book: LevelBook,
    *,
    refills: FloatArray,
    risk_aversion: float | None = None,
) -> list[RefillRow]:
    """Cost of every algorithm when the book is only partially replenished.

    ``refill = 1`` is the static book of the reference experiment.  As the refill
    falls the book is depleted by the schedule itself, so the same trade list costs
    more and, eventually, part of it has to be executed beyond the visible depth.
    The variance does not change: the schedule is the same and the book is
    deterministic.
    """
    lam = problem.risk_aversion if risk_aversion is None else risk_aversion
    schedules = build_schedules(problem, profile)
    rows: list[RefillRow] = []
    for refill in np.asarray(refills, dtype=float):
        liquidity = LiquidityModel(refill=float(refill))
        for name, trades in schedules.items():
            execution = book_execution(problem, trades, book, liquidity)
            expected = expected_execution_cost(problem, trades, book=book, liquidity=liquidity)
            variance = cost_sd(problem, trades) ** 2
            rows.append(
                RefillRow(
                    refill=float(refill),
                    algorithm=name,
                    expected_bps=to_bps(problem, expected),
                    objective_bps=to_bps(problem, expected + lam * variance),
                    beyond_shares=execution.beyond_total,
                    beyond_pct=100.0 * execution.beyond_total / problem.size,
                    max_child_pct=100.0 * execution.max_child_fraction,
                    depth_min_pct=float(100.0 * execution.depth_after.min() / book.capacity)
                    if execution.depth_after.size
                    else 100.0,
                    exhausted_intervals=execution.exhausted_intervals,
                )
            )
    return rows
