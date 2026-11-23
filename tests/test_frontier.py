"""The frontier and the risk-free quantification of TWAP's loss."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from execution_sim.almgren_chriss import optimal_trades
from execution_sim.core import ExecutionProblem
from execution_sim.frontier import (
    default_risk_aversions,
    frontier,
    lambda_for_cost_increase,
    point,
)
from execution_sim.impact import cost_sd, expected_cost, mean_variance, to_bps
from execution_sim.schedules import build_schedules, twap


def test_frontier_is_monotone(problem: ExecutionProblem) -> None:
    points = frontier(problem)
    expected = [item.expected_bps for item in points]
    dispersion = [item.sd_bps for item in points]
    assert all(later > earlier for earlier, later in itertools.pairwise(expected))
    assert all(later < earlier for earlier, later in itertools.pairwise(dispersion))


def test_frontier_starts_at_twap(problem: ExecutionProblem) -> None:
    start = frontier(problem)[0]
    trades = twap(problem)
    assert start.risk_aversion == 0.0
    assert start.expected_bps == pytest.approx(to_bps(problem, expected_cost(problem, trades)))
    assert start.sd_bps == pytest.approx(to_bps(problem, cost_sd(problem, trades)))
    assert start.kappa == 0.0


def test_no_frontier_point_is_dominated(problem: ExecutionProblem) -> None:
    points = frontier(problem)
    for lower, higher in itertools.pairwise(points):
        assert higher.expected_bps > lower.expected_bps
        assert higher.sd_bps < lower.sd_bps


def test_ac_dominates_every_algorithm_at_every_risk_aversion(
    problem: ExecutionProblem, profile
) -> None:
    """No other schedule can beat the optimum of its own objective."""
    schedules = build_schedules(problem, profile)
    for lam in (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        reference = mean_variance(problem, optimal_trades(problem, lam), lam)
        for name, trades in schedules.items():
            value = mean_variance(problem, trades, lam)
            assert value >= reference - 1e-9 * max(1.0, abs(reference)), (
                name,
                lam,
                value,
                reference,
            )


def test_point_reproduces_the_formula(problem: ExecutionProblem) -> None:
    item = point(problem, 1e-6)
    assert item.expected_bps == pytest.approx(to_bps(problem, item.expected_cost))
    assert item.sd_bps == pytest.approx(to_bps(problem, item.sd_cost))
    assert item.mean_variance == pytest.approx(item.expected_cost + 1e-6 * item.sd_cost**2)


def test_price_of_risk_is_positive_and_consistent(problem: ExecutionProblem) -> None:
    upgraded = lambda_for_cost_increase(problem, 1.0)
    baseline = to_bps(problem, expected_cost(problem, twap(problem)))
    assert upgraded.expected_bps == pytest.approx(baseline + 1.0, abs=0.01)
    assert upgraded.sd_bps < to_bps(problem, cost_sd(problem, twap(problem)))


def test_bigger_budget_buys_more_risk_reduction(problem: ExecutionProblem) -> None:
    small = lambda_for_cost_increase(problem, 0.5)
    large = lambda_for_cost_increase(problem, 2.0)
    assert large.sd_bps < small.sd_bps
    assert large.risk_aversion > small.risk_aversion


def test_price_of_risk_rejects_a_zero_budget(problem: ExecutionProblem) -> None:
    with pytest.raises(ValueError, match="extra_bps"):
        lambda_for_cost_increase(problem, 0.0)


def test_price_of_risk_rejects_an_unreachable_budget(problem: ExecutionProblem) -> None:
    with pytest.raises(ValueError, match="beyond the frontier"):
        lambda_for_cost_increase(problem, 10_000.0)


def test_default_grid_starts_at_zero_and_is_sorted(problem: ExecutionProblem) -> None:
    grid = default_risk_aversions(problem)
    assert grid[0] == 0.0
    assert np.all(np.diff(grid) > 0.0)


def test_zero_risk_aversion_grid_is_still_usable() -> None:
    from execution_sim.scenario import default_problem

    problem = default_problem().with_changes(risk_aversion=0.0)
    assert default_risk_aversions(problem)[0] == 0.0
    assert default_risk_aversions(problem)[1] > 0.0
