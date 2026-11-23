"""Closed-form cost, variance and inventory identities."""

from __future__ import annotations

import numpy as np
import pytest

from execution_sim.core import ExecutionProblem
from execution_sim.impact import (
    cost_sd,
    expected_cost,
    from_bps,
    inventories,
    mean_variance,
    sum_squares,
    to_bps,
    variance,
)
from execution_sim.schedules import twap


def test_inventory_starts_at_size_and_ends_at_zero(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    path = inventories(problem, trades)
    assert path[0] == pytest.approx(problem.size)
    assert path[-1] == 0.0
    assert np.all(np.diff(path) <= 0.0)


def test_expected_cost_matches_the_hand_formula(problem: ExecutionProblem) -> None:
    trades = np.array([problem.size / 2.0, problem.size / 2.0])
    problem = problem.with_changes(n_intervals=2)
    squares = trades @ trades
    expected = (
        0.5 * problem.market.gamma * (problem.size**2 - squares)
        + problem.market.eta * squares / problem.tau
        + problem.market.half_spread * problem.size
    )
    assert expected_cost(problem, trades) == pytest.approx(expected)
    assert expected_cost(problem, trades) == pytest.approx(
        problem.market.half_spread * problem.size
        + 0.5 * problem.market.gamma * problem.size**2
        + (problem.market.eta - 0.5 * problem.market.gamma * problem.tau) * squares / problem.tau
    )


def test_immediate_and_slow_schedules_bracket_the_impact(problem: ExecutionProblem) -> None:
    fast = np.zeros(problem.n_intervals)
    fast[0] = problem.size
    slow = twap(problem)
    assert expected_cost(problem, fast) > expected_cost(problem, slow)
    assert variance(problem, fast) < variance(problem, slow)


def test_variance_is_the_sum_of_squared_inventory(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    path = inventories(problem, trades)
    manual = problem.market.sigma**2 * problem.tau * float(np.sum(path[1:] ** 2))
    assert variance(problem, trades) == pytest.approx(manual)
    assert cost_sd(problem, trades) == pytest.approx(np.sqrt(manual))


def test_variance_ignores_impact_parameters(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    heavier = problem.with_changes(market=problem.market.with_changes(eta=1e-7, gamma=1e-8))
    assert variance(heavier, trades) == pytest.approx(variance(problem, trades))


def test_drift_rewards_waiting_when_prices_rise(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    assert expected_cost(problem, trades, drift=1.0) < expected_cost(problem, trades)
    assert expected_cost(problem, trades, drift=-1.0) > expected_cost(problem, trades)


def test_mean_variance_uses_the_problem_risk_aversion(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    manual = expected_cost(problem, trades) + problem.risk_aversion * variance(problem, trades)
    assert mean_variance(problem, trades) == pytest.approx(manual)
    assert mean_variance(problem, trades, 0.0) == pytest.approx(expected_cost(problem, trades))


def test_bps_round_trip(problem: ExecutionProblem) -> None:
    assert to_bps(problem, problem.notional) == pytest.approx(1e4)
    assert from_bps(problem, 10.0) == pytest.approx(problem.notional * 1e-3)
    assert to_bps(problem, from_bps(problem, 7.5)) == pytest.approx(7.5)


def test_sum_squares(problem: ExecutionProblem) -> None:
    trades = np.array([1.0, 2.0, 3.0])
    assert sum_squares(trades) == pytest.approx(14.0)
