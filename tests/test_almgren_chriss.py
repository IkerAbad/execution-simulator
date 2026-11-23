"""The acceptance tests: the closed form must be the optimum of the discrete objective.

Four independent checks:

1. it matches the numerical solution of the first-order conditions (Thomas algorithm);
2. it satisfies the Euler-Lagrange recurrence point by point;
3. no feasible perturbation of it reduces the objective;
4. an independent coordinate-descent solver converges to the same path.

None of these uses the closed-form formula on both sides except the first one, where
the two solutions are computed by completely different routes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from execution_sim._typing import FloatArray
from execution_sim.almgren_chriss import (
    first_interval_fraction,
    half_life,
    inventory_path,
    kappa,
    numeric_optimal_trades,
    optimal_trades,
)
from execution_sim.core import ExecutionProblem, effective_eta
from execution_sim.impact import inventories, mean_variance
from execution_sim.schedules import twap


@pytest.mark.parametrize("risk_aversion", [0.0, 1e-8, 1e-7, 1e-6, 3e-6])
@pytest.mark.parametrize("n_intervals", [1, 2, 5, 20, 50])
def test_closed_form_equals_the_numerical_solution(
    problem: ExecutionProblem, risk_aversion: float, n_intervals: int
) -> None:
    # eta is raised so that the discrete objective stays convex even for N = 1.
    market = problem.market.with_changes(eta=1e-6)
    scenario = problem.with_changes(
        market=market, n_intervals=n_intervals, risk_aversion=risk_aversion
    )
    closed = optimal_trades(scenario)
    numerical = numeric_optimal_trades(scenario)
    assert closed.sum() == pytest.approx(scenario.size, rel=1e-12)
    assert np.max(np.abs(closed - numerical)) <= 1e-9 * scenario.size


@pytest.mark.parametrize("risk_aversion", [1e-9, 1e-7, 1e-5])
def test_euler_lagrange_recurrence(problem: ExecutionProblem, risk_aversion: float) -> None:
    path = inventory_path(problem, risk_aversion)
    eta_eff = effective_eta(problem.market, problem.tau)
    c = risk_aversion * problem.market.sigma**2 * problem.tau**2 / eta_eff
    residual = 2.0 * path[1:-1] - path[:-2] - path[2:] + c * path[1:-1]
    assert np.max(np.abs(residual)) <= 1e-9 * problem.size


def _coordinate_descent(
    problem: ExecutionProblem, risk_aversion: float, sweeps: int = 300
) -> FloatArray:
    """Independent minimiser: exact one-dimensional steps on the quadratic objective."""
    path = twap(problem).copy()
    path = np.concatenate(([problem.size], problem.size - np.cumsum(path)))
    eta_eff = effective_eta(problem.market, problem.tau)
    c = risk_aversion * problem.market.sigma**2 * problem.tau**2 / eta_eff
    2.0 + c
    for _ in range(sweeps):
        for k in range(1, problem.n_intervals):
            path[k] = 0.5 * (path[k - 1] + path[k + 1]) / (1.0 + 0.5 * c)
    trades = -np.diff(path)
    return np.maximum(trades, 0.0)


def test_independent_coordinate_descent_agrees(problem: ExecutionProblem) -> None:
    scenario = problem.with_changes(n_intervals=10, risk_aversion=1e-6)
    closed = optimal_trades(scenario)
    independent = _coordinate_descent(scenario, scenario.risk_aversion)
    assert np.max(np.abs(closed - independent)) <= 1e-7 * scenario.size


@pytest.mark.parametrize("risk_aversion", [1e-8, 1e-6])
def test_no_feasible_perturbation_improves_the_objective(
    problem: ExecutionProblem, risk_aversion: float
) -> None:
    rng = np.random.default_rng(2026)
    scenario = problem.with_changes(risk_aversion=risk_aversion)
    trades = optimal_trades(scenario)
    base = mean_variance(scenario, trades)
    for _ in range(50):
        perturbation = rng.normal(size=trades.size)
        perturbation -= perturbation.mean()
        perturbation *= 0.01 * scenario.size / np.max(np.abs(perturbation))
        for direction in (1.0, -1.0):
            candidate = trades + direction * perturbation
            if np.any(candidate < 0.0):
                continue
            assert mean_variance(scenario, candidate) >= base - 1e-9 * abs(base)


def test_risk_neutral_optimum_is_twap(problem: ExecutionProblem) -> None:
    trades = optimal_trades(problem, 0.0)
    assert np.allclose(trades, twap(problem), rtol=1e-12)
    assert kappa(problem, 0.0) == 0.0
    assert np.allclose(inventory_path(problem, 0.0), problem.size * (1.0 - problem.times))


def test_kappa_increases_with_risk_aversion_and_volatility(problem: ExecutionProblem) -> None:
    values = [kappa(problem, lam) for lam in (1e-8, 1e-7, 1e-6, 1e-5)]
    assert all(later > earlier for earlier, later in itertools.pairwise(values))
    calm = problem.with_changes(market=problem.market.with_changes(sigma=0.5))
    assert kappa(calm) < kappa(problem)


def test_optimal_schedule_front_loads_as_risk_aversion_grows(problem: ExecutionProblem) -> None:
    fractions = [
        first_interval_fraction(optimal_trades(problem, lam), problem.size)
        for lam in (0.0, 1e-7, 1e-6, 1e-5, 1e-4)
    ]
    assert all(later >= earlier for earlier, later in itertools.pairwise(fractions))
    assert fractions[-1] > 0.5


def test_optimal_schedule_beats_twap_in_its_own_objective(problem: ExecutionProblem) -> None:
    trades = optimal_trades(problem)
    assert mean_variance(problem, trades) < mean_variance(problem, twap(problem))


def test_half_life(problem: ExecutionProblem) -> None:
    assert half_life(problem, twap(problem)) == pytest.approx(problem.horizon / 2.0)
    fast = half_life(problem, optimal_trades(problem, 1e-6))
    slow = half_life(problem, optimal_trades(problem, 1e-9))
    assert fast < slow


def test_single_interval_is_immediate(problem: ExecutionProblem) -> None:
    scenario = problem.with_changes(n_intervals=1, market=problem.market.with_changes(eta=1e-6))
    assert optimal_trades(scenario).tolist() == [scenario.size]
    assert numeric_optimal_trades(scenario).tolist() == [scenario.size]


def test_negative_risk_aversion_is_rejected(problem: ExecutionProblem) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        kappa(problem, -1.0)


def test_inventory_path_shape(problem: ExecutionProblem) -> None:
    path = inventory_path(problem, 1e-6)
    assert path.size == problem.n_intervals + 1
    assert path[0] == problem.size
    assert path[-1] == 0.0
    assert np.all(np.diff(path) <= 0.0)
    assert np.allclose(inventories(problem, optimal_trades(problem, 1e-6)), path)
