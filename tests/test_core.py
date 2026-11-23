"""The model objects and the validation that protects the rest of the package."""

from __future__ import annotations

import numpy as np
import pytest

from execution_sim.almgren_chriss import optimal_trades
from execution_sim.core import ExecutionProblem, MarketModel, check_trades, effective_eta


def test_half_spread_and_notional(problem: ExecutionProblem) -> None:
    assert problem.market.half_spread == pytest.approx(0.005)
    assert problem.notional == pytest.approx(5_000_000.0)
    assert problem.tau == pytest.approx(0.05)
    assert problem.times[0] == 0.0
    assert problem.times[-1] == pytest.approx(problem.horizon)
    assert problem.times.size == problem.n_intervals + 1


def test_effective_eta_contains_the_discrete_correction() -> None:
    market = MarketModel(gamma=2e-7, eta=4e-8)
    assert effective_eta(market, 0.05) == pytest.approx(4e-8 - 0.5 * 2e-7 * 0.05)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price", 0.0),
        ("price", -1.0),
        ("price", float("nan")),
        ("gamma", 0.0),
        ("eta", -1e-8),
        ("eta", float("inf")),
        ("sigma", -0.5),
        ("sigma", float("nan")),
        ("half_spread_bps", -1.0),
    ],
)
def test_invalid_market_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        MarketModel(**{field: value})


def test_zero_volatility_is_allowed() -> None:
    assert MarketModel(sigma=0.0).sigma == 0.0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("size", 0.0, ValueError),
        ("size", -5.0, ValueError),
        ("horizon", 0.0, ValueError),
        ("n_intervals", 0, ValueError),
        ("n_intervals", 2.0, TypeError),
        ("n_intervals", True, TypeError),
        ("risk_aversion", -1e-9, ValueError),
    ],
)
def test_invalid_problem_parameters(field: str, value: float, error: type[Exception]) -> None:
    with pytest.raises(error):
        ExecutionProblem(**{field: value})


def test_non_convex_model_is_rejected() -> None:
    market = MarketModel(gamma=1e-5, eta=1e-8)
    problem = ExecutionProblem(market=market, n_intervals=5)
    assert effective_eta(market, problem.tau) < 0.0
    with pytest.raises(ValueError, match="not convex"):
        optimal_trades(problem)


def test_check_trades_accepts_a_valid_schedule(problem: ExecutionProblem) -> None:
    check_trades(problem, np.full(problem.n_intervals, problem.size / problem.n_intervals))


def test_check_trades_rejects_bad_input(problem: ExecutionProblem) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        check_trades(problem, np.ones((2, 10)))
    with pytest.raises(ValueError, match="empty"):
        check_trades(problem, np.empty(0))
    with pytest.raises(ValueError, match="non-negative"):
        check_trades(problem, np.array([problem.size + 1.0, -1.0]))
    with pytest.raises(ValueError, match="sum to size"):
        check_trades(problem, np.full(problem.n_intervals, 1.0))
    bad = np.full(problem.n_intervals, problem.size / problem.n_intervals)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        check_trades(problem, bad)


def test_with_changes_keeps_the_rest(problem: ExecutionProblem) -> None:
    changed = problem.with_changes(size=1_000.0)
    assert changed.size == 1_000.0
    assert changed.horizon == problem.horizon
    market = problem.market.with_changes(eta=1e-7)
    assert market.eta == 1e-7
    assert market.gamma == problem.market.gamma
