"""Monte Carlo against the closed form: this is the numerical verification of the model."""

from __future__ import annotations

import numpy as np
import pytest

from execution_sim.book import LevelBook
from execution_sim.core import ExecutionProblem
from execution_sim.engine import (
    execution_slippages,
    expected_execution_cost,
    simulate,
)
from execution_sim.impact import cost_sd, expected_cost, variance
from execution_sim.schedules import build_schedules, twap


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_simulated_mean_and_sd_match_the_analytic_moments(
    problem: ExecutionProblem, profile, seed: int
) -> None:
    schedules = build_schedules(problem, profile)
    for name, trades in schedules.items():
        result = simulate(problem, trades, n_paths=40_000, seed=seed)
        analytic_mean = expected_cost(problem, trades)
        analytic_sd = cost_sd(problem, trades)
        assert abs(result.mean - analytic_mean) <= 5.0 * result.stderr, name
        if analytic_sd > 0.0:
            assert result.sd == pytest.approx(analytic_sd, rel=0.01), name
        else:
            assert result.sd == pytest.approx(0.0, abs=1e-9)


def test_deterministic_part_matches_the_linear_model(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    assert expected_execution_cost(problem, trades) == pytest.approx(expected_cost(problem, trades))


def test_book_changes_the_mean_but_not_the_dispersion(
    problem: ExecutionProblem, book: LevelBook
) -> None:
    # TWAP trades exactly the chunk size the book is calibrated at, so the two
    # execution models must agree to the cent for that schedule.
    twap_trades = twap(problem)
    linear = simulate(problem, twap_trades, n_paths=20_000, seed=4)
    ladder = simulate(problem, twap_trades, book=book, n_paths=20_000, seed=4)
    assert ladder.mean == pytest.approx(linear.mean, rel=1e-12)
    assert ladder.sd == pytest.approx(linear.sd, rel=0.01)
    assert ladder.sd == pytest.approx(cost_sd(problem, twap_trades), rel=0.01)

    # Any other shape pays a different temporary cost.
    uneven = np.full(problem.n_intervals, problem.size / problem.n_intervals)
    uneven[0] *= 3.0
    uneven[1:] *= (problem.size - uneven[0]) / uneven[1:].sum()
    assert expected_execution_cost(problem, uneven, book=book) != pytest.approx(
        expected_execution_cost(problem, uneven)
    )


def test_book_is_more_expensive_than_the_linear_model_for_big_blocks(
    problem: ExecutionProblem, book: LevelBook
) -> None:
    block = np.zeros(problem.n_intervals)
    block[0] = problem.size
    assert expected_execution_cost(problem, block, book=book) > expected_cost(problem, block) * 1.05


def test_simulation_is_reproducible(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    first = simulate(problem, trades, n_paths=5_000, seed=11)
    second = simulate(problem, trades, n_paths=5_000, seed=11)
    other = simulate(problem, trades, n_paths=5_000, seed=12)
    assert np.array_equal(first.costs, second.costs)
    assert not np.array_equal(first.costs, other.costs)
    assert first.stderr > 0.0
    assert first.n_paths == 5_000


def test_drift_enters_exactly_as_expected(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    drift = 0.5
    result = simulate(problem, trades, n_paths=40_000, seed=5, drift=drift)
    analytic = expected_cost(problem, trades, drift=drift)
    assert abs(result.mean - analytic) <= 5.0 * result.stderr
    assert expected_execution_cost(problem, trades, drift=drift) == pytest.approx(analytic)


def test_zero_impact_and_zero_volatility_gives_the_spread_only(
    problem: ExecutionProblem,
) -> None:
    flat = problem.with_changes(
        market=problem.market.with_changes(sigma=0.0, gamma=1e-15, eta=1e-15)
    )
    trades = twap(flat)
    result = simulate(flat, trades, n_paths=1_000, seed=0)
    assert result.sd == pytest.approx(0.0)
    assert result.mean == pytest.approx(flat.market.half_spread * flat.size, rel=1e-6)
    assert expected_cost(flat, trades) == pytest.approx(
        flat.market.half_spread * flat.size, rel=1e-6
    )


def test_execution_slippages_include_the_spread(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    linear = execution_slippages(problem, trades)
    assert linear[0] == pytest.approx(
        problem.market.half_spread + problem.market.eta * trades[0] / problem.tau
    )


def test_invalid_path_count(problem: ExecutionProblem) -> None:
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="n_paths"):
            simulate(problem, twap(problem), n_paths=bad)  # type: ignore[arg-type]


def test_variance_is_untouched_by_the_execution_model(
    problem: ExecutionProblem, book: LevelBook
) -> None:
    trades = twap(problem)
    assert variance(problem, trades) == pytest.approx(cost_sd(problem, trades) ** 2)
    assert expected_execution_cost(problem, trades, book=book) > 0.0
