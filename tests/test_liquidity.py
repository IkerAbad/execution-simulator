"""The running book: replenishment, depletion and the limits of slicing."""

from __future__ import annotations

import numpy as np
import pytest

from execution_sim.book import (
    BookState,
    InsufficientDepthError,
    LevelBook,
    LiquidityModel,
)
from execution_sim.core import ExecutionProblem
from execution_sim.engine import book_execution, expected_execution_cost, simulate
from execution_sim.impact import cost_sd, expected_cost, to_bps
from execution_sim.schedules import build_schedules, immediate, twap
from execution_sim.sensitivity import refill_scenarios
from execution_sim.volume import VolumeProfile


def _liquidity(problem: ExecutionProblem, refill: float) -> LiquidityModel:
    return LiquidityModel(refill=refill)


def test_refill_one_is_exactly_the_static_book(problem: ExecutionProblem, book: LevelBook) -> None:
    """The old behaviour must be a special case, not a lookalike."""
    schedules = build_schedules(problem, VolumeProfile())
    for trades in schedules.values():
        dynamic = book_execution(problem, trades, book, LiquidityModel(refill=1.0))
        assert np.allclose(dynamic.slippages, book.slippages(trades), rtol=0.0, atol=1e-12)
        assert dynamic.beyond_total == 0.0


def test_a_static_ladder_has_no_memory(problem: ExecutionProblem, book: LevelBook) -> None:
    """With no replenishment the temporary cost cannot depend on the slicing."""
    schedules = build_schedules(problem, VolumeProfile())
    zero = LiquidityModel(refill=0.0)
    temporary = {
        name: float(np.asarray(trades) @ book_execution(problem, trades, book, zero).slippages)
        for name, trades in schedules.items()
    }
    values = np.array(list(temporary.values()))
    assert np.ptp(values) < 1e-9 * problem.notional * 1e-4


def test_depletion_makes_any_schedule_more_expensive(
    problem: ExecutionProblem, book: LevelBook
) -> None:
    trades = build_schedules(problem, VolumeProfile())["TWAP"]
    previous = -np.inf
    for refill in (1.0, 0.75, 0.5, 0.25, 0.1, 0.0):
        cost = expected_execution_cost(
            problem, trades, book=book, liquidity=LiquidityModel(refill=refill)
        )
        assert cost > previous
        previous = cost


def test_replenishment_restores_the_missing_depth(book: LevelBook) -> None:
    state = LiquidityModel(refill=0.5).new_state(book)
    assert state.depth() == pytest.approx(book.capacity)
    state.walk(problem_quantity(book))
    depleted = state.depth()
    state.replenish()
    assert state.depth() == pytest.approx(depleted + 0.5 * (book.capacity - depleted))


def problem_quantity(book: LevelBook) -> float:
    """Half the book, to have something to deplete in the test above."""
    return 0.5 * book.capacity


def test_initial_fraction_scales_the_visible_depth(book: LevelBook) -> None:
    state = LiquidityModel(refill=1.0, initial_fraction=0.4).new_state(book)
    assert state.depth() == pytest.approx(0.4 * book.capacity)
    assert state.depth_fraction() == pytest.approx(0.4)


def test_no_replenishment_empties_the_book(book: LevelBook) -> None:
    state = LiquidityModel(refill=0.0).new_state(book)
    fill = state.walk(book.capacity)
    assert fill.quantity == pytest.approx(book.capacity)
    assert fill.beyond_quantity == 0.0
    assert state.depth() == pytest.approx(0.0, abs=1e-9)
    extra = state.walk(1.0)
    assert extra.beyond_quantity == pytest.approx(1.0)
    with pytest.raises(InsufficientDepthError):
        book.walk(book.capacity + 1.0)


def test_beyond_the_book_is_priced_by_extrapolation(book: LevelBook) -> None:
    state = LiquidityModel(refill=0.0).new_state(book)
    fill = state.walk(book.capacity + 5_000.0)
    assert fill.beyond_quantity == pytest.approx(5_000.0)
    assert fill.visible_quantity == pytest.approx(book.capacity)
    deep = LiquidityModel(refill=0.0).new_state(book)
    deep.walk(book.capacity)
    empty = deep.walk(5_000.0)
    assert empty.average_offset > fill.notional / fill.quantity - 1e-12
    assert empty.beyond_quantity == pytest.approx(5_000.0)


def test_strict_walk_refuses_to_go_past_the_book(book: LevelBook) -> None:
    with pytest.raises(InsufficientDepthError, match="available"):
        book.walk(book.capacity + 1.0)


def test_book_execution_reports_the_diagnostics(problem: ExecutionProblem, book: LevelBook) -> None:
    trades = immediate(problem)
    execution = book_execution(problem, trades, book, LiquidityModel(refill=0.5))
    assert execution.quantities.tolist() == trades.tolist()
    assert execution.max_child_fraction == pytest.approx(problem.size / book.capacity)
    assert execution.beyond_total == 0.0
    assert execution.exhausted_intervals == 0
    assert execution.depth_before[0] == pytest.approx(book.capacity)
    assert execution.depth_after[0] < execution.depth_before[0]


def test_immediate_execution_empties_half_the_ladder(
    problem: ExecutionProblem, book: LevelBook
) -> None:
    execution = book_execution(problem, immediate(problem), book, LiquidityModel(refill=0.25))
    assert execution.max_child_fraction > 0.5
    assert execution.depth_after[0] < 0.5 * book.capacity


def test_extreme_order_goes_beyond_the_book(problem: ExecutionProblem, book: LevelBook) -> None:
    stress = problem.with_changes(size=2.5 * problem.size, n_intervals=4)
    execution = book_execution(stress, immediate(stress), book, LiquidityModel(refill=0.25))
    assert execution.beyond_total > 0.0
    assert execution.exhausted_intervals == 1
    assert execution.max_child_fraction > 1.0


def test_max_child_fraction_handles_an_empty_book(problem: ExecutionProblem) -> None:
    small = LevelBook(level_size=10.0, decay=1.0, n_levels=1, tick=0.01)
    trades = np.array([10.0, 10.0])
    execution = book_execution(
        problem.with_changes(size=20.0, n_intervals=2), trades, small, LiquidityModel(refill=0.0)
    )
    assert np.isinf(execution.max_child_fraction)


def test_liquidity_model_validates_its_parameters() -> None:
    for refill in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="refill"):
            LiquidityModel(refill=refill)
    for fraction in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError, match="initial_fraction"):
            LiquidityModel(initial_fraction=fraction)


def test_refill_does_not_change_the_dispersion(problem: ExecutionProblem, book: LevelBook) -> None:
    """The book is deterministic: it moves the mean and leaves the variance alone."""
    trades = twap(problem)
    static = simulate(
        problem, trades, book=book, liquidity=LiquidityModel(refill=1.0), n_paths=20_000, seed=3
    )
    dynamic = simulate(
        problem, trades, book=book, liquidity=LiquidityModel(refill=0.25), n_paths=20_000, seed=3
    )
    assert dynamic.sd == pytest.approx(static.sd, rel=1e-9)
    assert dynamic.sd == pytest.approx(cost_sd(problem, trades), rel=0.01)
    assert dynamic.mean > static.mean


def test_simulated_mean_matches_the_running_book(
    problem: ExecutionProblem, book: LevelBook
) -> None:
    """The analytic machinery still works with the dynamic book."""
    for name, trades in build_schedules(problem, VolumeProfile()).items():
        liquidity = LiquidityModel(refill=0.25)
        analytic = expected_execution_cost(problem, trades, book=book, liquidity=liquidity)
        result = simulate(problem, trades, book=book, liquidity=liquidity, n_paths=40_000, seed=11)
        tolerance = max(5.0 * result.stderr, 1e-9 * abs(analytic))
        assert abs(result.mean - analytic) <= tolerance, name


def test_zero_refill_costs_match_the_block_cost(problem: ExecutionProblem, book: LevelBook) -> None:
    """Path independence: the block and the sliced schedule pay the same temporary cost."""
    zero = LiquidityModel(refill=0.0)

    def temporary(trades: np.ndarray) -> float:
        total = expected_execution_cost(problem, trades, book=book, liquidity=zero)
        array = np.asarray(trades)
        permanent = 0.5 * problem.market.gamma * (problem.size**2 - float(array @ array))
        return total - permanent

    assert temporary(twap(problem)) == pytest.approx(temporary(immediate(problem)), rel=1e-12)


def test_refill_scenarios_are_monotone(problem: ExecutionProblem, book: LevelBook) -> None:
    rows = refill_scenarios(
        problem,
        VolumeProfile(),
        book,
        refills=np.array([1.0, 0.5, 0.0]),
    )
    for name in ("TWAP", "Almgren-Chriss"):
        values = [row.expected_bps for row in rows if row.algorithm == name]
        assert values[0] < values[1] < values[2]
    assert all(row.max_child_pct > 0.0 for row in rows)
    assert all(row.depth_min_pct <= 100.0 for row in rows)


def test_refill_scenarios_keep_the_linear_cost_when_the_book_is_full(
    problem: ExecutionProblem, book: LevelBook
) -> None:
    rows = refill_scenarios(problem, VolumeProfile(), book, refills=np.array([1.0]))
    twap_row = next(row for row in rows if row.algorithm == "TWAP")
    assert twap_row.expected_bps == pytest.approx(
        to_bps(problem, expected_cost(problem, twap(problem))), rel=0.05
    )


def test_book_state_can_be_created_directly(book: LevelBook) -> None:
    state = BookState(book=book, available=book.sizes(), refill=1.0)
    assert state.depth_fraction() == pytest.approx(1.0)
    state.walk(1_000.0)
    assert state.depth_fraction() < 1.0
    state.replenish()
    assert state.depth_fraction() == pytest.approx(1.0, abs=1e-12)
