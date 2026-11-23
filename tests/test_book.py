"""The finite-depth book: consumption, monotonicity and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from execution_sim.book import BookFill, InsufficientDepthError, LevelBook
from execution_sim.core import ExecutionProblem
from execution_sim.scenario import default_book


def test_level_geometry() -> None:
    book = LevelBook(
        mid=50.0, tick=0.01, half_spread_bps=2.0, level_size=1_000.0, decay=0.9, n_levels=5
    )
    assert book.half_spread == pytest.approx(0.01)
    assert book.offsets().tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])
    assert book.sizes().tolist() == pytest.approx([1_000.0, 900.0, 810.0, 729.0, 656.1])
    assert book.capacity == pytest.approx(1_000.0 + 900.0 + 810.0 + 729.0 + 656.1)


def test_full_levels_have_constant_capacity() -> None:
    book = LevelBook(level_size=100.0, decay=1.0, n_levels=10)
    assert book.capacity == pytest.approx(1_000.0)


def test_walk_consumes_levels_in_order() -> None:
    book = LevelBook(
        mid=100.0, tick=0.05, half_spread_bps=0.0, level_size=100.0, decay=1.0, n_levels=10
    )
    fill = book.walk(250.0)
    assert fill.quantity == 250.0
    assert fill.levels_used == 3
    expected = 100 * 0.0 + 100 * 0.05 + 50 * 0.10
    assert fill.notional == pytest.approx(expected)
    assert fill.average_offset == pytest.approx(expected / 250.0)


def test_walk_at_an_exact_boundary() -> None:
    book = LevelBook(
        mid=100.0, tick=0.05, half_spread_bps=0.0, level_size=100.0, decay=1.0, n_levels=3
    )
    fill = book.walk(200.0)
    assert fill.levels_used == 2
    assert fill.average_offset == pytest.approx(0.025)


def test_slippage_is_increasing_and_convex() -> None:
    book = LevelBook(level_size=1_000.0, decay=0.95, n_levels=50)
    quantities = np.linspace(10.0, 0.9 * book.capacity, 400)
    costs = np.array([book.slippage(float(q)) for q in quantities])
    assert np.all(np.diff(costs) >= -1e-15)
    total = costs * quantities
    assert np.all(np.diff(total, 2) >= -1e-9)
    assert costs[-1] > 3.0 * costs[0]


def test_book_is_calibrated_at_the_mean_chunk_and_convex_beyond(
    problem: ExecutionProblem,
) -> None:
    book = default_book(problem)
    mean_chunk = problem.size / problem.n_intervals
    linear = problem.market.half_spread + problem.market.eta * mean_chunk / problem.tau
    assert book.slippage(mean_chunk) == pytest.approx(linear, rel=1e-12)

    bigger = 10.0 * mean_chunk
    linear_bigger = problem.market.half_spread + problem.market.eta * bigger / problem.tau
    assert book.slippage(bigger) > 1.4 * linear_bigger

    block = problem.size
    linear_block = problem.market.half_spread + problem.market.eta * block / problem.tau
    assert book.slippage(block) > 1.5 * linear_block


def test_capacity_is_enforced() -> None:
    book = LevelBook(level_size=100.0, decay=1.0, n_levels=3)
    with pytest.raises(InsufficientDepthError, match="exceeds"):
        book.walk(301.0)


def test_zero_quantity() -> None:
    book = LevelBook()
    fill = book.walk(0.0)
    assert fill == BookFill(quantity=0.0, notional=0.0, levels_used=0)
    assert fill.average_offset == 0.0


def test_vectorised_slippages() -> None:
    book = LevelBook()
    quantities = np.array([0.0, 100.0, 5_000.0])
    vector = book.slippages(quantities)
    assert vector.tolist() == pytest.approx([book.slippage(float(q)) for q in quantities])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("mid", 0.0, ValueError),
        ("tick", 0.0, ValueError),
        ("tick", float("nan"), ValueError),
        ("half_spread_bps", -1.0, ValueError),
        ("level_size", 0.0, ValueError),
        ("decay", 0.0, ValueError),
        ("decay", 1.5, ValueError),
        ("n_levels", 0, ValueError),
        ("n_levels", 2.5, TypeError),
    ],
)
def test_invalid_book_parameters(field: str, value: float, error: type[Exception]) -> None:
    with pytest.raises(error):
        LevelBook(**{field: value})


def test_negative_quantity_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LevelBook().walk(-1.0)


def test_calibrated_book_matches_its_target(problem: ExecutionProblem) -> None:
    book = LevelBook.calibrated(problem.tau, mid=50.0, eta=4e-8, n_levels=100)
    assert book.implied_eta(problem.tau) == pytest.approx(4e-8)
    assert book.capacity > problem.size
