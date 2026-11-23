"""Monte Carlo simulation of an execution schedule against the book.

The simulator separates the two sources of cost:

* a deterministic part, which comes from the spread, the temporary impact (either
  the linear formula or the book ladder) and the permanent impact left behind by
  the schedule;
* a random part, which comes from the mid price following a random walk while the
  position is open.

Because both parts are additive, the simulated mean converges to the analytic
expected cost and the simulated standard deviation to ``sigma * sqrt(tau * sum x**2)``.
The test suite checks exactly that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from execution_sim._typing import FloatArray
from execution_sim.book import LevelBook
from execution_sim.core import ExecutionProblem, check_trades


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Implementation shortfall of every simulated path, in currency units."""

    costs: FloatArray

    @property
    def mean(self) -> float:
        return float(self.costs.mean())

    @property
    def sd(self) -> float:
        return float(self.costs.std(ddof=1))

    @property
    def stderr(self) -> float:
        return self.sd / math.sqrt(self.costs.size)

    @property
    def n_paths(self) -> int:
        return int(self.costs.size)


def execution_slippages(
    problem: ExecutionProblem,
    trades: FloatArray,
    book: LevelBook | None = None,
) -> FloatArray:
    """Per-share execution cost relative to the mid, for each interval.

    With ``book`` the cost is the average offset obtained by walking the ladder
    (which already contains the half spread).  Without it, the linear model
    ``half_spread + eta * n / tau`` is used.
    """
    array = np.asarray(trades, dtype=float)
    if book is not None:
        return book.slippages(array)
    market = problem.market
    return market.half_spread + market.eta * array / problem.tau


def expected_execution_cost(
    problem: ExecutionProblem,
    trades: FloatArray,
    *,
    book: LevelBook | None = None,
    drift: float = 0.0,
) -> float:
    """Deterministic part of the implementation shortfall: no price noise.

    For the linear model this equals :func:`execution_sim.impact.expected_cost`.
    """
    check_trades(problem, trades)
    array = np.asarray(trades, dtype=float)
    slippages = execution_slippages(problem, array, book)
    traded = float(array.sum())
    temporary = float(array @ slippages)
    squares = float(array @ array)
    permanent = 0.5 * problem.market.gamma * (traded**2 - squares)
    timing = -drift * float(array @ problem.times[:-1])
    return temporary + permanent + timing


def simulate(
    problem: ExecutionProblem,
    trades: FloatArray,
    *,
    book: LevelBook | None = None,
    n_paths: int = 20_000,
    seed: int = 0,
    drift: float = 0.0,
) -> SimulationResult:
    """Simulate ``n_paths`` price paths and return the implementation shortfall.

    The trade of interval ``k`` executes at the mid observed at the start of the
    interval; afterwards the mid absorbs the interval's price innovation and the
    permanent impact of the trade.
    """
    check_trades(problem, trades)
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
    array = np.asarray(trades, dtype=float)
    market = problem.market
    slippages = execution_slippages(problem, array, book)
    rng = np.random.default_rng(seed)
    costs = np.zeros(n_paths, dtype=float)
    mid = np.zeros(n_paths, dtype=float)
    step = market.sigma * math.sqrt(problem.tau)
    for index in range(problem.n_intervals):
        costs += array[index] * (-mid + slippages[index])
        mid += drift * problem.tau + step * rng.standard_normal(n_paths)
        mid -= market.gamma * array[index]
    return SimulationResult(costs=costs)
