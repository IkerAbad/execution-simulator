"""Expected cost and variance of a liquidation schedule under linear impact.

All the formulas here are the closed-form consequences of the model in
:mod:`execution_sim.core`.  They are derived in ``docs/derivation.md``.
"""

from __future__ import annotations

import numpy as np

from execution_sim._typing import FloatArray
from execution_sim.core import ExecutionProblem, check_trades


def inventories(problem: ExecutionProblem, trades: FloatArray) -> FloatArray:
    """Remaining inventory ``x_0 .. x_N`` implied by a trade list ``n_1 .. n_N``."""
    array = np.asarray(trades, dtype=float)
    path = np.empty(problem.n_intervals + 1, dtype=float)
    path[0] = problem.size
    np.cumsum(array, out=path[1:])
    path[1:] = problem.size - path[1:]
    path[-1] = 0.0
    return path


def sum_squares(trades: FloatArray) -> float:
    """``sum(n_k**2)``, the quantity that drives expected impact cost."""
    array = np.asarray(trades, dtype=float)
    return float(array @ array)


def expected_cost(problem: ExecutionProblem, trades: FloatArray, *, drift: float = 0.0) -> float:
    """Expected implementation shortfall, in currency units.

    ``0.5 * gamma * (X**2 - sum(n_k**2)) + eta / tau * sum(n_k**2) + half_spread * X``
    minus ``drift * sum(n_k * t_{k-1})`` when the mid price has a drift.

    The first term is the permanent impact actually paid: trading in blocks lets the
    price recover between intervals, so only ``X**2 - sum(n_k**2)`` of permanent
    impact is unavoidable rather than ``X**2``.
    """
    check_trades(problem, trades)
    array = np.asarray(trades, dtype=float)
    market = problem.market
    squares = sum_squares(array)
    permanent = 0.5 * market.gamma * (problem.size**2 - squares)
    temporary = market.eta * squares / problem.tau
    spread = market.half_spread * problem.size
    timing = -drift * float(array @ problem.times[:-1])
    return permanent + temporary + spread + timing


def variance(problem: ExecutionProblem, trades: FloatArray) -> float:
    """Variance of the implementation shortfall, in currency units squared.

    ``sigma**2 * tau * sum(x_k**2)`` with ``x_k`` the inventory held after ``k``
    intervals.  It only depends on how long the position stays open, not on the
    impact parameters or the drift.
    """
    check_trades(problem, trades)
    path = inventories(problem, trades)
    return problem.market.sigma**2 * problem.tau * float(path[1:] @ path[1:])


def cost_sd(problem: ExecutionProblem, trades: FloatArray) -> float:
    """Standard deviation of the implementation shortfall, in currency units."""
    return float(np.sqrt(variance(problem, trades)))


def mean_variance(
    problem: ExecutionProblem,
    trades: FloatArray,
    risk_aversion: float | None = None,
    *,
    drift: float = 0.0,
) -> float:
    """``E[cost] + lambda * Var[cost]``, the objective Almgren-Chriss minimises.

    ``risk_aversion`` defaults to ``problem.risk_aversion``.
    """
    lam = problem.risk_aversion if risk_aversion is None else risk_aversion
    return expected_cost(problem, trades, drift=drift) + lam * variance(problem, trades)


def to_bps(problem: ExecutionProblem, amount: float) -> float:
    """Express a currency amount as basis points of the order notional."""
    return amount / problem.notional * 1e4


def from_bps(problem: ExecutionProblem, bps: float) -> float:
    """Express basis points of the order notional as a currency amount."""
    return bps * problem.notional * 1e-4
