"""The closed-form Almgren-Chriss optimal liquidation schedule.

The derivation, done by hand, lives in ``docs/derivation.md``.  In short, with
linear permanent impact, linear temporary impact and no drift, the mean-variance
objective

    J(x) = eta_eff / tau * sum(n_k**2) + lambda * sigma**2 * tau * sum(x_k**2)

(plus terms that do not depend on the schedule) is minimised by

    x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T),
    cosh(kappa * tau) = 1 + lambda * sigma**2 * tau**2 / (2 * eta_eff).

The special case ``lambda = 0`` collapses to the linear schedule, i.e. TWAP.
"""

from __future__ import annotations

import math

import numpy as np

from execution_sim._typing import FloatArray
from execution_sim.core import ExecutionProblem, check_convex


def kappa(problem: ExecutionProblem, risk_aversion: float | None = None) -> float:
    """The urgency parameter of the optimal schedule, in inverse days.

    ``kappa = arccosh(1 + lambda * sigma**2 * tau**2 / (2 * eta_eff)) / tau``.
    It is zero for a risk-neutral trader and grows with risk aversion, volatility
    and the ratio ``tau / eta_eff``.
    """
    lam = problem.risk_aversion if risk_aversion is None else risk_aversion
    if lam < 0.0:
        raise ValueError(f"risk_aversion must be non-negative, got {lam!r}")
    if lam == 0.0 or problem.market.sigma == 0.0:
        return 0.0
    tau = problem.tau
    eta_eff = check_convex(problem.market, tau)
    x = lam * problem.market.sigma**2 * tau**2 / (2.0 * eta_eff)
    # arccosh(1 + x) = log(1 + x + sqrt(x * (x + 2))), stable for small x.
    return math.log1p(x + math.sqrt(x * (x + 2.0))) / tau


def _sinh_ratio(numerator: FloatArray, denominator: float) -> FloatArray:
    """``sinh(a) / sinh(b)`` for ``0 <= a <= b`` without overflow at large ``b``."""
    if denominator <= 0.0:
        raise ValueError("denominator must be positive")
    if denominator < 1e-8:
        # sinh(a) / sinh(b) -> a / b as b -> 0.
        return numerator / denominator
    ratio: FloatArray = (
        np.exp(numerator - denominator) * np.expm1(-2.0 * numerator) / np.expm1(-2.0 * denominator)
    )
    return ratio


def inventory_path(problem: ExecutionProblem, risk_aversion: float | None = None) -> FloatArray:
    """Optimal remaining inventory ``x_0 .. x_N`` under mean-variance preferences."""
    lam = problem.risk_aversion if risk_aversion is None else risk_aversion
    rate = kappa(problem, lam)
    if rate == 0.0:
        path = problem.size * (1.0 - problem.times / problem.horizon)
    else:
        scaled = rate * (problem.horizon - problem.times)
        path = problem.size * _sinh_ratio(scaled, rate * problem.horizon)
    path[0] = problem.size
    path[-1] = 0.0
    return path


def optimal_trades(problem: ExecutionProblem, risk_aversion: float | None = None) -> FloatArray:
    """The Almgren-Chriss trade list ``n_1 .. n_N`` (a closed form)."""
    path = inventory_path(problem, risk_aversion)
    trades: FloatArray = -np.diff(path)
    trades[trades < 0.0] = 0.0
    return trades


def _solve_tridiagonal(
    lower: FloatArray, diag: FloatArray, upper: FloatArray, rhs: FloatArray
) -> FloatArray:
    """Thomas algorithm for a tridiagonal system, written out by hand."""
    n = diag.size
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.array([rhs[0] / diag[0]], dtype=float)
    c = np.zeros(n, dtype=float)
    d = np.zeros(n, dtype=float)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    solution = np.zeros(n, dtype=float)
    solution[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d[i] - c[i] * solution[i + 1]
    return solution


def numeric_optimal_trades(
    problem: ExecutionProblem, risk_aversion: float | None = None
) -> FloatArray:
    """The same optimum obtained by solving the first-order conditions numerically.

    The stationarity condition of the discrete objective is the linear system

        -x_{k-1} + (2 + c) * x_k - x_{k+1} = 0,   c = lambda * sigma**2 * tau**2 / eta_eff

    on the interior grid, with ``x_0 = X`` and ``x_N = 0``.  Solving it with the
    Thomas algorithm gives the exact discrete optimum without using the closed form,
    which is what makes it a useful independent check in the test suite.
    """
    lam = problem.risk_aversion if risk_aversion is None else risk_aversion
    n = problem.n_intervals
    if n == 1:
        return np.array([problem.size], dtype=float)
    tau = problem.tau
    eta_eff = check_convex(problem.market, tau)
    c = lam * problem.market.sigma**2 * tau**2 / eta_eff
    size = n - 1
    lower = np.full(size - 1, -1.0)
    diag = np.full(size, 2.0 + c)
    upper = np.full(size - 1, -1.0)
    rhs = np.zeros(size, dtype=float)
    rhs[0] = problem.size
    interior = _solve_tridiagonal(lower, diag, upper, rhs)
    path = np.empty(n + 1, dtype=float)
    path[0] = problem.size
    path[1:-1] = interior
    path[-1] = 0.0
    trades: FloatArray = -np.diff(path)
    trades[trades < 0.0] = 0.0
    return trades


def first_interval_fraction(trades: FloatArray, size: float) -> float:
    """Share of the order executed in the first interval."""
    return float(trades[0] / size) if trades.size else 0.0


def half_life(problem: ExecutionProblem, trades: FloatArray) -> float:
    """Time, in days, at which half of the order is done (linear interpolation)."""
    from execution_sim.impact import inventories

    path = inventories(problem, trades)
    target = 0.5 * problem.size
    below = np.nonzero(path <= target)[0]
    if below.size == 0:  # pragma: no cover - path[0] == size > target
        return 0.0
    index = int(below[0])
    if index == 0:  # pragma: no cover - defensive
        return 0.0
    previous = path[index - 1]
    fraction = (previous - target) / (previous - path[index])
    return float(problem.times[index - 1] + fraction * problem.tau)
