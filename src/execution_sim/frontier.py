"""Cost-risk frontier: the set of Almgren-Chriss schedules indexed by risk aversion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from execution_sim._typing import FloatArray
from execution_sim.almgren_chriss import (
    first_interval_fraction,
    half_life,
    kappa,
    optimal_trades,
)
from execution_sim.core import ExecutionProblem
from execution_sim.impact import cost_sd, expected_cost, to_bps


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One schedule on the frontier, with the numbers needed to justify it."""

    risk_aversion: float
    expected_cost: float
    sd_cost: float
    expected_bps: float
    sd_bps: float
    kappa: float
    first_interval_fraction: float
    half_life: float

    @property
    def mean_variance(self) -> float:
        """``E + lambda * Var`` at this point's own risk aversion."""
        return self.expected_cost + self.risk_aversion * self.sd_cost**2


def default_risk_aversions(problem: ExecutionProblem) -> FloatArray:
    """A log grid of risk aversions around the problem's own value.

    The scale of ``lambda`` is arbitrary (mean-variance is defined up to a positive
    factor), so only ratios to ``problem.risk_aversion`` are meaningful.
    """
    reference = problem.risk_aversion if problem.risk_aversion > 0.0 else 1e-7
    grid = np.logspace(-2.0, 2.0, 41) * reference
    return np.concatenate(([0.0], grid))


def point(problem: ExecutionProblem, risk_aversion: float) -> FrontierPoint:
    """Evaluate the optimal schedule for one risk aversion."""
    trades = optimal_trades(problem, risk_aversion)
    expected = expected_cost(problem, trades)
    sd = cost_sd(problem, trades)
    return FrontierPoint(
        risk_aversion=risk_aversion,
        expected_cost=expected,
        sd_cost=sd,
        expected_bps=to_bps(problem, expected),
        sd_bps=to_bps(problem, sd),
        kappa=kappa(problem, risk_aversion),
        first_interval_fraction=first_interval_fraction(trades, problem.size),
        half_life=half_life(problem, trades),
    )


def frontier(
    problem: ExecutionProblem, risk_aversions: FloatArray | None = None
) -> list[FrontierPoint]:
    """The whole frontier, sorted by increasing risk aversion."""
    grid = default_risk_aversions(problem) if risk_aversions is None else risk_aversions
    return [point(problem, float(lam)) for lam in np.sort(np.asarray(grid, dtype=float))]


def lambda_for_cost_increase(
    problem: ExecutionProblem,
    extra_bps: float,
    *,
    tol: float = 1e-10,
    max_iterations: int = 200,
) -> FrontierPoint:
    """Find the frontier point whose expected cost is ``extra_bps`` above TWAP.

    Expected cost increases monotonically with risk aversion along the frontier, so a
    bisection on ``log(lambda)`` is enough.  This is the lambda-free way of
    quantifying what a trader gives up by ignoring risk: it answers "how much risk
    reduction does one extra basis point of expected cost buy?".
    """
    if extra_bps <= 0.0:
        raise ValueError(f"extra_bps must be positive, got {extra_bps!r}")
    baseline = expected_cost(problem, _twap(problem))
    target = baseline + extra_bps * problem.notional * 1e-4
    low = 1e-12
    high = 1.0
    for _ in range(max_iterations):
        middle = math.sqrt(low * high)
        value = expected_cost(problem, optimal_trades(problem, middle))
        if value < target:
            low = middle
        else:
            high = middle
        if abs(value - target) < tol * problem.notional:  # pragma: no cover - safety valve
            break
    candidate = math.sqrt(low * high)
    return point(problem, candidate)


def _twap(problem: ExecutionProblem) -> FloatArray:
    from execution_sim.schedules import twap

    return twap(problem)
