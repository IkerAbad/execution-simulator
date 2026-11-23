"""Market model and the definition of an execution problem.

The whole package works with a single object type, :class:`ExecutionProblem`, so that
every algorithm and every cost formula sees exactly the same parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from execution_sim._typing import FloatArray


def effective_eta(market: MarketModel, tau: float) -> float:
    """Temporary-impact coefficient net of the discrete permanent-impact correction.

    With linear permanent impact applied once per interval, the expected cost of a
    schedule contains ``eta / tau * sum(n_k**2)`` from temporary impact and
    ``-gamma / 2 * sum(n_k**2)`` from the fact that the average price of your own
    permanent impact is half of the total.  The two combine into
    ``eta_eff / tau * sum(n_k**2)`` with ``eta_eff = eta - gamma * tau / 2``.

    A positive ``eta_eff`` is required for the mean-variance objective to be convex.
    The check is not done here because the cost function is perfectly well defined
    when it is negative; the optimiser is the part that cannot work.
    """
    return market.eta - 0.5 * market.gamma * tau


def check_convex(market: MarketModel, tau: float) -> float:
    """Return ``eta_eff`` after checking that the objective is convex."""
    eta_eff = effective_eta(market, tau)
    if eta_eff <= 0.0:
        raise ValueError(
            "the mean-variance objective is not convex: eta - gamma * tau / 2 = "
            f"{eta_eff!r} <= 0. Reduce gamma, raise eta or use shorter intervals. "
            "See docs/derivation.md."
        )
    return eta_eff


@dataclass(frozen=True, slots=True)
class MarketModel:
    """Linear market impact in the Almgren-Chriss tradition.

    The mid price follows

        S_k = S_{k-1} + sigma * sqrt(tau) * xi_k - gamma * n_k

    where ``n_k`` is the number of shares sold in interval ``k``, and an order of
    ``n`` shares executed inside one interval pays, per share,

        half_spread + eta * n / tau

    on top of the prevailing mid.

    Parameters
    ----------
    price:
        Arrival mid price, in currency units.
    sigma:
        Absolute volatility per unit of ``sqrt(time)`` (``time`` in days here).
        With ``price = 50`` and 2% daily volatility, ``sigma = 1.0``.
    gamma:
        Permanent impact, in currency units per share traded.
    eta:
        Temporary impact, in currency units per share, per share traded per day.
    half_spread_bps:
        Half of the quoted bid-ask spread, in basis points of ``price``.
    """

    price: float = 50.0
    sigma: float = 1.0
    gamma: float = 2e-7
    eta: float = 4e-8
    half_spread_bps: float = 1.0

    def __post_init__(self) -> None:
        for name in ("price", "gamma", "eta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative and finite, got {self.sigma!r}")
        if not math.isfinite(self.half_spread_bps) or self.half_spread_bps < 0.0:
            raise ValueError(
                f"half_spread_bps must be non-negative and finite, got {self.half_spread_bps!r}"
            )

    @property
    def half_spread(self) -> float:
        """Half of the bid-ask spread, in currency units per share."""
        return self.price * self.half_spread_bps * 1e-4

    def with_changes(self, **changes: Any) -> MarketModel:
        """Return a copy with some fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class ExecutionProblem:
    """A liquidation problem: sell ``size`` shares over ``horizon`` in ``n_intervals`` steps.

    Parameters
    ----------
    market:
        The impact and volatility parameters.
    size:
        Number of shares to sell.
    horizon:
        Time available to complete the order, in days.
    n_intervals:
        Number of trading intervals.  ``tau = horizon / n_intervals``.
    risk_aversion:
        Mean-variance risk aversion, in inverse currency units.  It multiplies the
        variance of the implementation shortfall, so its numerical scale depends on
        the notional of the order; mean-variance preferences are only defined up to
        a positive scale factor anyway.
    """

    market: MarketModel = field(default_factory=MarketModel)
    size: float = 100_000.0
    horizon: float = 1.0
    n_intervals: int = 20
    risk_aversion: float = 1e-7

    def __post_init__(self) -> None:
        if not math.isfinite(self.size) or self.size <= 0.0:
            raise ValueError(f"size must be a positive finite number, got {self.size!r}")
        if not math.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValueError(f"horizon must be a positive finite number, got {self.horizon!r}")
        if isinstance(self.n_intervals, bool) or not isinstance(self.n_intervals, int):
            raise TypeError(f"n_intervals must be an int, got {type(self.n_intervals).__name__}")
        if self.n_intervals < 1:
            raise ValueError(f"n_intervals must be >= 1, got {self.n_intervals}")
        if not math.isfinite(self.risk_aversion) or self.risk_aversion < 0.0:
            raise ValueError(
                f"risk_aversion must be non-negative and finite, got {self.risk_aversion!r}"
            )

    @property
    def tau(self) -> float:
        """Length of one interval, in days."""
        return self.horizon / self.n_intervals

    @property
    def times(self) -> FloatArray:
        """Interval boundaries ``t_0 .. t_N``, in days."""
        return np.linspace(0.0, self.horizon, self.n_intervals + 1)

    @property
    def notional(self) -> float:
        """Value of the order at the arrival price, in currency units."""
        return self.size * self.market.price

    def with_changes(self, **changes: Any) -> ExecutionProblem:
        """Return a copy with some fields replaced."""
        return replace(self, **changes)


def check_trades(problem: ExecutionProblem, trades: FloatArray, *, tol_scale: float = 1e-9) -> None:
    """Validate that ``trades`` is a feasible liquidation schedule.

    A schedule is feasible when it is one-dimensional, finite, non-negative and sums
    to ``problem.size``.
    """
    array = np.asarray(trades, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"trades must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("trades must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("trades contain non-finite values")
    tol = tol_scale * problem.size
    if np.any(array < -tol):
        raise ValueError("trades must be non-negative: this package models a liquidation")
    total = float(array.sum())
    if abs(total - problem.size) > tol:
        raise ValueError(f"trades must sum to size={problem.size}, got {total}")
