"""The execution algorithms that are compared: immediate, TWAP, VWAP, POV and AC."""

from __future__ import annotations

import numpy as np

from execution_sim._typing import FloatArray
from execution_sim.almgren_chriss import optimal_trades
from execution_sim.core import ExecutionProblem, check_trades
from execution_sim.volume import VolumeProfile


def immediate(problem: ExecutionProblem) -> FloatArray:
    """Send the whole order in the first interval."""
    trades = np.zeros(problem.n_intervals, dtype=float)
    trades[0] = problem.size
    return trades


def twap(problem: ExecutionProblem) -> FloatArray:
    """Equal shares in every interval."""
    return np.full(problem.n_intervals, problem.size / problem.n_intervals, dtype=float)


def vwap(problem: ExecutionProblem, profile: VolumeProfile) -> FloatArray:
    """Shares proportional to the expected volume of each interval."""
    weights = profile.weights(problem.n_intervals)
    trades: FloatArray = problem.size * weights
    return trades


def pov(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    participation: float,
    *,
    force_completion: bool = True,
) -> FloatArray:
    """Participate at a fixed fraction of market volume until the order is done.

    ``participation`` is a fraction of market volume (``0.05`` is 5%).  The horizon
    is endogenous: the order finishes when it finishes, and if it is still open at
    the end of the horizon the remainder is executed in the last interval
    (``force_completion=True``).  With ``participation`` equal to
    ``size / daily_volume`` this schedule collapses into :func:`vwap`.
    """
    if not 0.0 < participation <= 1.0:
        raise ValueError(f"participation must be in (0, 1], got {participation!r}")
    capacity = participation * profile.interval_volume(problem)
    trades = np.zeros(problem.n_intervals, dtype=float)
    remaining = problem.size
    for index, chunk in enumerate(capacity):
        filled = min(float(chunk), remaining)
        trades[index] = filled
        remaining -= filled
        if remaining <= 0.0:
            break
    if remaining > 1e-9 * problem.size:
        if force_completion:
            trades[-1] += remaining
        else:  # pragma: no cover - only reachable by explicitly disabling completion
            raise ValueError(
                f"participation={participation} does not complete the order within the horizon"
            )
    return trades


def completion_time(problem: ExecutionProblem, trades: FloatArray) -> float:
    """Time, in days, when the last share is executed."""
    active = np.nonzero(np.asarray(trades, dtype=float) > 0.0)[0]
    if active.size == 0:  # pragma: no cover - sales must be positive
        return 0.0
    return float(problem.times[int(active[-1]) + 1])


def build_schedules(
    problem: ExecutionProblem,
    profile: VolumeProfile | None = None,
    *,
    participation: float | None = None,
) -> dict[str, FloatArray]:
    """All algorithms on the same problem, keyed by display name.

    ``participation`` is expressed as a fraction of market volume.  By default POV
    runs at 1.5 times the neutral rate ``size / daily_volume``, i.e. it finishes
    earlier than the horizon instead of tracking volume exactly.
    """
    profile = VolumeProfile() if profile is None else profile
    rate = participation
    if rate is None:
        rate = 1.5 * problem.size / profile.daily_volume
    schedules = {
        "Immediate": immediate(problem),
        "TWAP": twap(problem),
        "VWAP": vwap(problem, profile),
        "POV": pov(problem, profile, rate),
        "Almgren-Chriss": optimal_trades(problem),
    }
    for name, trades in schedules.items():
        check_trades(problem, trades)
        if not np.isfinite(trades).all():  # pragma: no cover - defensive
            raise RuntimeError(f"schedule {name!r} contains non-finite values")
    return schedules
