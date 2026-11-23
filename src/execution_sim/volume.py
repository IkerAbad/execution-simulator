"""Synthetic intraday volume, used by VWAP and POV.

The profile is a U shape: activity concentrates at the open and at the close, which
is what equity volume does on a normal day.  It is deterministic on purpose, so the
schedules are reproducible; volume forecast error is out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from execution_sim._typing import FloatArray
from execution_sim.core import ExecutionProblem


@dataclass(frozen=True, slots=True)
class VolumeProfile:
    """U-shaped intraday volume.

    Parameters
    ----------
    daily_volume:
        Market volume traded over the horizon, in shares.
    u_shape:
        Strength of the U.  ``0`` is a flat profile, ``1.5`` gives roughly three
        times more volume in the first and last intervals than in the middle.
    """

    daily_volume: float = 2_000_000.0
    u_shape: float = 1.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.daily_volume) or self.daily_volume <= 0.0:
            raise ValueError(
                f"daily_volume must be a positive finite number, got {self.daily_volume!r}"
            )
        if not math.isfinite(self.u_shape) or self.u_shape < 0.0:
            raise ValueError(f"u_shape must be non-negative and finite, got {self.u_shape!r}")

    def weights(self, n_intervals: int) -> FloatArray:
        """Fraction of the daily volume traded in each interval; sums to one."""
        if n_intervals < 1:
            raise ValueError(f"n_intervals must be >= 1, got {n_intervals}")
        midpoints = (np.arange(n_intervals, dtype=float) + 0.5) / n_intervals
        raw = 1.0 + self.u_shape * (2.0 * midpoints - 1.0) ** 2
        weights: FloatArray = raw / raw.sum()
        return weights

    def interval_volume(self, problem: ExecutionProblem) -> FloatArray:
        """Market volume traded in each interval of ``problem``, in shares."""
        return self.daily_volume * self.weights(problem.n_intervals)
