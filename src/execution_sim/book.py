"""A finite-depth limit order book with level consumption.

The Almgren-Chriss model prices temporary impact as a linear function of the trade
size.  A real book is a step function: the first shares trade at the touch, the next
ones walk up (or down) the ladder and each level is thinner than the previous one.
This module walks that ladder exactly, so the slippage of a chunk comes from
consuming liquidity instead of from a formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from execution_sim._typing import FloatArray


class InsufficientDepthError(ValueError):
    """Raised when a child order is larger than the liquidity available in the book."""


@dataclass(frozen=True, slots=True)
class BookFill:
    """Result of walking the book with one market order."""

    quantity: float
    notional: float
    levels_used: int

    @property
    def average_offset(self) -> float:
        """Average distance from the mid, in currency units per share."""
        if self.quantity == 0.0:
            return 0.0
        return self.notional / self.quantity


@dataclass(frozen=True, slots=True)
class LevelBook:
    """A static, elastic book: ``n_levels`` levels with geometrically decaying size.

    Level ``j`` sits ``half_spread + j * tick`` away from the mid (below it, since
    this package liquidates) and holds ``level_size * decay**j`` shares.  The book is
    refilled after every interval, so a schedule is not penalised twice for the
    liquidity it consumes; intraday depletion is out of scope.

    Parameters
    ----------
    mid:
        Mid price the book is quoted around.
    tick:
        Distance between consecutive levels, in currency units.
    half_spread_bps:
        Distance from the mid to the first level, in basis points of ``mid``.
    level_size:
        Size of the first level, in shares.
    decay:
        Ratio between the size of level ``j + 1`` and level ``j``, in ``(0, 1]``.
    n_levels:
        Number of levels on the side being consumed.
    """

    mid: float = 50.0
    tick: float = 0.01
    half_spread_bps: float = 1.0
    level_size: float = 2_000.0
    decay: float = 0.99
    n_levels: int = 300

    def __post_init__(self) -> None:
        if not math.isfinite(self.mid) or self.mid <= 0.0:
            raise ValueError(f"mid must be a positive finite number, got {self.mid!r}")
        if not math.isfinite(self.tick) or self.tick <= 0.0:
            raise ValueError(f"tick must be a positive finite number, got {self.tick!r}")
        if not math.isfinite(self.half_spread_bps) or self.half_spread_bps < 0.0:
            raise ValueError(f"half_spread_bps must be non-negative, got {self.half_spread_bps!r}")
        if not math.isfinite(self.level_size) or self.level_size <= 0.0:
            raise ValueError(
                f"level_size must be a positive finite number, got {self.level_size!r}"
            )
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {self.decay!r}")
        if isinstance(self.n_levels, bool) or not isinstance(self.n_levels, int):
            raise TypeError(f"n_levels must be an int, got {type(self.n_levels).__name__}")
        if self.n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {self.n_levels}")

    @property
    def half_spread(self) -> float:
        """Distance from the mid to the first level, in currency units."""
        return self.mid * self.half_spread_bps * 1e-4

    def offsets(self) -> FloatArray:
        """Price offset of each level from the mid, in currency units per share."""
        levels = np.arange(self.n_levels, dtype=float)
        values: FloatArray = self.half_spread + levels * self.tick
        return values

    def sizes(self) -> FloatArray:
        """Size of each level, in shares."""
        levels = np.arange(self.n_levels, dtype=float)
        values: FloatArray = self.level_size * self.decay**levels
        return values

    @property
    def capacity(self) -> float:
        """Total liquidity available on the side being consumed, in shares."""
        if self.decay == 1.0:
            return self.level_size * self.n_levels
        return float(self.level_size * (1.0 - self.decay**self.n_levels) / (1.0 - self.decay))

    def implied_eta(self, tau: float) -> float:
        """Linear temporary-impact coefficient that matches the book for small orders.

        Inside the first level the average offset grows as
        ``half_spread + quantity * tick / (2 * level_size)``, so the Almgren-Chriss
        model with ``eta = tau * tick / (2 * level_size)`` reproduces the book for
        chunks smaller than the first level.
        """
        return tau * self.tick / (2.0 * self.level_size)

    @classmethod
    def calibrated(
        cls,
        tau: float,
        *,
        mid: float,
        eta: float,
        tick: float = 0.01,
        half_spread_bps: float = 1.0,
        decay: float = 0.99,
        n_levels: int = 300,
    ) -> LevelBook:
        """Build a book whose small-order slippage matches a target ``eta``."""
        level_size = tau * tick / (2.0 * eta)
        return cls(
            mid=mid,
            tick=tick,
            half_spread_bps=half_spread_bps,
            level_size=level_size,
            decay=decay,
            n_levels=n_levels,
        )

    def walk(self, quantity: float) -> BookFill:
        """Consume the book with a market order of ``quantity`` shares."""
        if not math.isfinite(quantity) or quantity < 0.0:
            raise ValueError(f"quantity must be non-negative and finite, got {quantity!r}")
        if quantity == 0.0:
            return BookFill(quantity=0.0, notional=0.0, levels_used=0)
        sizes = self.sizes()
        cumulative = np.cumsum(sizes)
        if quantity > cumulative[-1] + 1e-9 * max(1.0, quantity):
            raise InsufficientDepthError(
                f"order of {quantity:,.0f} shares exceeds book capacity "
                f"({cumulative[-1]:,.0f} shares): add levels or trade in smaller chunks"
            )
        index = int(np.searchsorted(cumulative, quantity, side="left"))
        index = min(index, self.n_levels - 1)
        filled_before = float(cumulative[index - 1]) if index > 0 else 0.0
        partial = quantity - filled_before
        offsets = self.offsets()
        notional = float(offsets[:index] @ sizes[:index]) + partial * float(offsets[index])
        return BookFill(quantity=quantity, notional=notional, levels_used=index + 1)

    def slippage(self, quantity: float) -> float:
        """Average execution offset from the mid, in currency units per share."""
        fill = self.walk(quantity)
        return fill.average_offset

    def slippages(self, quantities: FloatArray) -> FloatArray:
        """Vectorised :meth:`slippage`, used by the simulator over a whole schedule."""
        return np.array([self.slippage(float(q)) for q in quantities], dtype=float)
