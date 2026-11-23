"""A finite-depth limit order book with level consumption and partial replenishment.

The Almgren-Chriss model prices temporary impact as a linear function of the trade
size.  A real book is a step function: the first shares trade at the touch, the next
ones walk up (or down) the ladder and each level is thinner than the previous one.
This module walks that ladder exactly, so the slippage of a chunk comes from
consuming liquidity instead of from a formula.

Two layers live here:

* :class:`LevelBook` is the *layout*: how deep each level is and how far from the
  mid it sits.
* :class:`BookState` is a *running book*: how much of that layout is still available
  after the previous child orders, and how much of the missing depth comes back
  between intervals (:class:`LiquidityModel`).

With ``refill = 1`` every interval starts from the full ladder, which is the static
book used by the earlier version of the simulator.  With ``refill < 1`` the book is
depleted by large child orders and only slowly recovers, so the cost of an order
depends on what was traded before it.
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
    beyond_quantity: float = 0.0

    @property
    def average_offset(self) -> float:
        """Average distance from the mid, in currency units per share."""
        if self.quantity == 0.0:
            return 0.0
        return self.notional / self.quantity

    @property
    def visible_quantity(self) -> float:
        """Shares executed against levels that were actually on the book."""
        return self.quantity - self.beyond_quantity


@dataclass(frozen=True, slots=True)
class LevelBook:
    """A static ladder: ``n_levels`` levels with geometrically decaying size.

    Level ``j`` sits ``half_spread + j * tick`` away from the mid (below it, since
    this package liquidates) and holds ``level_size * decay**j`` shares.

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

    @property
    def last_level_size(self) -> float:
        """Size of the deepest level, used to extrapolate beyond the book."""
        return float(self.level_size * self.decay ** (self.n_levels - 1))

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

    def walk(self, quantity: float, *, extrapolate: bool = False) -> BookFill:
        """Consume the book with a market order of ``quantity`` shares.

        With ``extrapolate=False`` (the default) an order larger than the ladder
        raises :class:`InsufficientDepthError`.  With ``extrapolate=True`` the part
        beyond the last level is priced by continuing the ladder at the same tick
        spacing and the same size as the deepest level.
        """
        fill, _ = _walk(
            self.offsets(),
            self.sizes(),
            quantity,
            tick=self.tick,
            last_block=self.last_level_size,
            extrapolate=extrapolate,
        )
        return fill

    def slippage(self, quantity: float) -> float:
        """Average execution offset from the mid, in currency units per share."""
        fill = self.walk(quantity)
        return fill.average_offset

    def slippages(self, quantities: FloatArray) -> FloatArray:
        """Vectorised :meth:`slippage`, used by the simulator over a whole schedule."""
        return np.array([self.slippage(float(q)) for q in quantities], dtype=float)


@dataclass(frozen=True, slots=True)
class LiquidityModel:
    """How the visible depth is restored between intervals.

    Parameters
    ----------
    refill:
        Fraction of the *missing* depth that comes back after every interval.
        ``1`` means the book is fully replenished and the model is the static book
        of the original experiment; ``0`` means the book is never refilled and every
        share consumed is gone for the rest of the horizon.
    initial_fraction:
        Fraction of the ladder that is visible at ``t = 0``.
    """

    refill: float = 1.0
    initial_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.refill) or not 0.0 <= self.refill <= 1.0:
            raise ValueError(f"refill must be in [0, 1], got {self.refill!r}")
        if not math.isfinite(self.initial_fraction) or not 0.0 < self.initial_fraction <= 1.0:
            raise ValueError(f"initial_fraction must be in (0, 1], got {self.initial_fraction!r}")

    def new_state(self, book: LevelBook) -> BookState:
        """Create the running book for one simulation."""
        return BookState(
            book=book,
            available=self.initial_fraction * book.sizes(),
            refill=self.refill,
        )


@dataclass(slots=True)
class BookState:
    """A running book: the part of the ladder that is still available right now."""

    book: LevelBook
    available: FloatArray
    refill: float = 1.0

    def depth(self) -> float:
        """Shares still visible on the book, in total."""
        return float(self.available.sum())

    def depth_fraction(self) -> float:
        """Visible depth as a fraction of the full ladder."""
        return self.depth() / self.book.capacity

    def walk(self, quantity: float) -> BookFill:
        """Consume the running book, extrapolating beyond the deepest level."""
        fill, updated = _walk(
            self.book.offsets(),
            self.available,
            quantity,
            tick=self.book.tick,
            last_block=self.book.last_level_size,
            extrapolate=True,
        )
        self.available = updated
        return fill

    def replenish(self) -> None:
        """Restore ``refill`` of the missing depth at the end of an interval."""
        capacity = self.book.sizes()
        gap = capacity - self.available
        np.clip(self.available + self.refill * gap, 0.0, capacity, out=self.available)


def _walk(
    offsets: FloatArray,
    available: FloatArray,
    quantity: float,
    *,
    tick: float,
    last_block: float,
    extrapolate: bool,
) -> tuple[BookFill, FloatArray]:
    """Shared consumption routine: returns the fill and the updated availability."""
    if not math.isfinite(quantity) or quantity < 0.0:
        raise ValueError(f"quantity must be non-negative and finite, got {quantity!r}")
    if quantity == 0.0:
        return BookFill(quantity=0.0, notional=0.0, levels_used=0), available.copy()

    cumulative = np.cumsum(available)
    visible_depth = float(cumulative[-1])
    visible = min(quantity, visible_depth)
    beyond = quantity - visible

    notional = 0.0
    levels_used = 0
    index = 0
    partial = 0.0
    if visible > 0.0:
        index = min(int(np.searchsorted(cumulative, visible, side="left")), offsets.size - 1)
        filled_before = float(cumulative[index - 1]) if index > 0 else 0.0
        partial = visible - filled_before
        notional = float(offsets[:index] @ available[:index]) + partial * float(offsets[index])
        levels_used = index + 1

    if beyond > 1e-9 * max(1.0, quantity):
        if not extrapolate:
            raise InsufficientDepthError(
                f"order of {quantity:,.0f} shares exceeds the {visible_depth:,.0f} shares "
                "available on the book: add levels, trade in smaller chunks or allow "
                "extrapolation beyond the deepest level"
            )
        block = max(last_block, 1e-12)
        start = float(offsets[-1]) + tick
        whole = math.floor(beyond / block)
        rest = beyond - whole * block
        notional += block * (whole * start + 0.5 * tick * whole * (whole - 1))
        if rest > 0.0:
            notional += rest * (start + whole * tick)
        levels_used += whole + (1 if rest > 0.0 else 0)
    else:
        beyond = 0.0

    updated = available.copy()
    if index > 0:
        updated[:index] = 0.0
    if partial > 0.0:
        updated[index] = max(0.0, updated[index] - partial)

    return (
        BookFill(
            quantity=quantity,
            notional=notional,
            levels_used=levels_used,
            beyond_quantity=beyond,
        ),
        updated,
    )
