"""The reference scenario used by the README and the command line.

One liquid large cap, a 100k-share sell order, one trading day sliced into 20
intervals.  The numbers are illustrative but internally consistent: the book has a
fixed, realistic shape and the linear temporary impact is calibrated to it at the
average child order, which is the size the analytics are meant to describe.
"""

from __future__ import annotations

from execution_sim.book import LevelBook
from execution_sim.core import ExecutionProblem, MarketModel
from execution_sim.volume import VolumeProfile

DEFAULT_SIZE = 100_000.0
DEFAULT_PRICE = 50.0
DEFAULT_HORIZON = 1.0
DEFAULT_INTERVALS = 20
DEFAULT_VOLATILITY = 1.0  # 2% of a 50 currency-unit price
DEFAULT_GAMMA = 2e-7
DEFAULT_HALF_SPREAD_BPS = 1.0
DEFAULT_RISK_AVERSION = 1e-6
DEFAULT_DAILY_VOLUME = 2_000_000.0

# Book shape: 2,000 shares at the touch, then thinner levels one cent apart.
DEFAULT_TICK = 0.01
DEFAULT_LEVEL_SIZE = 2_000.0
DEFAULT_DECAY = 0.99
DEFAULT_LEVELS = 300


def default_book(problem: ExecutionProblem | None = None) -> LevelBook:
    """The reference L2 book, quoted around the problem's price."""
    price = DEFAULT_PRICE if problem is None else problem.market.price
    half_spread_bps = DEFAULT_HALF_SPREAD_BPS if problem is None else problem.market.half_spread_bps
    return LevelBook(
        mid=price,
        tick=DEFAULT_TICK,
        half_spread_bps=half_spread_bps,
        level_size=DEFAULT_LEVEL_SIZE,
        decay=DEFAULT_DECAY,
        n_levels=DEFAULT_LEVELS,
    )


def eta_matching_book(book: LevelBook, tau: float, chunk: float) -> float:
    """Temporary impact that reproduces the book's slippage at one chunk size."""
    return (book.slippage(chunk) - book.half_spread) * tau / chunk


def default_eta() -> float:
    """Linear temporary impact calibrated to the reference book at the mean chunk."""
    book = default_book()
    return eta_matching_book(
        book, DEFAULT_HORIZON / DEFAULT_INTERVALS, DEFAULT_SIZE / DEFAULT_INTERVALS
    )


def default_problem() -> ExecutionProblem:
    """The reference liquidation problem."""
    return ExecutionProblem(
        market=MarketModel(
            price=DEFAULT_PRICE,
            sigma=DEFAULT_VOLATILITY,
            gamma=DEFAULT_GAMMA,
            eta=default_eta(),
            half_spread_bps=DEFAULT_HALF_SPREAD_BPS,
        ),
        size=DEFAULT_SIZE,
        horizon=DEFAULT_HORIZON,
        n_intervals=DEFAULT_INTERVALS,
        risk_aversion=DEFAULT_RISK_AVERSION,
    )


def default_profile() -> VolumeProfile:
    """The reference intraday volume profile."""
    return VolumeProfile(daily_volume=DEFAULT_DAILY_VOLUME, u_shape=1.5)
