"""Shared fixtures."""

from __future__ import annotations

import pytest

from execution_sim.book import LevelBook
from execution_sim.core import ExecutionProblem, MarketModel
from execution_sim.scenario import default_book, default_problem, default_profile
from execution_sim.volume import VolumeProfile


@pytest.fixture
def problem() -> ExecutionProblem:
    return default_problem()


@pytest.fixture
def profile() -> VolumeProfile:
    return default_profile()


@pytest.fixture
def book(problem: ExecutionProblem) -> LevelBook:
    return default_book(problem)


@pytest.fixture
def small_problem() -> ExecutionProblem:
    """A cheap problem for tests that only need the mechanics."""
    return ExecutionProblem(
        market=MarketModel(price=100.0, sigma=2.0, gamma=1e-6, eta=1e-6, half_spread_bps=2.0),
        size=10_000.0,
        horizon=0.5,
        n_intervals=10,
        risk_aversion=1e-5,
    )
