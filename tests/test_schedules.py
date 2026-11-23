"""Algorithms: feasibility, shape and the identities that connect them."""

from __future__ import annotations

import numpy as np
import pytest

from execution_sim.core import ExecutionProblem
from execution_sim.schedules import (
    build_schedules,
    completion_time,
    immediate,
    pov,
    twap,
    vwap,
)
from execution_sim.volume import VolumeProfile


def test_immediate_sends_everything_at_once(problem: ExecutionProblem) -> None:
    trades = immediate(problem)
    assert trades[0] == problem.size
    assert trades.sum() == problem.size


def test_twap_is_flat(problem: ExecutionProblem) -> None:
    trades = twap(problem)
    assert np.allclose(trades, problem.size / problem.n_intervals)


def test_volume_weights_are_a_distribution() -> None:
    profile = VolumeProfile(daily_volume=1e6, u_shape=2.0)
    weights = profile.weights(20)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    assert weights[0] > weights[10]
    assert weights[-1] > weights[10]


def test_flat_profile_weights() -> None:
    weights = VolumeProfile(u_shape=0.0).weights(10)
    assert np.allclose(weights, 0.1)


def test_volume_profile_validates_its_parameters() -> None:
    with pytest.raises(ValueError, match="daily_volume"):
        VolumeProfile(daily_volume=0.0)
    with pytest.raises(ValueError, match="u_shape"):
        VolumeProfile(u_shape=-1.0)
    with pytest.raises(ValueError, match="n_intervals"):
        VolumeProfile().weights(0)


def test_vwap_is_proportional_to_volume(problem: ExecutionProblem, profile: VolumeProfile) -> None:
    trades = vwap(problem, profile)
    weights = profile.weights(problem.n_intervals)
    assert trades.sum() == pytest.approx(problem.size)
    assert np.allclose(trades / problem.size, weights)
    assert profile.interval_volume(problem).sum() == pytest.approx(profile.daily_volume)


def test_pov_at_the_neutral_rate_is_vwap(problem: ExecutionProblem, profile: VolumeProfile) -> None:
    rate = problem.size / profile.daily_volume
    assert np.allclose(pov(problem, profile, rate), vwap(problem, profile), rtol=1e-12)


def test_pov_finishes_early_when_aggressive(
    problem: ExecutionProblem, profile: VolumeProfile
) -> None:
    trades = pov(problem, profile, 5.0 * problem.size / profile.daily_volume)
    assert trades.sum() == pytest.approx(problem.size)
    assert completion_time(problem, trades) < problem.horizon
    assert np.count_nonzero(trades) < problem.n_intervals // 2


def test_pov_forces_the_residual_at_the_end(
    problem: ExecutionProblem, profile: VolumeProfile
) -> None:
    rate = 0.2 * problem.size / profile.daily_volume
    trades = pov(problem, profile, rate)
    assert trades.sum() == pytest.approx(problem.size)
    assert trades[-1] > rate * profile.interval_volume(problem)[-1]
    with pytest.raises(ValueError, match="does not complete"):
        pov(problem, profile, rate, force_completion=False)


def test_pov_rejects_bad_rates(problem: ExecutionProblem, profile: VolumeProfile) -> None:
    for rate in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="participation"):
            pov(problem, profile, rate)


def test_build_schedules_contains_every_algorithm(
    problem: ExecutionProblem, profile: VolumeProfile
) -> None:
    schedules = build_schedules(problem, profile)
    assert list(schedules) == ["Immediate", "TWAP", "VWAP", "POV", "Almgren-Chriss"]
    for trades in schedules.values():
        assert trades.size == problem.n_intervals
        assert trades.sum() == pytest.approx(problem.size, rel=1e-12)
        assert np.all(trades >= 0.0)


def test_build_schedules_accepts_a_participation_rate(
    problem: ExecutionProblem, profile: VolumeProfile
) -> None:
    schedules = build_schedules(problem, profile, participation=0.02)
    assert np.allclose(schedules["POV"], pov(problem, profile, 0.02))


def test_completion_time_of_twap_is_the_horizon(problem: ExecutionProblem) -> None:
    assert completion_time(problem, twap(problem)) == pytest.approx(problem.horizon)
