"""Sensitivity sweeps: shape, monotonicity and the economic sign of the effects."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from execution_sim.core import ExecutionProblem
from execution_sim.sensitivity import (
    ScenarioRow,
    drift_scenarios,
    impact_scenarios,
    volatility_curves,
)
from execution_sim.volume import VolumeProfile


def _lookup(rows: list[ScenarioRow], scenario: str, algorithm: str) -> ScenarioRow:
    for row in rows:
        if row.scenario == scenario and row.algorithm == algorithm:
            return row
    raise AssertionError(f"missing row: {scenario} / {algorithm}")


def test_impact_scenarios_cover_every_combination(
    problem: ExecutionProblem, profile: VolumeProfile
) -> None:
    rows = impact_scenarios(
        problem,
        profile,
        eta_multipliers=np.array([0.5, 1.0]),
        gamma_multipliers=np.array([1.0, 2.0]),
    )
    assert len(rows) == 2 * 2 * 5
    assert {row.scenario for row in rows} == {
        "eta x0.5 / gamma x1",
        "eta x0.5 / gamma x2",
        "eta x1 / gamma x1",
        "eta x1 / gamma x2",
    }


def test_more_temporary_impact_costs_more(
    problem: ExecutionProblem, profile: VolumeProfile
) -> None:
    rows = impact_scenarios(
        problem,
        profile,
        eta_multipliers=np.array([0.5, 1.0, 2.0]),
        gamma_multipliers=np.array([1.0]),
    )
    costs = [
        _lookup(rows, f"eta x{multiplier:g} / gamma x1", "TWAP").expected_bps
        for multiplier in (0.5, 1.0, 2.0)
    ]
    assert costs[0] < costs[1] < costs[2]


def test_permanent_impact_does_not_change_the_dispersion(
    problem: ExecutionProblem, profile: VolumeProfile
) -> None:
    rows = impact_scenarios(
        problem, profile, eta_multipliers=np.array([1.0]), gamma_multipliers=np.array([1.0, 2.0])
    )
    low = _lookup(rows, "eta x1 / gamma x1", "TWAP")
    high = _lookup(rows, "eta x1 / gamma x2", "TWAP")
    assert high.expected_bps > low.expected_bps
    assert high.sd_bps == pytest.approx(low.sd_bps)


def test_volatility_curves(problem: ExecutionProblem) -> None:
    curves = volatility_curves(problem, multipliers=np.array([0.5, 1.0]), n_points=15)
    assert [curve.label for curve in curves] == ["sigma x0.5", "sigma x1"]
    for curve in curves:
        assert len(curve.points) == 16
        assert curve.points[0].risk_aversion == 0.0
        sds = [item.sd_bps for item in curve.points]
        assert all(later < earlier for earlier, later in itertools.pairwise(sds))
    # A risk-neutral schedule is TWAP whatever the volatility is, so the endpoints
    # scale exactly with sigma.
    assert curves[0].points[0].sd_bps == pytest.approx(0.5 * curves[1].points[0].sd_bps)
    # Higher volatility raises urgency at a fixed risk aversion.
    assert curves[0].points[-1].kappa < curves[1].points[-1].kappa


def test_drift_helps_a_seller_who_waits(problem: ExecutionProblem, profile: VolumeProfile) -> None:
    rows = drift_scenarios(problem, profile, drifts_bps_per_day=np.array([-20.0, 0.0, 20.0]))
    twap_rows = [row for row in rows if row.algorithm == "TWAP"]
    assert twap_rows[0].expected_bps > twap_rows[1].expected_bps > twap_rows[2].expected_bps
    # For TWAP, sum(n_k * t_{k-1}) = X * T * (N - 1) / (2N); a 20 bp/day drift over a
    # one-day horizon with 20 intervals is worth 20 * 0.475 bp per side.
    assert twap_rows[1].expected_bps - twap_rows[2].expected_bps == pytest.approx(9.5, rel=1e-6)
    assert twap_rows[0].expected_bps - twap_rows[1].expected_bps == pytest.approx(9.5, rel=1e-6)
