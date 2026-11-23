"""Figures: they must build, be non-trivial and be byte-for-byte reproducible."""

from __future__ import annotations

from pathlib import Path

import pytest

from execution_sim.book import LevelBook
from execution_sim.core import ExecutionProblem
from execution_sim.figures import build_all
from execution_sim.volume import VolumeProfile

EXPECTED = {
    "cost_risk_inventory.png",
    "efficient_frontier.png",
    "sensitivity_volatility.png",
    "sensitivity_impact.png",
    "risk_aversion.png",
}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_all_figures_are_generated(
    problem: ExecutionProblem, profile: VolumeProfile, book: LevelBook, tmp_path: Path
) -> None:
    paths = build_all(problem, profile, book, tmp_path / "figures")
    assert {path.name for path in paths} == EXPECTED
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 20_000


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_figures_are_reproducible(
    problem: ExecutionProblem, profile: VolumeProfile, book: LevelBook, tmp_path: Path
) -> None:
    first = build_all(problem, profile, book, tmp_path / "a")
    second = build_all(problem, profile, book, tmp_path / "b")
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes(), left.name


def test_committed_figures_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "figures"
    if not root.exists():  # pragma: no cover - only before the first report run
        pytest.skip("figures have not been generated yet")
    assert {path.name for path in root.glob("*.png")} >= EXPECTED
