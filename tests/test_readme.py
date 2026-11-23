"""The README must quote numbers that the code actually produces.

Only the analytic tables are checked here: they do not depend on the Monte Carlo seed,
which is what makes a committed README safe to trust.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from execution_sim.report import build_report
from execution_sim.scenario import default_book, default_problem, default_profile

README = Path(__file__).resolve().parents[1] / "README.md"


def _table(header_prefix: str, columns: int) -> list[list[str]]:
    lines = README.read_text(encoding="utf-8").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith(header_prefix)),
        None,
    )
    if start is None:  # pragma: no cover - the README always has these tables
        pytest.fail(f"table not found in README: {header_prefix}")
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == columns
        rows.append(cells)
    return rows


@pytest.fixture(scope="module")
def report():
    problem = default_problem()
    return build_report(problem, default_profile(), default_book(problem), n_paths=200, seed=1)


def test_readme_algorithm_table(report) -> None:
    rows = _table("| Algorithm | E[cost] (bp)", 6)
    assert [row[0] for row in rows] == [item.name for item in report.algorithms]
    for row, expected in zip(rows, report.algorithms, strict=True):
        assert float(row[1]) == pytest.approx(expected.expected_bps, abs=5e-4)
        assert float(row[2]) == pytest.approx(expected.sd_bps, abs=5e-4)
        assert float(row[3]) == pytest.approx(expected.objective_bps, abs=5e-4)


def test_readme_book_table(report) -> None:
    rows = _table("| Algorithm | Static book (bp)", 6)
    assert [row[0] for row in rows] == [item.algorithm for item in report.book_rows]
    for row, expected in zip(rows, report.book_rows, strict=True):
        assert float(row[1]) == pytest.approx(expected.static_bps, abs=5e-4)
        assert float(row[2]) == pytest.approx(expected.dynamic_bps, abs=5e-4)
        assert float(row[3].rstrip("%")) == pytest.approx(expected.delta_pct, abs=0.05)
        assert float(row[4].rstrip("%")) == pytest.approx(expected.max_child_pct, abs=0.05)
        assert float(row[5].rstrip("%")) == pytest.approx(expected.depth_min_pct, abs=0.05)


def test_readme_extreme_table(report) -> None:
    rows = _table("| Algorithm | Max child / depth", 6)
    assert [row[0] for row in rows] == [item.algorithm for item in report.extreme.rows]
    for row, expected in zip(rows, report.extreme.rows, strict=True):
        assert float(row[1].rstrip("%")) == pytest.approx(expected.max_child_pct, abs=0.05)
        assert float(row[2].rstrip("%")) == pytest.approx(expected.beyond_pct, abs=0.05)
        assert float(row[3]) == pytest.approx(expected.static_bps, abs=5e-3)
        assert float(row[4]) == pytest.approx(expected.dynamic_bps, abs=5e-3)


def test_readme_quotes_the_path_independence(report) -> None:
    text = README.read_text(encoding="utf-8")
    assert "4·10⁻¹¹" in text
    assert report.temporary_spread_bps < 1e-6
