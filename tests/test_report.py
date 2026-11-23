"""The report and the command line must run end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution_sim.cli import main
from execution_sim.report import Report, build_report, to_dict, to_markdown
from execution_sim.scenario import default_book, default_profile


@pytest.fixture(scope="module")
def report() -> Report:
    from execution_sim.scenario import default_problem

    problem = default_problem()
    return build_report(
        problem,
        default_profile(),
        default_book(problem),
        n_paths=1_000,
        seed=3,
    )


def test_report_covers_every_algorithm(report: Report) -> None:
    names = [row.name for row in report.algorithms]
    assert names == ["Immediate", "TWAP", "VWAP", "POV", "Almgren-Chriss"]
    best = min(report.algorithms, key=lambda row: row.objective_bps)
    assert best.name == "Almgren-Chriss"
    assert best.excess_objective_bps == pytest.approx(0.0)


def test_report_quantifies_twap() -> None:
    from execution_sim.scenario import default_problem

    report = build_report(default_problem(), default_profile(), n_paths=500, seed=1)
    assert report.ladder[0].risk_aversion == 0.0
    assert report.ladder[0].objective_reduction_pct == pytest.approx(0.0, abs=1e-9)
    positive = [row for row in report.ladder[1:]]
    assert all(row.objective_reduction_pct > 0.0 for row in positive)
    assert report.price_of_risk["sd_reduction_bps"] > 0.0


def test_report_compares_static_and_dynamic_books(report: Report) -> None:
    names = [row.algorithm for row in report.book_rows]
    assert names == ["Immediate", "TWAP", "VWAP", "POV", "Almgren-Chriss"]
    twap = next(row for row in report.book_rows if row.algorithm == "TWAP")
    ac = next(row for row in report.book_rows if row.algorithm == "Almgren-Chriss")
    assert twap.dynamic_bps > twap.static_bps
    assert ac.dynamic_bps > ac.static_bps
    assert twap.delta_pct > 50.0
    assert report.extreme.rows and report.extreme.order_to_depth_pct > 100.0
    assert any(row.beyond_pct > 0.0 for row in report.extreme.rows)
    assert report.temporary_spread_bps < 1e-6
    assert {round(row.refill, 2) for row in report.refill_curve} >= {1.0, 0.25, 0.0}


def test_report_validates_the_model(report: Report) -> None:
    for row in report.validation:
        assert abs(row.expected_z_score) < 5.0
        assert abs(row.sd_error_pct) < 5.0
    assert report.book_gap_pct != 0.0
    assert report.derived_check_bps < 1e-4


def test_report_renders_markdown(report: Report) -> None:
    text = to_markdown(report)
    assert "# Execution report" in text
    assert "## TWAP versus the optimal schedule" in text
    assert "## Price of risk" in text
    assert "## The book does not refill for free" in text
    assert "## Extreme case" in text
    assert text.count("|") > 100


def test_report_is_json_serialisable(report: Report) -> None:
    payload = json.dumps(to_dict(report))
    assert "Almgren-Chriss" in payload


def test_cli_compare(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare"]) == 0
    out = capsys.readouterr().out
    assert "Almgren-Chriss" in out
    assert "TWAP" in out


def test_cli_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "results"
    figures = tmp_path / "figures"
    assert main(["report", "--out", str(out), "--figures", str(figures), "--paths", "500"]) == 0
    assert (out / "summary.md").read_text(encoding="utf-8").startswith("# Execution report")
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "algorithms" in payload
    assert len(list(figures.glob("*.png"))) == 6
    assert "summary.md" in capsys.readouterr().out


def test_cli_figures(tmp_path: Path) -> None:
    assert main(["figures", "--figures", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.png"))) == 6


def test_cli_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        main(["nope"])
