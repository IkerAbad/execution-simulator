"""Command line entry point: ``execution-sim report`` and friends."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from execution_sim import figures as figures_module
from execution_sim.impact import cost_sd, expected_cost, to_bps
from execution_sim.report import build_report, to_dict, to_markdown
from execution_sim.scenario import default_book, default_problem, default_profile
from execution_sim.schedules import build_schedules


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="execution-sim",
        description="Execution cost simulator for large orders (Almgren-Chriss).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="write the report and the figures")
    report.add_argument("--out", type=Path, default=Path("results"))
    report.add_argument("--figures", type=Path, default=Path("figures"))
    report.add_argument("--paths", type=int, default=20_000)
    report.add_argument("--seed", type=int, default=0)

    figure_parser = subparsers.add_parser("figures", help="regenerate the figures only")
    figure_parser.add_argument("--figures", type=Path, default=Path("figures"))

    subparsers.add_parser("compare", help="print the algorithm comparison table")
    return parser


def _compare() -> int:
    problem = default_problem()
    profile = default_profile()
    schedules = build_schedules(problem, profile)
    header = f"{'algorithm':<16}{'E[cost] (bp)':>14}{'SD (bp)':>12}{'E+lam*Var (bp)':>18}"
    print(header)
    print("-" * len(header))
    for name, trades in schedules.items():
        expected = expected_cost(problem, trades)
        sd = cost_sd(problem, trades)
        objective = expected + problem.risk_aversion * sd**2
        print(
            f"{name:<16}{to_bps(problem, expected):>14.3f}{to_bps(problem, sd):>12.3f}"
            f"{to_bps(problem, objective):>18.3f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return a process exit code."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "compare":
        return _compare()

    problem = default_problem()
    profile = default_profile()
    book = default_book(problem)

    if arguments.command == "figures":
        paths = figures_module.build_all(problem, profile, book, arguments.figures)
        for path in paths:
            print(path)
        return 0

    report = build_report(problem, profile, book, n_paths=arguments.paths, seed=arguments.seed)
    arguments.out.mkdir(parents=True, exist_ok=True)
    markdown = arguments.out / "summary.md"
    markdown.write_text(to_markdown(report), encoding="utf-8")
    payload = arguments.out / "summary.json"
    payload.write_text(json.dumps(to_dict(report), indent=2), encoding="utf-8")
    paths = figures_module.build_all(problem, profile, book, arguments.figures)
    print(markdown)
    print(payload)
    for path in paths:
        print(path)
    return 0
