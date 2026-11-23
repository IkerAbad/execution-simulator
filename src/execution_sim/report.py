"""Build the comparison report that both the CLI and the README rely on."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from execution_sim._typing import FloatArray
from execution_sim.almgren_chriss import (
    first_interval_fraction,
    half_life,
    kappa,
    numeric_optimal_trades,
    optimal_trades,
)
from execution_sim.book import LevelBook
from execution_sim.core import ExecutionProblem
from execution_sim.engine import simulate
from execution_sim.frontier import lambda_for_cost_increase
from execution_sim.impact import cost_sd, expected_cost, mean_variance, to_bps
from execution_sim.scenario import default_book
from execution_sim.schedules import build_schedules, completion_time
from execution_sim.sensitivity import (
    ScenarioRow,
    drift_scenarios,
    impact_scenarios,
    volatility_curves,
)
from execution_sim.volume import VolumeProfile


@dataclass(frozen=True, slots=True)
class AlgorithmRow:
    name: str
    expected_bps: float
    sd_bps: float
    objective_bps: float
    excess_objective_bps: float
    first_interval_pct: float
    half_life_days: float
    completion_days: float


@dataclass(frozen=True, slots=True)
class LadderRow:
    risk_aversion: float
    expected_twap_bps: float
    expected_ac_bps: float
    delta_expected_bps: float
    sd_twap_bps: float
    sd_ac_bps: float
    delta_sd_bps: float
    objective_twap_bps: float
    objective_ac_bps: float
    objective_reduction_pct: float


@dataclass(frozen=True, slots=True)
class ValidationRow:
    algorithm: str
    analytic_expected_bps: float
    simulated_expected_bps: float
    analytic_sd_bps: float
    simulated_sd_bps: float
    expected_z_score: float
    sd_error_pct: float


@dataclass(frozen=True, slots=True)
class Report:
    settings: dict[str, str]
    algorithms: list[AlgorithmRow]
    ladder: list[LadderRow]
    validation: list[ValidationRow]
    price_of_risk: dict[str, float]
    impact: list[ScenarioRow]
    volatility: list[tuple[str, list[tuple[float, float]]]]
    drift: list[ScenarioRow]
    book_gap_pct: float
    block_gap_pct: float
    derived_check_bps: float


def _algorithm_rows(
    problem: ExecutionProblem, profile: VolumeProfile
) -> tuple[list[AlgorithmRow], dict[str, FloatArray]]:
    schedules = build_schedules(problem, profile)
    reference = mean_variance(problem, schedules["Almgren-Chriss"])
    rows: list[AlgorithmRow] = []
    for name, trades in schedules.items():
        objective = mean_variance(problem, trades)
        rows.append(
            AlgorithmRow(
                name=name,
                expected_bps=to_bps(problem, expected_cost(problem, trades)),
                sd_bps=to_bps(problem, cost_sd(problem, trades)),
                objective_bps=to_bps(problem, objective),
                excess_objective_bps=to_bps(problem, objective - reference),
                first_interval_pct=100.0 * first_interval_fraction(trades, problem.size),
                half_life_days=half_life(problem, trades),
                completion_days=completion_time(problem, trades),
            )
        )
    return rows, schedules


def _ladder_rows(problem: ExecutionProblem, risk_aversions: FloatArray) -> list[LadderRow]:
    from execution_sim.schedules import twap

    twap_trades = twap(problem)
    twap_expected = expected_cost(problem, twap_trades)
    twap_sd = cost_sd(problem, twap_trades)
    rows: list[LadderRow] = []
    for lam in risk_aversions:
        trades = optimal_trades(problem, float(lam))
        expected = expected_cost(problem, trades)
        sd = cost_sd(problem, trades)
        twap_objective = twap_expected + float(lam) * twap_sd**2
        ac_objective = expected + float(lam) * sd**2
        rows.append(
            LadderRow(
                risk_aversion=float(lam),
                expected_twap_bps=to_bps(problem, twap_expected),
                expected_ac_bps=to_bps(problem, expected),
                delta_expected_bps=to_bps(problem, expected - twap_expected),
                sd_twap_bps=to_bps(problem, twap_sd),
                sd_ac_bps=to_bps(problem, sd),
                delta_sd_bps=to_bps(problem, sd - twap_sd),
                objective_twap_bps=to_bps(problem, twap_objective),
                objective_ac_bps=to_bps(problem, ac_objective),
                objective_reduction_pct=100.0 * (1.0 - ac_objective / twap_objective),
            )
        )
    return rows


def _validation_rows(
    problem: ExecutionProblem, schedules: dict[str, FloatArray], n_paths: int, seed: int
) -> list[ValidationRow]:
    rows: list[ValidationRow] = []
    for index, (name, trades) in enumerate(schedules.items()):
        result = simulate(problem, trades, n_paths=n_paths, seed=seed + index)
        analytic_expected = expected_cost(problem, trades)
        analytic_sd = cost_sd(problem, trades)
        z_score = 0.0 if result.stderr == 0.0 else (result.mean - analytic_expected) / result.stderr
        sd_error = 0.0 if analytic_sd == 0.0 else 100.0 * (result.sd - analytic_sd) / analytic_sd
        rows.append(
            ValidationRow(
                algorithm=name,
                analytic_expected_bps=to_bps(problem, analytic_expected),
                simulated_expected_bps=to_bps(problem, result.mean),
                analytic_sd_bps=to_bps(problem, analytic_sd),
                simulated_sd_bps=to_bps(problem, result.sd),
                expected_z_score=float(z_score),
                sd_error_pct=float(sd_error),
            )
        )
    return rows


def build_report(
    problem: ExecutionProblem,
    profile: VolumeProfile | None = None,
    book: LevelBook | None = None,
    *,
    n_paths: int = 20_000,
    seed: int = 0,
    ladder_risk_aversions: FloatArray | None = None,
) -> Report:
    """Run every comparison that the README quotes."""
    profile = VolumeProfile() if profile is None else profile
    book = default_book(problem) if book is None else book
    lams = (
        np.array([0.0, 1e-7, 1e-6, 1e-5, 1e-4])
        if ladder_risk_aversions is None
        else np.asarray(ladder_risk_aversions, dtype=float)
    )

    algorithms, schedules = _algorithm_rows(problem, profile)
    ladder = _ladder_rows(problem, lams)

    mean_chunk = problem.size / problem.n_intervals
    linear = problem.market.half_spread + problem.market.eta * mean_chunk / problem.tau
    book_cost = book.slippage(mean_chunk)
    book_gap = 100.0 * (book_cost - linear) / linear
    block_linear = problem.market.half_spread + problem.market.eta * problem.size / problem.tau
    block_book = book.slippage(problem.size)
    block_gap = 100.0 * (block_book - block_linear) / block_linear

    derived = optimal_trades(problem)
    numeric = numeric_optimal_trades(problem)
    derived_check = float(np.max(np.abs(derived - numeric))) / problem.size * 1e4

    curve = volatility_curves(problem, multipliers=np.array([0.5, 1.0, 1.5, 2.0]), n_points=21)
    volatility = [(item.label, [(p.expected_bps, p.sd_bps) for p in item.points]) for item in curve]

    price_of_risk = _price_of_risk(problem)

    settings = {
        "order size (shares)": f"{problem.size:,.0f}",
        "notional (currency)": f"{problem.notional:,.0f}",
        "horizon (days)": f"{problem.horizon:g}",
        "intervals": f"{problem.n_intervals}",
        "tau (days)": f"{problem.tau:g}",
        "price": f"{problem.market.price:g}",
        "sigma (per sqrt(day))": f"{problem.market.sigma:g}",
        "gamma (per share)": f"{problem.market.gamma:g}",
        "eta (per share per day)": f"{problem.market.eta:g}",
        "half spread (bps)": f"{problem.market.half_spread_bps:g}",
        "risk aversion": f"{problem.risk_aversion:g}",
        "kappa (1/day)": f"{kappa(problem):.4f}",
        "daily volume (shares)": f"{profile.daily_volume:,.0f}",
        "book levels": f"{book.n_levels}",
        "book capacity (shares)": f"{book.capacity:,.0f}",
        "simulated paths": f"{n_paths:,}",
    }

    return Report(
        settings=settings,
        algorithms=algorithms,
        ladder=ladder,
        validation=_validation_rows(problem, schedules, n_paths, seed),
        price_of_risk=price_of_risk,
        impact=impact_scenarios(
            problem,
            profile,
            eta_multipliers=np.array([0.5, 1.0, 2.0]),
            gamma_multipliers=np.array([0.5, 1.0, 2.0]),
        ),
        volatility=volatility,
        drift=drift_scenarios(
            problem,
            profile,
            drifts_bps_per_day=np.array([-20.0, 0.0, 20.0]),
        ),
        book_gap_pct=float(book_gap),
        block_gap_pct=float(block_gap),
        derived_check_bps=derived_check,
    )


def _price_of_risk(problem: ExecutionProblem) -> dict[str, float]:
    from execution_sim.schedules import twap

    baseline_expected = expected_cost(problem, twap(problem))
    baseline_sd = to_bps(problem, cost_sd(problem, twap(problem)))
    upgraded = lambda_for_cost_increase(problem, 1.0)
    reduction = baseline_sd - upgraded.sd_bps
    return {
        "baseline_expected_bps": to_bps(problem, baseline_expected),
        "baseline_sd_bps": baseline_sd,
        "extra_cost_bps": 1.0,
        "sd_after_bps": upgraded.sd_bps,
        "sd_reduction_bps": reduction,
        "cost_per_sd_bp": 1.0 / reduction if reduction > 0.0 else math.inf,
        "lambda": upgraded.risk_aversion,
        "kappa": upgraded.kappa,
    }


def _numeric(value: float, digits: int = 2) -> str:
    if math.isinf(value):
        return "inf"
    if abs(value) < 0.5 * 10**-digits:
        value = 0.0
    return f"{value:,.{digits}f}"


def to_markdown(report: Report) -> str:
    """Render the report as Markdown, tables included."""
    lines: list[str] = ["# Execution report", "", "## Parameters", "", "| Parameter | Value |"]
    lines.append("|---|---|")
    lines.extend(f"| {key} | {value} |" for key, value in report.settings.items())

    lines += ["", "## Algorithms", ""]
    lines.append(
        "| Algorithm | E[cost] (bp) | SD (bp) | E + lambda*Var (bp) | Excess vs AC (bp) "
        "| First interval (%) | Half-life (d) | Completion (d) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for algorithm in report.algorithms:
        lines.append(
            f"| {algorithm.name} | {_numeric(algorithm.expected_bps, 3)} "
            f"| {_numeric(algorithm.sd_bps, 3)} "
            f"| {_numeric(algorithm.objective_bps, 3)} "
            f"| {_numeric(algorithm.excess_objective_bps, 3)} "
            f"| {_numeric(algorithm.first_interval_pct, 1)} "
            f"| {_numeric(algorithm.half_life_days, 3)} "
            f"| {_numeric(algorithm.completion_days, 3)} |"
        )

    lines += ["", "## TWAP versus the optimal schedule", ""]
    lines.append(
        "| Risk aversion | E TWAP (bp) | E AC (bp) | dE (bp) | SD TWAP (bp) | SD AC (bp) "
        "| dSD (bp) | J TWAP (bp) | J AC (bp) | J reduction (%) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for rung in report.ladder:
        lines.append(
            f"| {rung.risk_aversion:g} | {_numeric(rung.expected_twap_bps, 3)} "
            f"| {_numeric(rung.expected_ac_bps, 3)} | {_numeric(rung.delta_expected_bps, 3)} "
            f"| {_numeric(rung.sd_twap_bps, 3)} | {_numeric(rung.sd_ac_bps, 3)} "
            f"| {_numeric(rung.delta_sd_bps, 3)} | {_numeric(rung.objective_twap_bps, 3)} "
            f"| {_numeric(rung.objective_ac_bps, 3)} "
            f"| {_numeric(rung.objective_reduction_pct, 2)} |"
        )

    risk = report.price_of_risk
    lines += [
        "",
        "## Price of risk",
        "",
        f"Starting from TWAP ({_numeric(risk['baseline_expected_bps'], 3)} bp expected, "
        f"{_numeric(risk['baseline_sd_bps'], 3)} bp SD), spending "
        f"{_numeric(risk['extra_cost_bps'], 1)} bp of expected cost buys "
        f"{_numeric(risk['sd_reduction_bps'], 3)} bp of risk reduction: "
        f"{_numeric(risk['cost_per_sd_bp'], 3)} bp of expected cost per bp of SD "
        f"(risk aversion {risk['lambda']:.3g}, kappa {risk['kappa']:.4f}/day).",
    ]

    lines += ["", "## Validation against the closed form", ""]
    lines.append(
        "| Algorithm | E analytic (bp) | E simulated (bp) | SD analytic (bp) | SD simulated (bp) "
        "| z-score | SD error (%) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for check in report.validation:
        lines.append(
            f"| {check.algorithm} | {_numeric(check.analytic_expected_bps, 3)} "
            f"| {_numeric(check.simulated_expected_bps, 3)} | {_numeric(check.analytic_sd_bps, 3)} "
            f"| {_numeric(check.simulated_sd_bps, 3)} | {_numeric(check.expected_z_score, 2)} "
            f"| {_numeric(check.sd_error_pct, 2)} |"
        )
    lines += [
        "",
        f"Maximum difference between the closed-form schedule and the numerical "
        f"solution of the first-order conditions: {report.derived_check_bps:.2e} bp of size.",
        f"Book slippage minus the linear model at the mean chunk ({report.book_gap_pct:+.2f}%).",
        f"At the full order as a single block the same gap is {report.block_gap_pct:+.1f}%: "
        "the linear model is a local approximation.",
    ]

    lines += ["", "## Sensitivity to impact parameters", ""]
    names = _algorithm_names(report.impact)
    scenarios = _scenario_names(report.impact)
    lines.append("| Scenario | " + " | ".join(f"E {name} (bp)" for name in names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    lookup = {(row.scenario, row.algorithm): row.expected_bps for row in report.impact}
    for scenario in scenarios:
        cells = " | ".join(_numeric(lookup[(scenario, name)], 3) for name in names)
        lines.append(f"| {scenario} | {cells} |")

    lines += ["", "## Sensitivity to volatility", ""]
    lines.append("| Curve | SD range (bp) | Cost range (bp) |")
    lines.append("|---|---|---|")
    for label, points in report.volatility:
        sds = [sd for _, sd in points]
        costs = [cost for cost, _ in points]
        lines.append(
            f"| {label} | {_numeric(min(sds), 3)} to {_numeric(max(sds), 3)} "
            f"| {_numeric(min(costs), 3)} to {_numeric(max(costs), 3)} |"
        )

    lines += ["", "## Sensitivity to drift", ""]
    lines.append("| Scenario | " + " | ".join(f"E {name} (bp)" for name in names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    lookup = {(row.scenario, row.algorithm): row.expected_bps for row in report.drift}
    for scenario in _scenario_names(report.drift):
        cells = " | ".join(_numeric(lookup[(scenario, name)], 3) for name in names)
        lines.append(f"| {scenario} | {cells} |")

    return "\n".join(lines) + "\n"


def _algorithm_names(rows: list[ScenarioRow]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row.algorithm not in seen:
            seen.append(row.algorithm)
    return seen


def _scenario_names(rows: list[ScenarioRow]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row.scenario not in seen:
            seen.append(row.scenario)
    return seen


def to_dict(report: Report) -> dict[str, object]:
    """Plain-data view of the report, for JSON output."""
    return asdict(report)
