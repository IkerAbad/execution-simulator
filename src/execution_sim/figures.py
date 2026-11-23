"""Figures for the report.  Every figure is generated from the model, nothing drawn by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from execution_sim._typing import FloatArray
from execution_sim.almgren_chriss import optimal_trades
from execution_sim.book import LevelBook, LiquidityModel
from execution_sim.core import ExecutionProblem
from execution_sim.engine import book_execution
from execution_sim.frontier import lambda_for_cost_increase, point
from execution_sim.impact import cost_sd, expected_cost, to_bps
from execution_sim.schedules import build_schedules
from execution_sim.sensitivity import impact_scenarios, refill_scenarios, volatility_curves
from execution_sim.volume import VolumeProfile

COLORS = {
    "Immediate": "#c0392b",
    "TWAP": "#2c3e50",
    "VWAP": "#8e44ad",
    "POV": "#d68910",
    "Almgren-Chriss": "#1f77b4",
}
MARKERS = {
    "Immediate": "X",
    "TWAP": "o",
    "VWAP": "s",
    "POV": "D",
    "Almgren-Chriss": "*",
}
LABELS = {
    "Immediate": "Immediate",
    "TWAP": "TWAP",
    "VWAP": "VWAP",
    "POV": "POV",
    "Almgren-Chriss": "Almgren-Chriss",
}


def _style(ax: Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, metadata={"Software": "execution-sim"})
    return path


def cost_risk_inventory(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    book: LevelBook,
    path: Path,
    *,
    n_sizes: int = 17,
) -> Path:
    """Expected cost, dispersion, inventory paths and real book slippage."""
    sizes = np.linspace(0.1 * problem.size, 1.8 * problem.size, n_sizes)
    scenarios = [problem.with_changes(size=float(size)) for size in sizes]
    names = list(build_schedules(problem, profile))
    expected: dict[str, list[float]] = {name: [] for name in names}
    dispersion: dict[str, list[float]] = {name: [] for name in names}
    for scenario in scenarios:
        schedules = build_schedules(scenario, profile)
        for name in names:
            expected[name].append(to_bps(scenario, expected_cost(scenario, schedules[name])))
            dispersion[name].append(to_bps(scenario, cost_sd(scenario, schedules[name])))

    figure = Figure(figsize=(11.5, 8.0), layout="constrained")
    axes = figure.subplots(2, 2)
    share = sizes / problem.size * 100.0

    ax = axes[0, 0]
    for name in names:
        ax.plot(
            share,
            expected[name],
            color=COLORS[name],
            marker=MARKERS[name],
            markersize=4,
            label=LABELS[name],
        )
    ax.set_xlabel("Order size (% of the base order)")
    ax.set_ylabel("Expected cost (bp)")
    ax.set_title("(a) Expected cost against size")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[0, 1]
    for name in names:
        ax.plot(
            share,
            dispersion[name],
            color=COLORS[name],
            marker=MARKERS[name],
            markersize=4,
            label=LABELS[name],
        )
    ax.set_xlabel("Order size (% of the base order)")
    ax.set_ylabel("Standard deviation of cost (bp)")
    ax.set_title("(b) Dispersion: scale-invariant in percentage terms")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1, 0]
    times = problem.times / problem.horizon * 100.0
    for name, trades in build_schedules(problem, profile).items():
        inventory = np.concatenate(([problem.size], problem.size - np.cumsum(trades)))
        ax.plot(
            times,
            inventory / problem.size * 100.0,
            color=COLORS[name],
            linewidth=2.0,
            label=LABELS[name],
        )
    ax.set_xlabel("Time (% of the horizon)")
    ax.set_ylabel("Remaining inventory (%)")
    ax.set_title("(c) Inventory trajectories")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1, 1]
    chunks = np.logspace(2, np.log10(book.capacity * 0.95), 60)
    book_bps = np.array([book.slippage(float(q)) for q in chunks]) / problem.market.price * 1e4
    mean_chunk = problem.size / problem.n_intervals
    linear_bps = (
        (problem.market.half_spread + problem.market.eta * chunks / problem.tau)
        / problem.market.price
        * 1e4
    )
    ax.plot(chunks, book_bps, color="#16a085", linewidth=2.0, label="Book walk")
    ax.plot(
        chunks,
        linear_bps,
        color="#7f8c8d",
        linestyle="--",
        linewidth=1.6,
        label="Linear impact model",
    )
    ax.axvline(mean_chunk, color="#2c3e50", linestyle=":", linewidth=1.2)
    ax.annotate(
        "TWAP chunk",
        xy=(mean_chunk, book.slippage(float(mean_chunk)) / problem.market.price * 1e4),
        xytext=(6, 18),
        textcoords="offset points",
        fontsize=8,
        color="#2c3e50",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Child order size (shares)")
    ax.set_ylabel("Slippage including spread (bp)")
    ax.set_title("(d) Finite liquidity: book against linear model")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    return _save(figure, path)


def efficient_frontier(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    path: Path,
) -> Path:
    """The cost-risk frontier and where each algorithm sits relative to it."""
    grid = np.logspace(-3.0, 1.2, 240) * problem.risk_aversion
    frontier_points = [point(problem, 0.0)] + [point(problem, float(lam)) for lam in grid]
    sds = [item.sd_bps for item in frontier_points]
    costs = [item.expected_bps for item in frontier_points]

    figure = Figure(figsize=(8.6, 6.2), layout="constrained")
    ax = figure.subplots()
    ax.plot(sds, costs, color="#1f77b4", linewidth=2.2, label="Almgren-Chriss frontier")

    schedules = build_schedules(problem, profile)
    placements = {
        "Immediate": (0.0, 0.0),
        "TWAP": (-12, -14),
        "VWAP": (-6, 12),
        "POV": (0, -18),
        "Almgren-Chriss": (-8, 14),
    }
    for name, trades in schedules.items():
        x = to_bps(problem, cost_sd(problem, trades))
        y = to_bps(problem, expected_cost(problem, trades))
        ax.scatter(
            [x], [y], s=80, marker=MARKERS[name], color=COLORS[name], zorder=5, label=LABELS[name]
        )
        dx, dy = placements[name]
        if name == "Immediate":
            continue
        ax.annotate(
            LABELS[name],
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color=COLORS[name],
            horizontalalignment="right" if dx < 0 else "left",
        )

    immediate_cost = to_bps(problem, expected_cost(problem, schedules["Immediate"]))
    top = 1.05 * max(costs)
    ax.set_ylim(min(costs) * 0.9, top)
    ax.text(
        0.02,
        0.96,
        f"Immediate: {immediate_cost:.1f} bp at zero risk (off scale)",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["Immediate"],
        verticalalignment="top",
    )

    upgraded = lambda_for_cost_increase(problem, 1.0)
    twap_sd = to_bps(problem, cost_sd(problem, schedules["TWAP"]))
    ax.annotate(
        f"+1 bp of expected cost\nbuys {twap_sd - upgraded.sd_bps:.1f} bp of SD",
        xy=(upgraded.sd_bps, upgraded.expected_bps),
        xytext=(upgraded.sd_bps + 9.0, upgraded.expected_bps + 1.6),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.0},
        color="#333333",
    )
    ax.set_xlabel("Standard deviation of cost (bp)")
    ax.set_ylabel("Expected cost (bp)")
    ax.set_title("Cost-risk frontier: TWAP is only optimal for a risk-neutral trader")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)
    return _save(figure, path)


def sensitivity_volatility(
    problem: ExecutionProblem,
    path: Path,
    *,
    multipliers: FloatArray | None = None,
) -> Path:
    """How the frontier moves with volatility."""
    values = np.array([0.5, 1.0, 1.5, 2.0]) if multipliers is None else multipliers
    curves = volatility_curves(problem, multipliers=values, n_points=30)
    figure = Figure(figsize=(8.2, 6.0), layout="constrained")
    ax = figure.subplots()
    colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]
    for color, curve in zip(colors, curves, strict=False):
        ax.plot(
            [item.sd_bps for item in curve.points],
            [item.expected_bps for item in curve.points],
            color=color,
            linewidth=2.0,
            label=f"$\\sigma$ x{curve.parameter:g}",
        )
    twap = to_bps(problem, expected_cost(problem, optimal_trades(problem, 0.0)))
    ax.axhline(twap, color="#7f8c8d", linestyle=":", linewidth=1.0)
    ax.annotate(
        "TWAP expected cost",
        xy=(ax.get_xlim()[0], twap),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
    )
    ax.set_xlabel("Standard deviation of cost (bp)")
    ax.set_ylabel("Expected cost (bp)")
    ax.set_title("Sensitivity of the frontier to volatility")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _save(figure, path)


def sensitivity_impact(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    path: Path,
    *,
    multipliers: FloatArray | None = None,
    drift_bps: FloatArray | None = None,
) -> Path:
    """Cost against temporary impact, permanent impact and drift."""
    values = np.array([0.25, 0.5, 1.0, 2.0, 4.0]) if multipliers is None else multipliers
    gamma_values = np.array([0.5, 1.0, 2.0])
    drift_values = np.array([-30.0, -15.0, 0.0, 15.0, 30.0]) if drift_bps is None else drift_bps
    rows = impact_scenarios(
        problem, profile, eta_multipliers=values, gamma_multipliers=gamma_values
    )
    from execution_sim.sensitivity import drift_scenarios

    drift_rows = drift_scenarios(problem, profile, drifts_bps_per_day=drift_values)
    names = list(build_schedules(problem, profile))

    figure = Figure(figsize=(12.0, 4.6), layout="constrained")
    axes = figure.subplots(1, 3)

    for ax, kind, title in (
        (axes[0], "eta", "(a) Temporary impact"),
        (axes[1], "gamma", "(b) Permanent impact"),
    ):
        grid = values if kind == "eta" else gamma_values
        for name in names:
            xs: list[float] = []
            ys: list[float] = []
            for value in grid:
                label = (
                    f"eta x{value:g} / gamma x1" if kind == "eta" else f"eta x1 / gamma x{value:g}"
                )
                for row in rows:
                    if row.scenario == label and row.algorithm == name:
                        xs.append(float(value))
                        ys.append(row.expected_bps)
            ax.plot(
                xs, ys, color=COLORS[name], marker=MARKERS[name], markersize=4, label=LABELS[name]
            )
        ax.set_xscale("log")
        ax.set_xticks(grid)
        ax.set_xticklabels([f"x{value:g}" for value in grid])
        ax.minorticks_off()
        ax.set_xlabel("Multiplier on the base parameter")
        ax.set_ylabel("Expected cost (bp)")
        ax.set_title(title)
        _style(ax)

    ax = axes[2]
    for name in names:
        xs = []
        ys = []
        for value in drift_values:
            label = f"drift {value:+.1f} bp/day"
            for row in drift_rows:
                if row.scenario == label and row.algorithm == name:
                    xs.append(float(value))
                    ys.append(row.expected_bps)
        ax.plot(xs, ys, color=COLORS[name], marker=MARKERS[name], markersize=4, label=LABELS[name])
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Price drift (bp per day)")
    ax.set_ylabel("Expected cost (bp)")
    ax.set_title("(c) Drift is not in the optimiser")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)

    return _save(figure, path)


def risk_aversion(
    problem: ExecutionProblem,
    path: Path,
) -> Path:
    """How the optimal schedule responds to the risk-aversion parameter."""
    grid = np.logspace(-2.0, 2.0, 41) * problem.risk_aversion
    points = [point(problem, float(lam)) for lam in grid]
    lams = [item.risk_aversion for item in points]

    figure = Figure(figsize=(12.0, 4.2), layout="constrained")
    axes = figure.subplots(1, 3)
    axes[0].plot(
        lams,
        [100.0 * item.first_interval_fraction for item in points],
        color="#1f77b4",
        linewidth=2.0,
    )
    axes[0].set_ylabel("Executed in the first interval (%)")
    axes[0].set_title("(a) Urgency")
    axes[1].plot(lams, [item.expected_bps for item in points], color="#c0392b", linewidth=2.0)
    axes[1].set_ylabel("Expected cost (bp)")
    axes[1].set_title("(b) Cost paid for urgency")
    axes[2].plot(lams, [item.sd_bps for item in points], color="#16a085", linewidth=2.0)
    axes[2].set_ylabel("Standard deviation of cost (bp)")
    axes[2].set_title("(c) Risk avoided")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Risk aversion")
        ax.axvline(problem.risk_aversion, color="#7f8c8d", linestyle=":", linewidth=1.0)
        _style(ax)
    return _save(figure, path)


def book_dynamics(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    book: LevelBook,
    path: Path,
    *,
    refill: float = 0.25,
    refill_grid: FloatArray | None = None,
    extreme_multiple: float = 2.5,
    extreme_intervals: int = 4,
) -> Path:
    """What changes when the book is depleted instead of restored every interval."""
    grid = (
        np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
        if refill_grid is None
        else refill_grid
    )
    schedules = build_schedules(problem, profile)
    names = list(schedules)
    rows = refill_scenarios(problem, profile, book, refills=grid)

    figure = Figure(figsize=(12.5, 8.2), layout="constrained")
    axes = figure.subplots(2, 2)

    ax = axes[0, 0]
    liquidity = LiquidityModel(refill=refill)
    time_pct = problem.times[1:] / problem.horizon * 100.0
    for name, trades in schedules.items():
        execution = book_execution(problem, trades, book, liquidity)
        ax.plot(
            time_pct,
            execution.depth_after / book.capacity * 100.0,
            color=COLORS[name],
            linewidth=2.0,
            marker=MARKERS[name],
            markersize=4,
            label=LABELS[name],
        )
    ax.set_xlabel("Time (% of the horizon)")
    ax.set_ylabel("Visible depth (% of the initial ladder)")
    ax.set_title(f"(a) Depth left after each child order (refill {refill:g})")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[0, 1]
    lookup = {(row.refill, row.algorithm): row for row in rows}
    for name in names:
        values = [lookup[(float(r), name)].expected_bps for r in grid]
        ax.plot(
            grid, values, color=COLORS[name], marker=MARKERS[name], markersize=4, label=LABELS[name]
        )
    ax.set_xlim(1.02, -0.02)
    ax.set_yscale("log")
    ax.set_xlabel("Refill per interval (1 = static book, 0 = never refilled)")
    ax.set_ylabel("Expected cost (bp, log scale)")
    ax.set_title("(b) Cost against replenishment")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    _style(ax)

    ax = axes[1, 0]
    for name in ("TWAP", "VWAP", "POV"):
        gaps = [
            100.0
            * (
                lookup[(float(r), name)].objective_bps
                / lookup[(float(r), "Almgren-Chriss")].objective_bps
                - 1.0
            )
            for r in grid
        ]
        ax.plot(
            grid, gaps, color=COLORS[name], marker=MARKERS[name], markersize=4, label=LABELS[name]
        )
    ax.axhline(0.0, color=COLORS["Almgren-Chriss"], linewidth=1.4)
    ax.annotate(
        "Almgren-Chriss",
        xy=(0.5, 0.0),
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=8,
        color=COLORS["Almgren-Chriss"],
    )
    ax.set_xlim(1.02, -0.02)
    ax.set_xlabel("Refill per interval")
    ax.set_ylabel("Mean-variance cost above AC (%)")
    ax.set_title("(c) Does the optimal schedule survive depletion?")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1, 1]
    stress = problem.with_changes(
        size=extreme_multiple * problem.size, n_intervals=extreme_intervals
    )
    stress_rows = refill_scenarios(stress, profile, book, refills=np.array([refill]))
    positions = np.arange(len(names))
    heights = [
        next(row for row in stress_rows if row.algorithm == name).max_child_pct for name in names
    ]
    colors = [COLORS[name] for name in names]
    ax.barh(positions, heights, color=colors, height=0.55)
    ax.axvline(50.0, color="#c0392b", linestyle="--", linewidth=1.2, label="Half the visible depth")
    beyond = {row.algorithm: row.beyond_pct for row in stress_rows}
    for position, name in zip(positions, names, strict=True):
        if beyond[name] > 0.0:
            ax.annotate(
                f"{beyond[name]:.1f}% beyond the book",
                xy=(heights[position], position),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color="#c0392b",
                verticalalignment="center",
            )
    ax.set_xlim(0.0, max(heights) * 1.22)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_yticks(positions, [LABELS[name] for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Largest child order (% of the depth available at that moment)")
    ax.set_title(f"(d) {stress.size:,.0f} shares in {stress.n_intervals} intervals")
    _style(ax)

    return _save(figure, path)


def build_all(
    problem: ExecutionProblem,
    profile: VolumeProfile,
    book: LevelBook,
    directory: Path,
) -> list[Path]:
    """Generate every committed figure."""
    directory.mkdir(parents=True, exist_ok=True)
    return [
        cost_risk_inventory(problem, profile, book, directory / "cost_risk_inventory.png"),
        efficient_frontier(problem, profile, directory / "efficient_frontier.png"),
        sensitivity_volatility(problem, directory / "sensitivity_volatility.png"),
        sensitivity_impact(problem, profile, directory / "sensitivity_impact.png"),
        risk_aversion(problem, directory / "risk_aversion.png"),
        book_dynamics(problem, profile, book, directory / "book_dynamics.png"),
    ]
