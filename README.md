# execution-simulator

[![CI](https://github.com/IkerAbad/execution-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/IkerAbad/execution-simulator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/execution-simulator.svg)](https://pypi.org/project/execution-simulator/)
[![Release](https://img.shields.io/github/v/release/IkerAbad/execution-simulator?display_name=tag)](https://github.com/IkerAbad/execution-simulator/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

Execution cost simulator for large orders. It compares immediate execution, TWAP,
VWAP, POV and the Almgren-Chriss optimal schedule under an explicit impact model,
against a finite-depth order book, and measures the **total cost of the order**, not
its average price.

The point is the last two words: two schedules can show the same average execution
price and very different total cost once impact, spread and the risk of holding the
position are accounted for.

## What is modelled

A liquidation of `X` shares over a horizon `T`, sliced into `N` intervals of length
`τ = T/N`. The mid price and the execution price of interval `k` are

```
S_k  = S_{k−1} + σ·√τ·ξ_k − γ·n_k            permanent impact + noise
S̃_k  = S_{k−1} − (s + η·n_k/τ)                half spread + temporary impact
```

with `σ` the volatility per √time, `γ` the permanent impact per share, `η` the
temporary impact and `s` the half spread. The implementation shortfall of the
schedule `n_1 .. n_N` then has closed-form moments

```
E[IS]  = ½·γ·X² + s·X + (η − ½·γ·τ)/τ · Σ n_k²
Var[IS] = σ²·τ·Σ x_k²
```

where `x_k` is the inventory still open after interval `k`. All the algebra, including
the discrete `½·γ·τ` correction, is in [`docs/derivation.md`](docs/derivation.md).

The reference scenario is a 100,000-share sell order at 50 currency units (5,000,000
notional), one trading day in 20 intervals, 2% daily volatility, and a book with
2,000 shares at the touch and thinning levels one cent apart. The linear `η` is
calibrated to reproduce the book's slippage at the average child order, so the two
execution models agree at the size the analytics are meant to describe.

The book can be static (restored every interval, `refill = 1`) or depleting
(`refill < 1`), and a child order larger than the visible ladder is priced by
extrapolating the ladder past its deepest level. Both modes are compared below with
the same market parameters.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -e ".[dev]"
python -m execution_sim compare
```

```
algorithm         E[cost] (bp)     SD (bp)    E+lam*Var (bp)
------------------------------------------------------------
Immediate               33.160       0.000            33.160
TWAP                     4.508     111.131            10.683
VWAP                     4.641     107.500            10.419
POV                      5.234      80.389             8.465
Almgren-Chriss           5.766      67.172             8.022
```

Regenerate every number and figure in this README:

```bash
python -m execution_sim report --out results --figures figures --paths 20000 --seed 7
```

## The five algorithms

| Algorithm | What it does | Where it comes from |
|---|---|---|
| Immediate | Sends the whole order in the first interval | Baseline, zero risk, maximum impact |
| TWAP | Equal shares per interval | Minimum expected cost when impact is linear |
| VWAP | Shares proportional to a synthetic U-shaped volume profile | Tracks the market's volume |
| POV | Fixed participation rate of market volume, horizon endogenous | Rate algorithm, finishes early |
| Almgren-Chriss | Minimises `E[IS] + λ·Var[IS]` | Closed form, derived by hand |

POV runs at 1.5 times the neutral rate `X / daily_volume`, so it completes at 0.75 of
the horizon. At exactly the neutral rate POV collapses into the VWAP schedule; there
is a test for that identity.

## Results

Reference scenario, risk aversion `λ = 10⁻⁶` (implied `κ = 3.64 /day`):

| Algorithm | E[cost] (bp) | SD (bp) | E + λ·Var (bp) | First interval (%) | Completion (d) |
|---|---|---|---|---|---|
| Immediate | 33.160 | 0.000 | 33.160 | 100.0 | 0.05 |
| TWAP | 4.508 | 111.131 | 10.683 | 5.0 | 1.00 |
| VWAP | 4.641 | 107.500 | 10.419 | 7.9 | 1.00 |
| POV | 5.234 | 80.389 | 8.465 | 11.8 | 0.75 |
| Almgren-Chriss | 5.766 | 67.172 | 8.022 | 16.7 | 1.00 |

Units are basis points of the 5,000,000 notional. The Almgren-Chriss row minimises the
mean-variance objective; every other row pays an excess of between 0.44 and 25 bp of
risk-adjusted cost. Note that the optimal schedule does **not** have the lowest
expected cost: it trades expected cost for a large reduction in dispersion.

![Cost, risk, inventory and book slippage](https://raw.githubusercontent.com/IkerAbad/execution-simulator/main/figures/cost_risk_inventory.png)

## TWAP is not optimal

TWAP minimises expected cost. That is a theorem under linear impact: since

```
E[IS] = constants + (η − ½γτ)/τ · Σ n_k²
```

and `Σ n_k²` is minimised by equal trades subject to `Σ n_k = X`. It is also the
`λ → 0` endpoint of the frontier, so it is not dominated in the (risk, cost) plane
either.

The catch is that it is optimal only for a **risk-neutral** trader. Any `λ > 0` makes
the objective

```
J = E[IS] + λ·Var[IS]
```

and TWAP is not a stationary point of `J`: the first-order effect on expected cost is
zero (it is a minimum of `E`), while the first-order effect on variance is not. The
loss is easy to quantify with the frontier:

| Risk aversion | E TWAP (bp) | E AC (bp) | ΔE (bp) | SD TWAP (bp) | SD AC (bp) | ΔSD (bp) | J TWAP (bp) | J AC (bp) | Loss (%) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.508 | 4.508 | 0.000 | 111.131 | 111.131 | 0.000 | 4.508 | 4.508 | 0.00 |
| 10⁻⁷ | 4.508 | 4.554 | 0.046 | 111.131 | 101.875 | −9.255 | 5.125 | 5.073 | 1.02 |
| 10⁻⁶ | 4.508 | 5.766 | 1.258 | 111.131 | 67.172 | −43.959 | 10.683 | 8.022 | **24.91** |
| 10⁻⁵ | 4.508 | 11.344 | 6.836 | 111.131 | 30.750 | −80.381 | 66.258 | 16.072 | **75.74** |
| 10⁻⁴ | 4.508 | 23.304 | 18.796 | 111.131 | 8.906 | −102.225 | 622.008 | 27.270 | **95.62** |

Read the middle row: by paying 1.26 bp more in expected cost, the optimal schedule
removes 44 bp of standard deviation and cuts the risk-adjusted cost by 25%. For a
trader with `λ = 10⁻⁵` the same TWAP schedule is 76% worse than necessary.

The frontier is flat near `λ = 0`, so the argument is even sharper when it is made
without choosing a `λ` at all: **one extra basis point of expected cost buys 39.8 bp
of risk reduction**, moving the standard deviation from 111.1 bp to 71.3 bp. That is
0.025 bp of expected cost per bp of standard deviation. Ignoring risk is cheap only
if you never look at the second moment.

![Efficient frontier](https://raw.githubusercontent.com/IkerAbad/execution-simulator/main/figures/efficient_frontier.png)

## Finite liquidity

The linear model is a local approximation. The simulator walks a real ladder and
charges each level its own price, so a child order that fits in the touch pays the
touch and a block pays the whole book. At the average child order (5,000 shares) the
calibrated linear model and the book agree to the cent. At the full order as one
block the book is **84% more expensive** than the linear model, and the gap keeps
growing with size.

That is the difference between "average price" and "total cost": the same volume
traded in three child orders instead of one is cheaper by a factor that the average
price of the parent order does not show.

## Does the book refill?

Everything above assumes that the ladder is restored between intervals. That is the
optimistic extreme. The simulator also runs a **depleting book**: after each child
order only a fraction `refill` of the missing depth comes back, so a large order eats
the liquidity it needs and pays for it later.

With `refill = 0.25` (a quarter of the gap per interval), the same five schedules
cost:

| Algorithm | Static book (bp) | Running book (bp) | Change | Max child / depth | Min depth left |
|---|---|---|---|---|---|
| Immediate | 61.065 | 61.065 | 0.0% | 52.6% | 47.4% |
| TWAP | 4.508 | 11.317 | +151.0% | 2.9% | 89.5% |
| VWAP | 4.686 | 11.332 | +141.8% | 4.5% | 87.4% |
| POV | 5.640 | 13.794 | +144.6% | 6.2% | 85.9% |
| Almgren-Chriss | 6.542 | 14.743 | +125.4% | 8.8% | 82.5% |

Costs rise by 125% to 151%. More interesting than the level is the structure, and it
shows up at the limit.

### With no replenishment, slicing buys nothing

Set `refill = 0` and the ladder never comes back. The temporary cost of the five
schedules then becomes **identical to machine precision** (4·10⁻¹¹ bp between them):
a static ladder has no memory, so the cost of buying 100,000 shares in twenty pieces
is exactly the cost of buying them in one, and only the discrete permanent-impact
term differs. The whole benefit of slicing comes from the book refilling.

![The book does not refill for free](https://raw.githubusercontent.com/IkerAbad/execution-simulator/main/figures/book_dynamics.png)

The consequence for the mean-variance objective is blunt. At the model's risk
aversion, the advantage of the Almgren-Chriss schedule over TWAP falls from **17.6%**
in the static book to **2.8%** at `refill = 0.25` and **2.7%** at `refill = 0.20`:
the cost of speed rises while the risk penalty stays the same, so front-loading pays
much less. At `refill = 0` immediate execution wins outright (61.07 bp of objective
against 65.14 for AC and 69.14 for TWAP), because impact is path independent and the
block carries no risk at all.

Two honest qualifications:

* TWAP remains the worst of the *sliced* schedules at every refill level in the
  sweep. What collapses with a depleting book is the case for slicing at all, not
  the case against TWAP.
* Between `refill = 0.3` and `refill = 0.7` the **POV schedule beats the AC one** by
  up to 0.13 bp (1.0% of the objective). The AC schedule minimises the linear model's
  objective, not the running book's, so this is the model risk of optimising against
  the wrong liquidity model, and it is published rather than hidden.

## Extreme case: a child order larger than half the visible depth

A 250,000-share order (2.5 times the reference order, 131% of the visible ladder)
completed in four intervals, with `refill = 0.25`:

| Algorithm | Max child / depth | Beyond the visible book | Static (bp) | Running (bp) | Change |
|---|---|---|---|---|---|
| Immediate | 131.4% | 23.9% | 415.81 | 415.81 | 0.0% |
| TWAP | 76.4% | 0.0% | 38.72 | 119.42 | +208.4% |
| VWAP | 81.7% | 0.0% | 41.59 | 118.57 | +185.1% |
| POV | 87.3% | 0.0% | 58.52 | 139.08 | +137.6% |
| Almgren-Chriss | 84.0% | 0.0% | 87.20 | 129.95 | +49.0% |

What happens to each one:

* **Immediate** empties the ladder in the first interval: 190,192 shares of visible
  depth, 59,808 shares (23.9% of the order) priced by extrapolating the ladder past
  its deepest level. Cost: 415.8 bp, unchanged by the refill parameter, because a
  single block has no second interval to be penalised in.
* **TWAP** never goes past the last level, but ends the day with 10% of the ladder
  left and pays 119.4 bp instead of 38.7 bp: the same schedule, three times the cost.
* **VWAP** is the cheapest of the sliced schedules in this stress (118.6 bp): its
  profile puts 31% of the order in the first interval, when the book is still full,
  and only 19% in the middle of the horizon, when it is thin.
* **POV** finishes the day with 4.8% of the ladder and pays 139.1 bp.
* **Almgren-Chriss** front-loads, uses the cheap levels early and then trades little,
  so it is the least damaged in relative terms (+49%), although its absolute cost is
  not the lowest: it is no longer the optimum of this model.

The book never refuses to trade: past the deepest visible level it continues at the
same tick spacing with the size of the deepest level, and the fraction priced that
way is reported. A strict mode (`LevelBook.walk(..., extrapolate=False)`) raises
`InsufficientDepthError` instead, which is what a venue would do.


## Sensitivity

Temporary impact is the parameter that punishes immediacy: multiplying `η` by four
takes immediate execution from 33 bp to 130 bp while the sliced schedules move by a
few basis points. Permanent impact moves the sliced schedules almost linearly, and it
does not touch immediate execution at all, because in this discrete convention the
first trade never trades against its own permanent impact (see the derivation).
Volatility shifts the whole frontier to the right without changing its shape. A drift
that the optimiser ignores moves the ranking: with a rising price the slow schedules
gain, because waiting is rewarded, and the expected-cost ordering inverts.

![Sensitivity to impact and drift](https://raw.githubusercontent.com/IkerAbad/execution-simulator/main/figures/sensitivity_impact.png)
![Sensitivity to volatility](https://raw.githubusercontent.com/IkerAbad/execution-simulator/main/figures/sensitivity_volatility.png)
![Sensitivity to risk aversion](https://raw.githubusercontent.com/IkerAbad/execution-simulator/main/figures/risk_aversion.png)

## How the optimum is obtained

No external optimisation library. The first-order conditions of the discrete problem
form a tridiagonal linear system, which is solved with a hand-written Thomas
algorithm; the closed form is obtained by solving the Euler-Lagrange recurrence

```
−x_{k−1} + (2 + c)·x_k − x_{k+1} = 0,      c = λ·σ²·τ²/η̃
```

with a hyperbolic ansatz, giving

```
x_k = X · sinh(κ·(T − t_k)) / sinh(κ·T),      cosh(κ·τ) = 1 + λ·σ²·τ²/(2·η̃)
```

Both routes are implemented and must agree. See
[`docs/derivation.md`](docs/derivation.md) for the full derivation.

## Validation

* **Closed form vs numerical solution.** `tests/test_almgren_chriss.py` compares the
  sinh solution with the Thomas solution for 25 combinations of risk aversion and
  number of intervals. Maximum difference: `2.9·10⁻¹¹` shares on a 100,000-share
  order.
* **Optimality.** The test suite also checks the Euler-Lagrange residual point by
  point, verifies that no feasible perturbation lowers the objective, and runs an
  independent coordinate-descent solver that converges to the same path.
* **Analytic moments vs simulation.** `tests/test_engine.py` runs 40,000 Monte Carlo
  paths per algorithm with linear impact and no drift, and checks the mean against
  `E[IS]` (within five standard errors) and the standard deviation against
  `σ·√(τ·Σx²)` (within 1%). In the reference report the z-scores stay within
  ±1.9 and the dispersion error within 0.7%.
* **The book.** Level consumption is tested exactly, the calibrated small-order
  slippage matches the linear model at the touch, and the cost function is checked to
  be convex in the child order size.
* **The running book.** `tests/test_liquidity.py` checks that `refill = 1` reproduces
  the static book to 10⁻¹², that the variance is unchanged by depletion (the book is
  deterministic), that the simulated mean matches `expected_execution_cost(..., book,
  liquidity)` within five standard errors for all five schedules at `refill = 0.25`,
  and that the temporary cost is path independent at `refill = 0`. The analytic
  validation of the Almgren-Chriss solution is untouched by all of this: it runs on
  the linear model, where the closed form is exact.

`ruff`, `mypy --strict` and `pytest` run in CI on every push, together with a job that
regenerates the report and the figures and uploads them as artifacts.

## Repository layout

```
src/execution_sim/
├── core.py              model parameters and the execution problem
├── impact.py            closed-form expected cost and variance
├── almgren_chriss.py    closed form, Thomas solution, kappa, half-life
├── schedules.py         immediate, TWAP, VWAP, POV, the algorithm registry
├── book.py              finite-depth ladder, running book and level consumption
├── volume.py            synthetic U-shaped intraday volume
├── engine.py            Monte Carlo simulator
├── frontier.py          cost-risk frontier and the price of risk
├── sensitivity.py       parameter and volatility sweeps
├── report.py            the tables quoted in this README
├── figures.py           every figure in this README
└── cli.py               `execution-sim`
tests/                   analytic, book, engine, frontier, sensitivity, figures, CLI
docs/derivation.md       the algebra, by hand
results/summary.md       the generated report
figures/                 the generated figures
```

## Limitations

* **No real data.** Prices, volumes and impact parameters are synthetic. Nothing here
  is calibrated to a specific venue, and the simulator does not connect to any
  exchange.
* **Linear impact for the analytics.** The closed form assumes linear permanent and
  temporary impact and no drift. With a drift the Almgren-Chriss schedule is a
  plug-in rule, not the exact optimum.
* **The book has no memory of queues.** Liquidity can be depleted and replenished
  (`LiquidityModel`), but there is no queue position, no transient impact decay, and
  replenishment is exogenous: it does not react to prices or to volatility.
* **Gaussian prices.** No jumps, no stochastic volatility, no cross-impact between
  assets.
* **Risk is measured as variance**, so the model is symmetric and risk-neutral in the
  tails. Expected shortfall would tell a different story.

## License

MIT. See [LICENSE](LICENSE).
