# Derivation

Everything in `src/execution_sim` follows from the algebra below. It is written out
so that the closed form can be checked line by line, and so that the two subtleties
that bite in practice (the discrete permanent-impact correction and the fact that
TWAP is optimal only for a risk-neutral trader) are explicit.

Notation: a sell order of `X` shares is split into `N` child orders `n_1 .. n_N`,
executed at times `t_k = k·τ` with `τ = T/N`. `x_k = X − Σ_{j≤k} n_j` is the
inventory still open after interval `k`, so `x_0 = X`, `x_N = 0` and
`n_k = x_{k−1} − x_k`.

## 1. Price process

```
S_k = S_{k−1} + σ·√τ·ξ_k − γ·n_k,        ξ_k ~ N(0, 1)
S̃_k = S_{k−1} − h_k,                     h_k = s + (η/τ)·n_k
```

`S̃_k` is the average price obtained in interval `k`, `s` is the half spread,
`γ` is permanent impact (currency units per share) and `η` is temporary impact
(currency units per share, per share per day).

Two timing conventions matter:

* the child order in interval `k` executes at the mid observed **before** that
  interval's price innovation, so the first trade never trades against its own
  permanent impact;
* the permanent impact of `n_k` is applied to the price that all **later**
  intervals see.

## 2. Expected cost

The implementation shortfall is the difference between the arrival value of the
position and the cash actually received:

```
IS = X·S_0 − Σ_k n_k·S̃_k
```

Substituting the price process and using `Σ_k n_k·Σ_{j<k} ξ_j = Σ_j ξ_j·x_j`:

```
IS = −σ·√τ·Σ_j ξ_j·x_j  +  γ·Σ_k n_k·C_{k−1}  +  Σ_k n_k·h_k
```

with `C_{k−1} = Σ_{j<k} n_j`. The permanent-impact sum is a telescoping identity:

```
Σ_k n_k·C_{k−1} = ½·[(Σ_k n_k)² − Σ_k n_k²] = ½·(X² − Σ_k n_k²)
```

so the noise term has zero mean and

```
E[IS] = ½·γ·(X² − Σ n_k²) + s·X + (η/τ)·Σ n_k²
      = ½·γ·X² + s·X + (η − ½·γ·τ)/τ · Σ n_k²
```

The coefficient `η̃ = η − ½·γ·τ` is the **effective temporary impact**. It is not a
typo: a block that is large relative to the interval trades, on average, at a price
that already includes half of its own permanent impact, so the discrete model
discounts the temporary term by `½·γ·τ`. Almgren and Chriss (2000, eq. 7) obtain the
same expression.

Two consequences worth knowing:

* `η̃ > 0` is required for the objective to be convex. That is why the optimiser
  refuses problems with very large `γ` or very long intervals.
* with a single interval (`N = 1`) the permanent term vanishes entirely and the
  ½·γ·X² is only recovered as `τ → 0`. It is an artefact of the discrete timing,
  not an economic statement.

## 3. Variance

Only the noise term is random:

```
Var[IS] = σ²·τ·Σ_{k=1}^{N} x_k²
```

(interval `N` has `x_N = 0`, so the last trade carries no risk; it does not need to
be excluded explicitly).

The variance depends on the inventory path only, not on the impact parameters and
not on the drift.

## 4. The mean-variance programme

With risk aversion `λ ≥ 0`, the Almgren-Chriss objective is

```
J = E[IS] + λ·Var[IS]
  = ½·γ·X² + s·X + (η̃/τ)·Σ n_k² + λ·σ²·τ·Σ x_k²
```

The first two terms do not depend on the schedule, so the problem is

```
minimise   (η̃/τ)·Σ (x_{k−1} − x_k)² + λ·σ²·τ·Σ x_k²,    x_0 = X,  x_N = 0
```

This is a strictly convex quadratic in `x_1 .. x_{N−1}`, so the first-order
conditions are necessary and sufficient.

## 5. First-order conditions

Differentiate with respect to an interior `x_k`. Only `n_k` and `n_{k+1}` depend on
it, with `∂n_k/∂x_k = −1` and `∂n_{k+1}/∂x_k = +1`:

```
(2·η̃/τ)·[−(x_{k−1} − x_k) + (x_k − x_{k+1})] + 2·λ·σ²·τ·x_k = 0
```

which rearranges to the linear recurrence

```
−x_{k−1} + (2 + c)·x_k − x_{k+1} = 0,      c = λ·σ²·τ² / η̃
```

with boundary rows `x_0 = X` and `x_N = 0`. In matrix form it is tridiagonal and is
solved exactly by the Thomas algorithm in
[`almgren_chriss.numeric_optimal_trades`](../src/execution_sim/almgren_chriss.py).

## 6. Closed form

Guess `x_k = A·e^{κk} + B·e^{−κk}`. Substituting into the recurrence gives

```
2·cosh(κ·τ) = 2 + c      ⇒      cosh(κ·τ) = 1 + λ·σ²·τ² / (2·η̃)
```

and imposing the boundaries:

```
x_k = X · sinh(κ·(T − t_k)) / sinh(κ·T)
n_k = X · [sinh(κ·(T − t_k)) − sinh(κ·(T − t_{k−1}))] / sinh(κ·T)
```

with

```
κ = arccosh(1 + λ·σ²·τ²/(2·η̃)) / τ
```

`κ` has units of 1/time and is the urgency of the schedule: larger `κ` front-loads
the order. The implementation computes the ratio of sinh functions in exponential
form, `e^{a−b}·expm1(−2a)/expm1(−2b)`, to avoid overflow for very risk-averse
traders.

## 7. Limits and sanity checks

* `λ = 0` gives `κ = 0`, and the sinh ratio tends to `(N − k)/N`: the trades are
  equal and the schedule is **TWAP**. TWAP therefore minimises expected cost, and is
  on the frontier only at its left end.
* `λ → ∞` gives `κ → ∞` and `x_1 → 0`: the order is executed immediately.
* `σ = 0` has the same effect as `λ = 0`: no risk, so no reason to hurry.
* the closed form and the Thomas solution agree to machine precision for every
  parameter set tested (see `tests/test_almgren_chriss.py`), which is a strong check
  because the two are derived by different routes.
* the Monte Carlo simulator reproduces `E[IS]` and `Var[IS]` within sampling error
  (`tests/test_engine.py`), which validates the price process itself, not just the
  algebra.

## 8. Why TWAP is not optimal

For `λ > 0`, TWAP is feasible but is not a stationary point of `J`: perturbing it
towards the front strictly reduces `J` because the first-order effect on expected
cost is zero (TWAP is the minimum of `E`) while the effect on variance is first
order. Consequently the loss from using TWAP grows quickly with the risk aversion
the trader actually has, even though it is invisible in expected cost alone.

Because the frontier is flat near `λ = 0`, the reverse comparison is more striking:
the extra expected cost needed to move from TWAP to a much lower variance is tiny.
The report computes it as the cost of buying one basis point of risk reduction, and
the README quotes the number for the reference scenario.

## 9. The finite-depth book

The linear model is a local approximation. The book used by the simulator is a
staircase: level `j` sits `s + j·Δ` away from the mid and holds `L·d^j` shares, with
`Δ` the tick, `L` the size of the first level and `d` the decay. A child order of `q`
shares consumes whole levels and then part of the next one:

```
C(q) = Σ_{j<J} L·d^j·(s + jΔ) + (q − Q_J)·(s + JΔ),     Q_J = Σ_{j<J} L·d^j
```

`C(q)` is convex and piecewise linear in `q`, with slopes `s + jΔ` that increase
with depth. For orders that stay inside the first level the average price is flat,
which is exactly the difference between a lumpy book and a smooth model.

The reference scenario calibrates `η` so that the linear model reproduces the book's
slippage at the average child order (`scenario.eta_matching_book`). Away from that
size the two disagree, and the disagreement is the point: in the reference
scenario the book costs 84% more than the linear model when the whole order is sent
as one block.

The book is refilled after every interval. Intraday liquidity depletion, queue
position and transient impact decay are out of scope.
