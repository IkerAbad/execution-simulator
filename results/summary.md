# Execution report

## Parameters

| Parameter | Value |
|---|---|
| order size (shares) | 100,000 |
| notional (currency) | 5,000,000 |
| horizon (days) | 1 |
| intervals | 20 |
| tau (days) | 0.05 |
| price | 50 |
| sigma (per sqrt(day)) | 1 |
| gamma (per share) | 2e-07 |
| eta (per share per day) | 8.04e-08 |
| half spread (bps) | 1 |
| risk aversion | 1e-06 |
| kappa (1/day) | 3.6368 |
| daily volume (shares) | 2,000,000 |
| book levels | 300 |
| book capacity (shares) | 190,192 |
| simulated paths | 20,000 |

## Algorithms

| Algorithm | E[cost] (bp) | SD (bp) | E + lambda*Var (bp) | Excess vs AC (bp) | First interval (%) | Half-life (d) | Completion (d) |
|---|---|---|---|---|---|---|---|
| Immediate | 33.160 | 0.000 | 33.160 | 25.138 | 100.0 | 0.025 | 0.050 |
| TWAP | 4.508 | 111.131 | 10.683 | 2.661 | 5.0 | 0.500 | 1.000 |
| VWAP | 4.641 | 107.500 | 10.419 | 2.397 | 7.9 | 0.500 | 1.000 |
| POV | 5.234 | 80.389 | 8.465 | 0.443 | 11.8 | 0.274 | 0.750 |
| Almgren-Chriss | 5.766 | 67.172 | 8.022 | 0.000 | 16.7 | 0.191 | 1.000 |

## TWAP versus the optimal schedule

| Risk aversion | E TWAP (bp) | E AC (bp) | dE (bp) | SD TWAP (bp) | SD AC (bp) | dSD (bp) | J TWAP (bp) | J AC (bp) | J reduction (%) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.508 | 4.508 | 0.000 | 111.131 | 111.131 | 0.000 | 4.508 | 4.508 | 0.00 |
| 1e-07 | 4.508 | 4.554 | 0.046 | 111.131 | 101.875 | -9.255 | 5.125 | 5.073 | 1.02 |
| 1e-06 | 4.508 | 5.766 | 1.258 | 111.131 | 67.172 | -43.959 | 10.683 | 8.022 | 24.91 |
| 1e-05 | 4.508 | 11.344 | 6.836 | 111.131 | 30.750 | -80.381 | 66.258 | 16.072 | 75.74 |
| 0.0001 | 4.508 | 23.304 | 18.796 | 111.131 | 8.906 | -102.225 | 622.008 | 27.270 | 95.62 |

## Price of risk

Starting from TWAP (4.508 bp expected, 111.131 bp SD), spending 1.0 bp of expected cost buys 39.803 bp of risk reduction: 0.025 bp of expected cost per bp of SD (risk aversion 8.04e-07, kappa 3.2610/day).

## Validation against the closed form

| Algorithm | E analytic (bp) | E simulated (bp) | SD analytic (bp) | SD simulated (bp) | z-score | SD error (%) |
|---|---|---|---|---|---|---|
| Immediate | 33.160 | 33.160 | 0.000 | 0.000 | 0.00 | 0.00 |
| TWAP | 4.508 | 4.193 | 111.131 | 111.845 | -0.40 | 0.64 |
| VWAP | 4.641 | 3.239 | 107.500 | 107.262 | -1.85 | -0.22 |
| POV | 5.234 | 5.281 | 80.389 | 80.527 | 0.08 | 0.17 |
| Almgren-Chriss | 5.766 | 6.159 | 67.172 | 66.697 | 0.83 | -0.71 |

Maximum difference between the closed-form schedule and the numerical solution of the first-order conditions: 2.91e-12 bp of size.
Book slippage minus the linear model at the mean chunk (+0.00%).
At the full order as a single block the same gap is +84.2%: the linear model is a local approximation.

## Sensitivity to impact parameters

| Scenario | E Immediate (bp) | E TWAP (bp) | E VWAP (bp) | E POV (bp) | E Almgren-Chriss (bp) |
|---|---|---|---|---|---|
| eta x0.5 / gamma x0.5 | 17.080 | 2.754 | 2.820 | 3.117 | 3.927 |
| eta x0.5 / gamma x1 | 17.080 | 3.704 | 3.766 | 4.043 | 4.861 |
| eta x0.5 / gamma x2 | 17.080 | 5.604 | 5.657 | 5.895 | 6.721 |
| eta x1 / gamma x0.5 | 33.160 | 3.558 | 3.695 | 4.308 | 4.816 |
| eta x1 / gamma x1 | 33.160 | 4.508 | 4.641 | 5.234 | 5.766 |
| eta x1 / gamma x2 | 33.160 | 6.408 | 6.532 | 7.086 | 7.666 |
| eta x2 / gamma x0.5 | 65.320 | 5.166 | 5.444 | 6.690 | 6.290 |
| eta x2 / gamma x1 | 65.320 | 6.116 | 6.390 | 7.615 | 7.246 |
| eta x2 / gamma x2 | 65.320 | 8.016 | 8.281 | 9.467 | 9.156 |

## Sensitivity to volatility

| Curve | SD range (bp) | Cost range (bp) |
|---|---|---|
| sigma x0.5 | 10.173 to 55.565 | 4.508 to 15.496 |
| sigma x1 | 8.906 to 111.131 | 4.508 to 23.304 |
| sigma x1.5 | 7.213 to 166.696 | 4.508 to 27.334 |
| sigma x2 | 5.898 to 222.261 | 4.508 to 29.436 |

## Sensitivity to drift

| Scenario | E Immediate (bp) | E TWAP (bp) | E VWAP (bp) | E POV (bp) | E Almgren-Chriss (bp) |
|---|---|---|---|---|---|
| drift -20.0 bp/day | 33.160 | 14.008 | 14.141 | 10.965 | 10.498 |
| drift +0.0 bp/day | 33.160 | 4.508 | 4.641 | 5.234 | 5.766 |
| drift +20.0 bp/day | 33.160 | -4.992 | -4.859 | -0.497 | 1.035 |
