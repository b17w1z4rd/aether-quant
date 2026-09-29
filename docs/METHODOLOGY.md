# Model specification

## 1. Normalize the option chain

With spot S, constant continuously compounded rate r and dividend yield q, maturity T = days/365:

```math
F_T = S e^{(r-q)T}, \qquad D_T=e^{-rT}, \qquad
x=K/F_T, \qquad c=C/(D_TF_T).
```

European call bounds become max(1-x,0) ≤ c < 1. Midquote IV is inferred by bracketed root finding in the normalized Black–Scholes call formula. Values effectively at intrinsic receive a small positive IV floor; values beyond the solver's 800% volatility bracket are rejected. The chain must contain European calls and compatible deterministic proportional carry. Discrete dividends and American exercise are not handled.

## 2. Split before fitting

Within each expiry, hold out every fourth interior quote, beginning at sorted index 2. Endpoints stay in training. The models fit only training quote values. The shared grid uses the intersection of expiry moneyness ranges. Heldout quotes outside this intersection are excluded from every comparison and counted in the report.

This fixed split evaluates interpolation in the same snapshot. Quote coordinates define the evaluation domain; heldout targets do not influence fitting. A regression test alters heldout targets and verifies that both the raw and projected surfaces remain identical. Hyperparameters, SVI restart choices, and hedge quantities are not selected on evaluation values.

## 3. Learn log total variance

The GP inputs are (log x, log T); the target is log w, where w = IV² T. The kernel is a learned constant times an anisotropic RBF plus a learned white-noise kernel. Training targets are standardized by scikit-learn. One bounded marginal-likelihood optimization is used, with captured convergence warnings.

For posterior predictive mean μ and standard deviation s in log-variance space:

```math
\hat\sigma(x,T)=\sqrt{e^{\mu(x,T)}/T},\qquad
[\sigma_L,\sigma_U]=\sqrt{e^{\mu\mp1.96s}/T}.
```

The point estimate is a transformed median, not the mean of a lognormal variable. White-kernel noise contributes to the predictive band. The band assumes the fitted GP likelihood and has not been calibrated for empirical coverage. It is displayed only for the raw model.

The exact GP has cubic training cost, so the implementation is aimed at small research snapshots. There is no quote downsampling or scalable sparse-GP approximation. Bid/ask width is not used as a training weight.

## 4. Fit the SVI benchmark

For each expiry independently, raw SVI represents total variance as:

```math
w(k)=a+b\left[\rho(k-m)+\sqrt{(k-m)^2+s^2}\right], \qquad k=\log x.
```

Bounded least squares minimizes training total-variance residuals. Parameterizing the positive minimum variance as v gives a = v - bs√(1-ρ²), so a may be negative while total variance remains positive. Three initial m values (-0.1, 0, 0.1) are tried; the smallest training residual chooses the result. Bounds and solver settings are in `surface.py`. The GP and SVI minimize different training objectives, which is disclosed rather than treated as a controlled likelihood comparison.

This benchmark is not the SSVI arbitrage-free parameterization. Butterfly and calendar constraints are not enforced. Parameters, convergence flags, and training costs are exported.

## 5. Project normalized call prices

Evaluate the GP on a shared 31-strike grid at the observed expiries. Convert IV to normalized call prices and minimize:

```math
\min_c \frac12\|c-c_{GP}\|_2^2.
```

Linear inequalities enforce:

1. Intrinsic lower bound and upper bound of 1.
2. Strike-cell slopes between -1 and 0.
3. Nondecreasing strike-cell slopes (convexity).
4. Nondecreasing normalized call values with maturity at fixed x.

SLSQP starts at a feasible constant-30%-volatility surface and uses an analytic objective gradient. Every grid point receives equal weight; GP uncertainty does not weight the projection. A successful solver status and all diagnostic violations ≤ 1e-6 are required. The calendar condition relies on deterministic proportional carry and the normalized forward process.

The implied-volatility grid is recovered from repaired call values. Heldout projected-price evaluation linearly interpolates **normalized call prices**. Stress repricing linearly interpolates **IV** in maturity and forward moneyness instead. These are different operations, and off-grid IV interpolation is not guaranteed to preserve the price constraints. The projection does not enforce bid/ask envelopes, continuous-domain conditions, or tail boundary conditions.

## 6. Specify scenario risk

The five-calendar-day return is generated from a unit-variance Student-t variate with six degrees of freedom, annual volatility 25%, and a jump added to log return with probability 1.5%. Jump log size is normal with mean -4% and standard deviation 5%. Parallel IV shocks equal -0.55 times the simple spot return plus independent normal noise with standard deviation 0.018.

This is a transparent stress distribution, not a calibrated return model or an exact martingale. Black–Scholes marks provide option values; the physical scenario distribution need not match those pricing assumptions. Changing the assumptions can reverse the apparent benefits of a hedge.

All positions are repriced after five calendar days using a sticky-forward-moneyness IV surface plus the parallel shock, with an IV floor of 1%. Strikes remain fixed in currency. The initial premium is financed at the given risk-free rate. Multiplier is 100. Entry points outside the grid domain are rejected. Scenario points outside the domain use the nearest boundary coordinate, with affected scenario counts exported. Dashboard stress-grid clamping is counted separately.

## 7. Minimize expected shortfall

Let Bᵢ be the unhedged portfolio P&L, Hᵢⱼ the incremental P&L of one hedge unit after flat execution friction, and wⱼ its continuous long-only quantity. Loss is Lᵢ(w) = -Bᵢ - Σⱼ Hᵢⱼwⱼ. At confidence α = 97.5%, solve:

```math
\min_{w,\eta,u}\ \eta+\frac{1}{N(1-\alpha)}\sum_i u_i
```

subject to uᵢ ≥ Lᵢ(w)-η, uᵢ ≥ 0, 0 ≤ wⱼ ≤ 5, and Σⱼ pⱼwⱼ ≤ budget. Here pⱼ is positive entry premium plus one unit of execution friction. A sparse linear program is solved with SciPy/HiGHS. All supplied hedge candidates must represent one long option.

Train on 1,500 scenarios, then hold weights fixed and evaluate on 4,000 independently seeded scenarios. Evaluation computes empirical expected shortfall using the empirical quantile and fractional tail mass, including non-integer tail sizes. The same evaluation sample is reused across budget points to make their comparison less noisy. It is a sensitivity display, not a budget-selection procedure.

The objective can prefer added long exposure with favorable outcomes under the assumed distribution. It does not distinguish such exposure from economically desirable hedging. No margin constraint, integer contract optimization, liquidation spread, impact, or robust ambiguity set is present. A lower simulated ES is not evidence of a better live strategy.

## Further reading

- [scikit-learn: Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [Gatheral and Jacquier: Arbitrage-free SVI volatility surfaces](https://arxiv.org/abs/1204.0646)
- [Rockafellar and Uryasev: Optimization of Conditional Value-at-Risk](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)
- [SciPy: SLSQP minimization](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

These references explain the underlying methods. They do not imply that this finite-grid research implementation provides the stronger guarantees of the referenced work.
