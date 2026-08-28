// Regression Analysis - Lecture Notes
// MA2003B - Application of Multivariate Methods in Data Science
// Tecnológico de Monterrey

#let info(body) = block(
  fill: rgb("#E8F4F8"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [
    #text(weight: "bold", fill: rgb("#1E88E5"))[Info: ]
    #body
  ]
)

#let warning(body) = block(
  fill: rgb("#FFF3E0"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [
    #text(weight: "bold", fill: rgb("#F57C00"))[Warning: ]
    #body
  ]
)

#let tip(body) = block(
  fill: rgb("#E8F5E9"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [
    #text(weight: "bold", fill: rgb("#43A047"))[Tip: ]
    #body
  ]
)

#let example(body) = block(
  fill: rgb("#F3E5F5"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [
    #text(weight: "bold", fill: rgb("#8E24AA"))[Example: ]
    #body
  ]
)

#set document(
  title: "Regression Analysis - Lecture Notes",
  author: "MA2003B",
  date: auto,
)

#set page(
  paper: "us-letter",
  margin: (x: 1.5cm, y: 2cm),
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
)

#set heading(numbering: "1.1")

#show heading.where(level: 1): it => [
  #set text(18pt, weight: "bold")
  #block(above: 1.5em, below: 1em)[#it]
]

#show heading.where(level: 2): it => [
  #set text(14pt, weight: "bold")
  #block(above: 1.2em, below: 0.8em)[#it]
]

#align(center)[
  #text(24pt, weight: "bold")[Regression Analysis]

  #text(16pt)[Comprehensive Study Guide & Theoretical Notes]

  #v(0.5em)

  #text(12pt)[MA2003B — Application of Multivariate Methods in Data Science]

  #v(0.5em)

  #text(11pt)[Tecnológico de Monterrey]

  #v(2em)
]

#pagebreak()

= Course and Chapter Overview

== Course Context
In the *MA2003B* curriculum, regression analysis serves as the essential stepping stone connecting univariate inference to high-dimensional multivariate models. This chapter develops both classical Ordinary Least Squares (OLS) foundation and modern diagnostics for building predictive statistical relationships.

== Learning Objectives
By the end of this module, students will be able to:
- Derive and interpret Simple and Multiple Linear Regression models.
- Decompose sums of squares via ANOVA tables and evaluate overall and partial $F$-tests.
- Construct and contrast confidence intervals for mean response vs. prediction intervals for new observations.
- Conduct residual diagnostic procedures to evaluate normality, linearity, and homoskedasticity.
- Implement forward selection, backward elimination, and information-criterion-based model searches (AIC, BIC).
- Incorporate non-linear polynomial terms and logarithmic transformations.
- Identify multi-collinearity using Variance Inflation Factors (VIF).
- Detect high-leverage points and influential outliers using Cook's Distance and Studentized residuals.
- Address heteroskedasticity through Weighted Least Squares (WLS) and robust standard errors (HC3).

---

= 1.1 Simple Linear Regression

== Mathematical Formulation
Simple Linear Regression models the linear association between a continuous scalar response variable $Y$ and a single explanatory predictor variable $X$:

$ y_i = beta_0 + beta_1 x_i + epsilon_i, quad i = 1, 2, dots.h, n $

where:
- $beta_0 in bb(R)$ is the population intercept parameter.
- $beta_1 in bb(R)$ is the population slope coefficient (expected marginal change in $Y$ per unit change in $X$).
- $epsilon_i$ is the unobserved random error term satisfying the classical Gauss-Markov assumptions:
  $ bb(E)[epsilon_i] = 0, quad op("Var")(epsilon_i) = sigma^2, quad op("Cov")(epsilon_i, epsilon_j) = 0 \ (i eq.not j) $

Under the normality assumption, $epsilon_i attach(tilde.op, t: "iid") cal(N)(0, sigma^2)$.

== Ordinary Least Squares (OLS) Derivation
The OLS estimators $(hat(beta)_0, hat(beta)_1)$ minimize the Residual Sum of Squares:

$ S(beta_0, beta_1) = sum_(i=1)^n (y_i - beta_0 - beta_1 x_i)^2 $

Setting partial derivatives to zero yields the normal equations:
$ frac(partial S, partial beta_0) = -2 sum_(i=1)^n (y_i - beta_0 - beta_1 x_i) = 0 ==> hat(beta)_0 = overline(y) - hat(beta)_1 overline(x) $
$ frac(partial S, partial beta_1) = -2 sum_(i=1)^n x_i (y_i - beta_0 - beta_1 x_i) = 0 ==> hat(beta)_1 = frac(sum_(i=1)^n (x_i - overline(x))(y_i - overline(y)), sum_(i=1)^n (x_i - overline(x))^2) = frac(S_(x y), S_(x x)) $

#info[
  *Gauss-Markov Theorem:* Under the assumptions of linearity, strict exogeneity, homoskedasticity, and no autocorrelation, the OLS estimator $hat(bold(beta))$ is the *Best Linear Unbiased Estimator (BLUE)*.
]

---

= 1.2 ANOVA and Confidence Intervals for Predictions

== Sum of Squares Decomposition
The total sample variation in $Y$ is partitioned into explained (regression) and unexplained (residual) components:

$ underbrace(sum_(i=1)^n (y_i - overline(y))^2, "SST (Total)") = underbrace(sum_(i=1)^n (hat(y)_i - overline(y))^2, "SSR (Regression)") + underbrace(sum_(i=1)^n (y_i - hat(y)_i)^2, "SSE (Error)") $

Degrees of freedom: $"df"_("Total") = n - 1$, $"df"_("Regression") = 1$ (or $k$ predictors), $"df"_("Error") = n - k - 1$.

#table(
  columns: (2fr, 1.5fr, 1.5fr, 1.5fr, 1.5fr),
  align: center,
  table.header([*Source*], [*Sum of Squares*], [*Degrees of Freedom*], [*Mean Square (MS)*], [*$F$-Statistic*]),
  [Regression], [SSR], [$k$], [$"MSR" = "SSR"/k$], [$F = frac("MSR", "MSE")$],
  [Residual (Error)], [SSE], [$n - k - 1$], [$"MSE" = "SSE"/(n - k - 1)$], [],
  [Total], [SST], [$n - 1$], [], []
)

Coefficient of determination:
$ R^2 = frac("SSR", "SST") = 1 - frac("SSE", "SST") $

== Mean Response Confidence Interval vs. Individual Prediction Interval
For a specified predictor value $x_0$:

*1. Confidence Interval for the Expected Mean $bb(E)[Y | X = x_0]$:*
$ hat(y)_0 plus.minus t_(n-2, 1 - alpha/2) dot.op s sqrt(frac(1, n) + frac((x_0 - overline(x))^2, S_(x x))) $

*2. Prediction Interval for a Single New Observation $Y_("new")$:*
$ hat(y)_0 plus.minus t_(n-2, 1 - alpha/2) dot.op s sqrt(1 + frac(1, n) + frac((x_0 - overline(x))^2, S_(x x))) $

#tip[
  Prediction intervals are always strictly wider than confidence intervals because they account for both model parameter uncertainty and individual observation variance $sigma^2$.
]

---

= 1.3 Residual Analysis & Condition of Normality

== Residual Definitions
- *Raw Residuals:* $e_i = y_i - hat(y)_i$.
- *Standardized Residuals:* $d_i = frac(e_i, s)$.
- *Studentized (Internally Studentized) Residuals:*
  $ r_i = frac(e_i, s sqrt(1 - h_(i i))) $
  where $h_(i i) = bold(x)_i^T (bold(X)^T bold(X))^(-1) bold(x)_i$ is the $i$-th diagonal element of the hat matrix $bold(H) = bold(X)(bold(X)^T bold(X))^(-1) bold(X)^T$.

== Diagnostic Tools
- *Residuals vs. Fitted Values Plot:* Evaluates linearity (should show horizontal band around zero) and homoskedasticity (constant vertical spread).
- *Normal Q-Q Plot:* Plots sample quantiles of studentized residuals against theoretical normal quantiles. Departures from the $45^compose$ reference line indicate skewness or heavy tails.
- *Shapiro-Wilk Test:* Statistical hypothesis test with null hypothesis $H_0$: residuals are normally distributed.

---

= 1.4 & 1.5 Variable Selection Methods

== Forward Selection
1. Start with the intercept-only null model: $y = beta_0$.
2. For each candidate predictor $X_j$, compute the $F$-to-enter statistic (or $p$-value) when added to the current model.
3. Add the predictor with the lowest $p$-value if $p < alpha_("enter")$ (typically $0.05$).
4. Repeat until no remaining predictor meets the entry criterion.

== Backward Elimination
1. Start with the full model containing all $p$ candidate predictors.
2. For each predictor in the model, calculate the partial $F$-test or $t$-test $p$-value.
3. Remove the variable with the highest $p$-value if $p > alpha_("remove")$ (typically $0.10$).
4. Re-fit and repeat until all remaining variables are statistically significant.

== Information Criteria Comparison
- *Akaike Information Criterion (AIC):* $"AIC" = 2k - 2ln(hat(L)) approx n ln("SSE"/n) + 2k$. Penalizes model complexity.
- *Bayesian Information Criterion (BIC):* $"BIC" = k ln(n) - 2ln(hat(L)) approx n ln("SSE"/n) + k ln(n)$. Imposes a stronger penalty for large $n$, favoring more parsimonious models.

---

= 1.6 Non-Linear Regression & Transformations

When the true relationship between $Y$ and $X$ exhibits curvature:

== Polynomial Models
$ y_i = beta_0 + beta_1 x_i + beta_2 x_i^2 + dots.h + beta_d x_i^d + epsilon_i $
Linear in parameters $beta$, estimated using standard OLS.

== Logarithmic Transformations

The percentage-change interpretations below are *local (small-coefficient) approximations*, valid for $beta_1$ near 0 or for describing an infinitesimal change in $X$. For the exact effect of a full one-unit (or 1%) change, use the exact formula alongside each approximation.

- *Log-Linear (Exponential growth):* $ln(y_i) = beta_0 + beta_1 x_i + epsilon_i$. Approximate: $100 dot.op beta_1 %$ change in $Y$ per unit $X$. Exact one-unit effect: $100 (e^(beta_1) - 1) %$.
- *Linear-Log:* $y_i = beta_0 + beta_1 ln(x_i) + epsilon_i$. Approximate: $frac(beta_1, 100)$ change in $Y$ per $1%$ increase in $X$ (exact for small $Delta X\/X$; for a full 1% increase, $Delta Y approx beta_1 ln(1.01)$).
- *Log-Log (Elasticity):* $ln(y_i) = beta_0 + beta_1 ln(x_i) + epsilon_i$. Approximate: $beta_1 %$ change in $Y$ per $1%$ increase in $X$. Exact one-percent effect: $100 (1.01^(beta_1) - 1) %$. $beta_1$ is the *elasticity* of $Y$ with respect to $X$ exactly, regardless of the size of the change.

---

= 1.7 Multiple Linear Regression

== Matrix Formulation
$ bold(y) = bold(X) bold(beta) + bold(epsilon.alt) $

where $bold(y) in bb(R)^(n times 1)$, $bold(X) in bb(R)^(n times (k+1))$, $bold(beta) in bb(R)^((k+1) times 1)$, $bold(epsilon.alt) tilde.op cal(N)_n(bold(0), sigma^2 bold(I)_n)$.

== Parameter Estimation & Covariance Matrix
$ hat(bold(beta)) = (bold(X)^T bold(X))^(-1) bold(X)^T bold(y) $
$ op("Cov")(hat(bold(beta))) = sigma^2 (bold(X)^T bold(X))^(-1) quad ("Sample estimator: " s^2 (bold(X)^T bold(X))^(-1)) $

== Multicollinearity & Variance Inflation Factor (VIF)
Multicollinearity occurs when two or more predictor variables in $bold(X)$ are highly linearly correlated.
For predictor $X_j$:
$ "VIF"_j = frac(1, 1 - R_j^2) $
where $R_j^2$ is the coefficient of determination from regressing $X_j$ on all other remaining predictors.
- $"VIF" = 1$: No collinearity.
- $"VIF" > 5$: Moderate collinearity requiring attention.
- $"VIF" > 10$: Severe collinearity causing unstable coefficient estimates and inflated standard errors.

---

= 1.8 Aberrant Data & Heteroskedasticity Problems

== Leverage and Influence Diagnostics
- *Hat Matrix Leverage ($h_(i i)$):* Measures how far observation $i$'s predictors are from the multivariate centroid:
  $ overline(h) = frac(k + 1, n) quad ("High leverage if " h_(i i) > 2 overline(h)) $
- *Cook's Distance ($D_i$):* Measures the aggregate shift in fitted values when observation $i$ is deleted:
  $ D_i = frac(sum_(j=1)^n (hat(y)_j - hat(y)_(j(i)))^2, (k + 1) s^2) = frac(r_i^2, k + 1) ( frac(h_(i i), 1 - h_(i i)) ) $
  Values of $D_i > 1.0$ (or $> frac(4, n)$) indicate highly influential points.

== Heteroskedasticity Detection & Remedies
- *Breusch-Pagan Test:* Regresses squared standardized residuals $e_i^2 / hat(sigma)^2$ on predictors $bold(X)$. Under $H_0$ (homoskedasticity), the test statistic follows $chi_k^2$.
- *Weighted Least Squares (WLS):* When $op("Var")(epsilon_i) = sigma^2 w_i^(-1)$, minimize $sum w_i (y_i - bold(x)_i^T bold(beta))^2$:
  $ hat(bold(beta))_("WLS") = (bold(X)^T bold(W) bold(X))^(-1) bold(X)^T bold(W) bold(y), quad bold(W) = op("diag")(w_1, dots.h, w_n) $
- *Heteroskedasticity-Consistent Covariance (White / HC3):*
  $ op("Cov")_("HC3")(hat(bold(beta))) = (bold(X)^T bold(X))^(-1) ( sum_(i=1)^n frac(e_i^2, (1 - h_(i i))^2) bold(x)_i bold(x)_i^T ) (bold(X)^T bold(X))^(-1) $

---

= Practical Python Implementation

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
import pandas as pd
import numpy as np

# 1. Fit Multiple Linear Regression
model = smf.ols("sale_price ~ sqft_living + bedrooms + bathrooms + house_age_years + np.power(house_age_years, 2) + dist_city_center_km + building_grade", data=df).fit()
print(model.summary())

# 2. Multicollinearity Assessment (VIF)
X = model.model.exog
vif_data = pd.DataFrame({
    "Variable": model.model.exog_names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})
print(vif_data)

# 3. Breusch-Pagan Test for Heteroskedasticity
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"BP Lagrange Multiplier p-value: {bp_test[1]:.4e}")

# 4. Robust Standard Errors (HC3)
robust_model = model.get_robustcov_results(cov_type="HC3")
print(robust_model.summary())
```
