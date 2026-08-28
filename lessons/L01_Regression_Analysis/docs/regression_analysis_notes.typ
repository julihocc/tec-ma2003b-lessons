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

$ y_i = beta_0 + beta_1 x_i + epsilon_i, \quad i = 1, 2, \dots, n $

where:
- $beta_0 \in \mathbb{R}$ is the population intercept parameter.
- $beta_1 \in \mathbb{R}$ is the population slope coefficient (expected marginal change in $Y$ per unit change in $X$).
- $epsilon_i$ is the unobserved random error term satisfying the classical Gauss-Markov assumptions:
  $ \mathbb{E}[epsilon_i] = 0, \quad \operatorname{Var}(epsilon_i) = sigma^2, \quad \operatorname{Cov}(epsilon_i, epsilon_j) = 0 \ (i \neq j) $

Under the normality assumption, $epsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, sigma^2)$.

== Ordinary Least Squares (OLS) Derivation
The OLS estimators $(\hat{beta}_0, \hat{beta}_1)$ minimize the Residual Sum of Squares:

$ S(beta_0, beta_1) = \sum_{i=1}^n (y_i - beta_0 - beta_1 x_i)^2 $

Setting partial derivatives to zero yields the normal equations:
$ \frac{\partial S}{\partial beta_0} = -2 \sum_{i=1}^n (y_i - beta_0 - beta_1 x_i) = 0 \implies \hat{beta}_0 = \bar{y} - \hat{beta}_1 \bar{x} $
$ \frac{\partial S}{\partial beta_1} = -2 \sum_{i=1}^n x_i (y_i - beta_0 - beta_1 x_i) = 0 \implies \hat{beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{S_{x y}}{S_{x x}} $

#info[
  *Gauss-Markov Theorem:* Under the assumptions of linearity, strict exogeneity, homoskedasticity, and no autocorrelation, the OLS estimator $\hat{\boldsymbol{\beta}}$ is the *Best Linear Unbiased Estimator (BLUE)*.
]

---

= 1.2 ANOVA and Confidence Intervals for Predictions

== Sum of Squares Decomposition
The total sample variation in $Y$ is partitioned into explained (regression) and unexplained (residual) components:

$ \underbrace{\sum_{i=1}^n (y_i - \bar{y})^2}_{\text{SST (Total)}} = \underbrace{\sum_{i=1}^n (\hat{y}_i - \bar{y})^2}_{\text{SSR (Regression)}} + \underbrace{\sum_{i=1}^n (y_i - \hat{y}_i)^2}_{\text{SSE (Error)}} $

Degrees of freedom: $\text{df}_{\text{Total}} = n - 1$, $\text{df}_{\text{Regression}} = 1$ (or $k$ predictors), $\text{df}_{\text{Error}} = n - k - 1$.

#table(
  columns: (2fr, 1.5fr, 1.5fr, 1.5fr, 1.5fr),
  align: center,
  table.header([*Source*], [*Sum of Squares*], [*Degrees of Freedom*], [*Mean Square (MS)*], [*$F$-Statistic*]),
  [Regression], [SSR], [$k$], [$\text{MSR} = \text{SSR}/k$], [$F = \frac{\text{MSR}}{\text{MSE}}$],
  [Residual (Error)], [SSE], [$n - k - 1$], [$\text{MSE} = \text{SSE}/(n - k - 1)$], [],
  [Total], [SST], [$n - 1$], [], []
)

Coefficient of determination:
$ R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}} $

== Mean Response Confidence Interval vs. Individual Prediction Interval
For a specified predictor value $x_0$:

*1. Confidence Interval for the Expected Mean $\mathbb{E}[Y | X = x_0]$:*
$ \hat{y}_0 \pm t_{n-2, 1 - \alpha/2} \cdot s \sqrt{\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{x x}}} $

*2. Prediction Interval for a Single New Observation $Y_{\text{new}}$:*
$ \hat{y}_0 \pm t_{n-2, 1 - \alpha/2} \cdot s \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{x x}}} $

#tip[
  Prediction intervals are always strictly wider than confidence intervals because they account for both model parameter uncertainty and individual observation variance $\sigma^2$.
]

---

= 1.3 Residual Analysis & Condition of Normality

== Residual Definitions
- *Raw Residuals:* $e_i = y_i - \hat{y}_i$.
- *Standardized Residuals:* $d_i = \frac{e_i}{s}$.
- *Studentized (Internally Studentized) Residuals:*
  $ r_i = \frac{e_i}{s \sqrt{1 - h_{i i}}} $
  where $h_{i i} = \mathbf{x}_i^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_i$ is the $i$-th diagonal element of the hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$.

== Diagnostic Tools
- *Residuals vs. Fitted Values Plot:* Evaluates linearity (should show horizontal band around zero) and homoskedasticity (constant vertical spread).
- *Normal Q-Q Plot:* Plots sample quantiles of studentized residuals against theoretical normal quantiles. Departures from the $45^\circ$ reference line indicate skewness or heavy tails.
- *Shapiro-Wilk Test:* Statistical hypothesis test with null hypothesis $H_0$: residuals are normally distributed.

---

= 1.4 & 1.5 Variable Selection Methods

== Forward Selection
1. Start with the intercept-only null model: $y = \beta_0$.
2. For each candidate predictor $X_j$, compute the $F$-to-enter statistic (or $p$-value) when added to the current model.
3. Add the predictor with the lowest $p$-value if $p < \alpha_{\text{enter}}$ (typically $0.05$).
4. Repeat until no remaining predictor meets the entry criterion.

== Backward Elimination
1. Start with the full model containing all $p$ candidate predictors.
2. For each predictor in the model, calculate the partial $F$-test or $t$-test $p$-value.
3. Remove the variable with the highest $p$-value if $p > \alpha_{\text{remove}}$ (typically $0.10$).
4. Re-fit and repeat until all remaining variables are statistically significant.

== Information Criteria Comparison
- *Akaike Information Criterion (AIC):* $\text{AIC} = 2k - 2\ln(\hat{L}) \approx n \ln(\text{SSE}/n) + 2k$. Penalizes model complexity.
- *Bayesian Information Criterion (BIC):* $\text{BIC} = k \ln(n) - 2\ln(\hat{L}) \approx n \ln(\text{SSE}/n) + k \ln(n)$. Imposes a stronger penalty for large $n$, favoring more parsimonious models.

---

= 1.6 Non-Linear Regression & Transformations

When the true relationship between $Y$ and $X$ exhibits curvature:

== Polynomial Models
$ y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \dots + \beta_d x_i^d + \epsilon_i $
Linear in parameters $\beta$, estimated using standard OLS.

== Logarithmic Transformations
- *Log-Linear (Exponential growth):* $\ln(y_i) = \beta_0 + \beta_1 x_i + \epsilon_i \implies 100 \cdot \beta_1 \%$ percentage change in $Y$ per unit $X$.
- *Linear-Log:* $y_i = \beta_0 + \beta_1 \ln(x_i) + \epsilon_i \implies \frac{\beta_1}{100}$ change in $Y$ per $1\%$ increase in $X$.
- *Log-Log (Elasticity):* $\ln(y_i) = \beta_0 + \beta_1 \ln(x_i) + \epsilon_i \implies \beta_1\%$ change in $Y$ per $1\%$ increase in $X$.

---

= 1.7 Multiple Linear Regression

== Matrix Formulation
$ \mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon} $

where $\mathbf{y} \in \mathbb{R}^{n \times 1}$, $\mathbf{X} \in \mathbb{R}^{n \times (k+1)}$, $\boldsymbol{\beta} \in \mathbb{R}^{(k+1) \times 1}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}_n(\mathbf{0}, \sigma^2 \mathbf{I}_n)$.

== Parameter Estimation & Covariance Matrix
$ \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} $
$ \operatorname{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1} \quad \left(\text{Sample estimator: } s^2 (\mathbf{X}^T \mathbf{X})^{-1}\right) $

== Multicollinearity & Variance Inflation Factor (VIF)
Multicollinearity occurs when two or more predictor variables in $\mathbf{X}$ are highly linearly correlated.
For predictor $X_j$:
$ \text{VIF}_j = \frac{1}{1 - R_j^2} $
where $R_j^2$ is the coefficient of determination from regressing $X_j$ on all other remaining predictors.
- $\text{VIF} = 1$: No collinearity.
- $\text{VIF} > 5$: Moderate collinearity requiring attention.
- $\text{VIF} > 10$: Severe collinearity causing unstable coefficient estimates and inflated standard errors.

---

= 1.8 Aberrant Data & Heteroskedasticity Problems

== Leverage and Influence Diagnostics
- *Hat Matrix Leverage ($h_{i i}$):* Measures how far observation $i$'s predictors are from the multivariate centroid:
  $ \bar{h} = \frac{k + 1}{n} \quad (\text{High leverage if } h_{i i} > 2\bar{h}) $
- *Cook's Distance ($D_i$):* Measures the aggregate shift in fitted values when observation $i$ is deleted:
  $ D_i = \frac{\sum_{j=1}^n (\hat{y}_j - \hat{y}_{j(i)})^2}{(k + 1) s^2} = \frac{r_i^2}{k + 1} \left( \frac{h_{i i}}{1 - h_{i i}} \right) $
  Values of $D_i > 1.0$ (or $> \frac{4}{n}$) indicate highly influential points.

== Heteroskedasticity Detection & Remedies
- *Breusch-Pagan Test:* Regresses squared standardized residuals $e_i^2 / \hat{\sigma}^2$ on predictors $\mathbf{X}$. Under $H_0$ (homoskedasticity), the test statistic follows $\chi_k^2$.
- *Weighted Least Squares (WLS):* When $\operatorname{Var}(\epsilon_i) = \sigma^2 w_i^{-1}$, minimize $\sum w_i (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2$:
  $ \hat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}, \quad \mathbf{W} = \operatorname{diag}(w_1, \dots, w_n) $
- *Heteroskedasticity-Consistent Covariance (White / HC3):*
  $ \operatorname{Cov}_{\text{HC3}}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^T \mathbf{X})^{-1} \left( \sum_{i=1}^n \frac{e_i^2}{(1 - h_{i i})^2} \mathbf{x}_i \mathbf{x}_i^T \right) (\mathbf{X}^T \mathbf{X})^{-1} $

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
