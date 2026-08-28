// Regression Analysis Presentation using Touying
#import "@preview/touying:0.5.3": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Regression Analysis],
    subtitle: [Linear, Multiple, and Diagnostic Modeling],
    author: [Juliho Castillo Colmenares],
    date: none,
    institution: [Tec de Monterrey],
    logo: none,
  ),
  config-colors(
    primary: rgb("#003366"),
    secondary: rgb("#1E88E5"),
    tertiary: rgb("#666666"),
  ),
)

#show link: set text(fill: rgb("#1E88E5"))

#title-slide()

// ============================================================================
// AGENDA
// ============================================================================

#slide[
  = Today's Agenda

  + Foundations of Simple Linear Regression (OLS)
  + ANOVA & Intervals (Confidence vs. Prediction)
  + Residual Diagnostics & Normality Assessment
  + Stepwise Variable Selection & Information Criteria
  + Non-Linearity & Polynomial Transformations
  + Multiple Linear Regression (Matrix Formulation & VIF)
  + Aberrant Observations & Heteroskedasticity (Cook's D, HC3)
  + Practical Case Study: Residential Property Valuation
]

// ============================================================================
// PART 1: LINEAR REGRESSION FOUNDATIONS
// ============================================================================

#slide[
  = Simple Linear Regression: The OLS Model

  *Population Equation:*
  $ y_i = beta_0 + beta_1 x_i + epsilon.alt_i, quad i = 1, dots.h, n $

  where $bb(E)[epsilon.alt_i] = 0$, $op("Var")(epsilon.alt_i) = sigma^2$, and $op("Cov")(epsilon.alt_i, epsilon.alt_j) = 0$.

  #v(0.6em)

  *Ordinary Least Squares Estimators:*
  $ hat(beta)_1 = frac(sum_(i=1)^n (x_i - overline(x))(y_i - overline(y)), sum_(i=1)^n (x_i - overline(x))^2) = frac(S_(x y), S_(x x)), quad quad hat(beta)_0 = overline(y) - hat(beta)_1 overline(x) $

  #v(0.6em)

  *Gauss-Markov Theorem:*
  Under the classical assumptions, OLS estimators are the *Best Linear Unbiased Estimators (BLUE)*.
]

#slide[
  = ANOVA & Variation Decomposition

  *Total Variation Partitioning:*
  $ underbrace(sum (y_i - overline(y))^2, "SST") = underbrace(sum (hat(y)_i - overline(y))^2, "SSR (Explained)") + underbrace(sum (y_i - hat(y)_i)^2, "SSE (Residual)") $

  #v(0.5em)

  *Goodness of Fit ($R^2$):*
  $ R^2 = frac("SSR", "SST") = 1 - frac("SSE", "SST") $

  #v(0.5em)

  *Overall Model $F$-Test:*
  $ F = frac("MSR", "MSE") = frac("SSR"/k, "SSE"/(n - k - 1)) tilde.op F_(k, n - k - 1) $
]

#slide[
  = Mean Response vs. Individual Prediction Intervals

  For a new observation at $x_0$:

  #v(0.6em)

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *1. Mean Response CI:*
      $ hat(y)_0 plus.minus t_("crit") dot.op s sqrt(frac(1, n) + frac((x_0 - overline(x))^2, S_(x x))) $
      - Captures uncertainty in the *average response* $bb(E)[Y | X = x_0]$.
      - Narrows as sample size $n arrow.r infinity$.
    ],
    [
      *2. Individual Prediction PI:*
      $ hat(y)_0 plus.minus t_("crit") dot.op s sqrt(1 + frac(1, n) + frac((x_0 - overline(x))^2, S_(x x))) $
      - Captures uncertainty in a *single new outcome* $Y_("new")$.
      - Always strictly wider due to $+1$ term ($sigma^2$).
    ]
  )
]

// ============================================================================
// PART 2: MULTIPLE REGRESSION & DIAGNOSTICS
// ============================================================================

#slide[
  = Multiple Linear Regression (Matrix Formulation)

  *Vector Model:* $bold(y) = bold(X) bold(beta) + bold(epsilon.alt)$, with $bold(epsilon.alt) tilde.op cal(N)_n(bold(0), sigma^2 bold(I)_n)$.

  #v(0.6em)

  *Normal Equations & OLS Solution:*
  $ bold(X)^T bold(X) hat(bold(beta)) = bold(X)^T bold(y) ==> hat(bold(beta)) = (bold(X)^T bold(X))^(-1) bold(X)^T bold(y) $

  #v(0.6em)

  *Covariance of Estimates & Hat Matrix:*
  $ op("Cov")(hat(bold(beta))) = sigma^2 (bold(X)^T bold(X))^(-1), quad quad hat(bold(y)) = bold(H) bold(y) = bold(X)(bold(X)^T bold(X))^(-1) bold(X)^T bold(y) $

  The diagonal $h_(i i)$ represents the *leverage* of observation $i$.
]

#slide[
  = Multicollinearity: The VIF Diagnostic

  *Problem:* High correlation between predictors inflates standard errors $op("Var")(hat(beta)_j)$.

  #v(0.6em)

  *Variance Inflation Factor (VIF):*
  $ "VIF"_j = frac(1, 1 - R_j^2) $
  where $R_j^2$ is the $R^2$ obtained from regressing $X_j$ onto all other remaining predictors.

  #v(0.6em)

  *Interpretation Guidelines:*
  - $"VIF" = 1$: Completely independent features.
  - $1 < "VIF" < 5$: Acceptable mild correlation.
  - $"VIF" gt.eq 10$: Severe collinearity; requires feature pruning or Ridge/PCA regularization.
]

#slide[
  = Aberrant Data: Outliers & Cook's Distance

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *1. Studentized Residuals:*
      $ r_i = frac(e_i, s sqrt(1 - h_(i i))) $
      - Identifies $Y$-space outliers ($|r_i| > 3$).

      #v(0.5em)
      *2. Leverage ($h_(i i)$):*
      - Identifies $X$-space extreme points ($h_(i i) > frac(2(k+1), n)$).
    ],
    [
      *3. Cook's Distance ($D_i$):*
      $ D_i = frac(r_i^2, k + 1) ( frac(h_(i i), 1 - h_(i i)) ) $
      - Measures the overall influence of point $i$ on *all* fitted values $hat(bold(y))$.
      - Threshold: $D_i > frac(4, n)$ warrants detailed investigation.
    ]
  )
]

#slide[
  = Heteroskedasticity: Breusch-Pagan & Robust SE

  *Violation:* $op("Var")(epsilon.alt_i | bold(X)) = sigma_i^2 eq.not "constant"$.

  #v(0.5em)

  *Breusch-Pagan Test:*
  Regress squared residuals $e_i^2 / hat(sigma)^2$ on $bold(X)$. Test statistic $L M = frac(1, 2)"SSR"_("aux") tilde.op chi_k^2$. Reject $H_0$ if $p < 0.05$.

  #v(0.5em)

  *Remedies:*
  1. *Heteroskedasticity-Consistent (HC3) Robust SE:*
     $ op("Cov")_("HC3")(hat(bold(beta))) = (bold(X)^T bold(X))^(-1) ( sum_(i=1)^n frac(e_i^2, (1 - h_(i i))^2) bold(x)_i bold(x)_i^T ) (bold(X)^T bold(X))^(-1) $
  2. *Weighted Least Squares (WLS):* Weight observations by $w_i = 1/sigma_i^2$.
]

// ============================================================================
// PART 3: CASE STUDY & SUMMARY
// ============================================================================

#slide[
  = Case Study: Housing Valuation Workflow

  *Scenario:* 1,000 residential transactions with 7 structural & geographic predictors.

  #v(0.5em)

  ```python
  import statsmodels.formula.api as smf

  # 1. Quadratic Polynomial Specification
  formula = "sale_price ~ sqft_living + bedrooms + bathrooms + " \
            "dist_city_center_km + building_grade + house_age_years + I(house_age_years**2)"

  # 2. Fit with Robust HC3 Covariance
  model = smf.ols(formula, data=df).fit()
  robust_model = model.get_robustcov_results(cov_type="HC3")
  print(robust_model.summary())
  ```

  #v(0.5em)

  *Findings:* Model explains $88.4%$ of price variance with valid robust $p$-values.
]

#slide[
  = Summary & Key Takeaways

  - *Linear Regression* is the foundational tool for parametric inference and prediction.
  - Always verify OLS assumptions: linearity, normality of residuals, homoskedasticity, and no extreme collinearity ($"VIF" < 5$).
  - When non-linear patterns exist, incorporate polynomial terms or logarithmic transforms.
  - Guard against influential outliers using Cook's distance.
  - When heteroskedasticity is present, employ *HC3 robust standard errors* to protect inferential validity.

  #v(1em)

  #align(center)[
    *Next Chapter: Chapter 2 — Multivariate Analysis Foundations*
  ]
]
