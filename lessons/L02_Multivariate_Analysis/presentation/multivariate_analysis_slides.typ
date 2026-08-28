// Multivariate Analysis Presentation using Touying
#import "@preview/touying:0.5.3": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Multivariate Analysis],
    subtitle: [Random Vectors, Joint Distributions, and Matrix Diagnostics],
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

  + Multivariate Random Vectors & Probability Densities
  + Sample Statistics: Mean Vectors ($overline(bold(x))$) & Covariances ($bold(S), bold(R)$)
  + The Multivariate Normal Distribution (MVN)
  + Geometry of Contour Ellipsoids & Mahalanobis Distance
  + Missing Data Diagnostics & Modern Imputation (KNN, MICE)
  + Multivariate Outlier Detection: Classical vs. Robust MCD
  + Fisher $z$-Transformation & Correlation Intervals
  + Case Study: Atmospheric Monitoring Network Analysis
]

// ============================================================================
// PART 1: RANDOM VECTORS & COVARIANCE ALGEBRA
// ============================================================================

#slide[
  = Random Vectors & Population Parameters

  Let $bold(X) = [X_1, X_2, dots.h, X_p]^T in bb(R)^p$ be a $p$-dimensional random vector.

  #v(0.6em)

  *Population Mean Vector:*
  $ bold(mu) = bb(E)[bold(X)] = mat(delim: "[", bb(E)[X_1], bb(E)[X_2], dots.h, bb(E)[X_p])^T $

  #v(0.6em)

  *Population Covariance Matrix:*
  $ bold(Sigma) = op("Cov")(bold(X)) = bb(E)[(bold(X) - bold(mu))(bold(X) - bold(mu))^T] in bb(R)^(p times p) $

  #v(0.6em)

  *Property:* $bold(Sigma)$ is always symmetric and positive semi-definite ($bold(Sigma) succ.eq 0$).
]

#slide[
  = Sample Statistics & Geometric Interpretation

  Given data matrix $bold(X) in bb(R)^(n times p)$:

  #v(0.5em)

  *Sample Mean & Covariance Matrix:*
  $ overline(bold(x)) = frac(1, n) bold(X)^T bold(1)_n, quad quad bold(S) = frac(1, n - 1) sum_(i=1)^n (bold(x)_i - overline(bold(x)))(bold(x)_i - overline(bold(x)))^T $

  #v(0.5em)

  *Sample Correlation Matrix:*
  $ bold(R) = bold(D)_(bold(S))^(-1/2) bold(S) bold(D)_(bold(S))^(-1/2), quad r_(j k) = frac(s_(j k), s_j s_k) = cos(theta_(j k)) $

  #v(0.5em)

  The correlation $r_(j k)$ is the cosine of the angle between mean-centered observation vectors in sample space $bb(R)^n$.
]

// ============================================================================
// PART 2: THE MULTIVARIATE NORMAL DISTRIBUTION
// ============================================================================

#slide[
  = The Multivariate Normal Distribution (MVN)

  $bold(X) tilde.op cal(N)_p(bold(mu), bold(Sigma))$ has the joint probability density function:

  $ f(bold(x)) = frac(1, (2pi)^(p/2) |bold(Sigma)|^(1/2)) exp( -frac(1, 2) (bold(x) - bold(mu))^T bold(Sigma)^(-1) (bold(x) - bold(mu)) ) $

  #v(0.6em)

  *Fundamental Properties:*
  - Any linear combination $bold(a)^T bold(X)$ is univariate normal $cal(N)(bold(a)^T bold(mu), bold(a)^T bold(Sigma) bold(a))$.
  - All marginals and conditionals are multivariate normal.
  - $op("Cov")(X_j, X_k) = 0 <==> X_j " and " X_k " are statistically independent"$.
]

#slide[
  = Geometry: Contour Ellipsoids & Mahalanobis Distance

  *Squared Mahalanobis Distance:*
  $ D^2(bold(x), bold(mu)) = (bold(x) - bold(mu))^T bold(Sigma)^(-1) (bold(x) - bold(mu)) $

  #v(0.5em)

  *Contour Surfaces:*
  Surfaces of constant probability density form hyper-ellipsoids:
  $ \{ bold(x) : (bold(x) - bold(mu))^T bold(Sigma)^(-1) (bold(x) - bold(mu)) = c^2 \} $

  #v(0.5em)

  *Chi-Square Distribution Property:*
  $ D^2(bold(X), bold(mu)) tilde.op chi_p^2 $
  Enables empirical testing of multivariate normality via *Chi-Square Q-Q Plots*.
]

// ============================================================================
// PART 3: DIAGNOSTICS & CASE STUDY
// ============================================================================

#slide[
  = Missing Values & Outlier Detection

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *1. Missing Data Imputation:*
      - *MCAR / MAR:* Probability of missingness depends on observed features.
      - *KNN Imputation:* Fills missing coordinates using the $K$-nearest complete neighbors.
      - *MICE:* Iterative chained equation modeling.
    ],
    [
      *2. Multivariate Outliers:*
      - Data points may look normal marginally, but violate covariance $bold(Sigma)$.
      - *Classical Cutoff:* $D_i^2 > chi_(p, 1 - alpha)^2$.
      - *Robust MCD:* Computes minimum covariance determinant to prevent outlier masking.
    ]
  )
]

#slide[
  = Fisher $z$-Transformation for Correlations

  *Problem:* Sampling distribution of $r$ is skewed when $|rho| > 0$.

  #v(0.5em)

  *Fisher's Transformation:*
  $ z = op("arctanh")(r) = frac(1, 2) ln( frac(1 + r, 1 - r) ) attach(tilde.op, t: "approx") cal(N)( op("arctanh")(rho), frac(1, n - 3) ) $

  #v(0.5em)

  *95% Confidence Interval for $rho$:*
  $ [rho_L, rho_U] = [ tanh( z - 1.96 frac(1, sqrt(n - 3)) ), \, tanh( z + 1.96 frac(1, sqrt(n - 3)) ) ] $
]

#slide[
  = Summary & Key Takeaways

  - Multivariate analysis accounts for *joint dependence structures* that univariate methods miss.
  - The *Multivariate Normal (MVN)* distribution provides the mathematical foundation for linear multivariate techniques.
  - Mahalanobis distance measures generalized distance accounting for variance and correlation.
  - Modern data cleaning requires *multivariate imputation (KNN)* and *robust outlier detection (MCD)*.
  - Andrews curves and correlation heatmaps enable exploratory visualization of high-dimensional observations.

  #v(1em)

  #align(center)[
    *Next Chapter: Chapter 3 — Principal Component Analysis (PCA)*
  ]
]
