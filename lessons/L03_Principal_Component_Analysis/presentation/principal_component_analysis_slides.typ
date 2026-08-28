// Principal Component Analysis Presentation using Touying
#import "@preview/touying:0.5.3": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Principal Component Analysis (PCA)],
    subtitle: [Dimensionality Reduction, Spectral Decomposition, and Factor Modeling],
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

  + Motivations & Use Cases for Principal Component Analysis
  + Geometric Foundations: Rotation & Variance Maximization
  + Spectral Decomposition of Covariance ($bold(S)$) and Correlation ($bold(R)$)
  + Mathematical Derivation via Lagrange Multipliers
  + Component Retention Criteria (Kaiser, Scree, Parallel Analysis)
  + Loadings, Scores, and Communalities
  + High-Dimensional 2D/3D Biplots
  + Practical Case Study: Global Financial Asset Allocation
]

// ============================================================================
// PART 1: GEOMETRY & SPECTRAL DECOMPOSITION
// ============================================================================

#slide[
  = What is Principal Component Analysis?

  *Core Objective:*
  Transform $p$ correlated variables $X_1, dots.h, X_p$ into $k lt.double p$ uncorrelated variables $Y_1, dots.h, Y_k$ (*Principal Components*) that capture the maximum possible variance.

  #v(0.6em)

  *Key Applications:*
  - *Data Compression:* Compress feature space with minimal information loss.
  - *Multicollinearity Cure:* Replace correlated regressors with orthogonal components.
  - *Exploratory Visualization:* Project high dimensions onto 2D/3D score planes.
  - *Risk Modeling:* Identify macroeconomic drivers across assets.
]

#slide[
  = Geometry: Finding Axes of Maximum Variance

  #grid(
    columns: (1.2fr, 1fr),
    gutter: 1.5em,
    [
      *1. First Principal Component ($Y_1$):*
      $ Y_1 = bold(a)_1^T bold(X) = a_(11)X_1 + dots.h + a_(1p)X_p $
      Direction of maximum data dispersion subject to $\|bold(a)_1\| = 1$.

      #v(0.5em)
      *2. Second Principal Component ($Y_2$):*
      $ Y_2 = bold(a)_2^T bold(X) quad "with " bold(a)_2^T bold(a)_1 = 0 $
      Direction of maximum remaining variance, orthogonal to $Y_1$.
    ],
    [
      *Principal Axes of Ellipsoid:*
      Eigenvectors specify the orientation of the constant-density ellipsoid axes. Half-lengths are proportional to $plus.minus c sqrt(lambda_j)$.
    ]
  )
]

#slide[
  = Mathematical Derivation (Lagrange Multipliers)

  Maximize $op("Var")(Y_1) = bold(a)_1^T bold(Sigma) bold(a)_1$ subject to $bold(a)_1^T bold(a)_1 = 1$.

  #v(0.5em)

  *Lagrangian Objective:*
  $ cal(L)(bold(a)_1, lambda_1) = bold(a)_1^T bold(Sigma) bold(a)_1 - lambda_1 (bold(a)_1^T bold(a)_1 - 1) $

  #v(0.5em)

  *First-Order Condition:*
  $ frac(partial cal(L), partial bold(a)_1) = 2 bold(Sigma) bold(a)_1 - 2 lambda_1 bold(a)_1 = bold(0) ==> bold(Sigma) bold(a)_1 = lambda_1 bold(a)_1 $

  #v(0.5em)

  *Result:* $lambda_1$ is the largest eigenvalue of $bold(Sigma)$, and $bold(a)_1$ is the corresponding normalized eigenvector.
]

#slide[
  = Spectral Decomposition & Variance Explained

  *Spectral Decomposition:*
  $ bold(S) = bold(V) bold(Lambda) bold(V)^T = sum_(j=1)^p lambda_j bold(v)_j bold(v)_j^T $

  where $bold(V) = [bold(v)_1, dots.h, bold(v)_p]$ is the eigenvector matrix and $bold(Lambda) = op("diag")(lambda_1, dots.h, lambda_p)$.

  #v(0.6em)

  *Total Variance Preservation:*
  $ "Total Sample Variance" = op("tr")(bold(S)) = sum_(j=1)^p s_(j j) = sum_(j=1)^p lambda_j $

  #v(0.6em)

  *Proportion of Explained Variance:*
  $ "Prop"_j = frac(lambda_j, sum_(k=1)^p lambda_k) $
]

// ============================================================================
// PART 2: RETENTION & INTERPRETATION
// ============================================================================

#slide[
  = How Many Components to Retain?

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *1. Kaiser-Guttman Rule:*
      - Retain $lambda_j > 1.0$ (for correlation $bold(R)$).
      - Retains components explaining more variance than a single standardized variable.

      #v(0.5em)
      *2. Cumulative Variance Threshold:*
      - Retain components until cumulative variance reaches $70% - 85%$.
    ],
    [
      *3. Cattell Scree Plot:*
      - Visually identify the "elbow" where eigenvalue decrease flattens out.

      #v(0.5em)
      *4. Horn's Parallel Analysis:*
      - Retain components whose eigenvalues exceed the 95th percentile of simulated random noise.
    ]
  )
]

#slide[
  = Loadings, Scores, and the PCA Biplot

  *Component Loadings (Correlations with Variables):*
  $ L_(j k) = op("Corr")(X_k, Y_j) = a_(j k) sqrt(lambda_j) quad ("for standardized data") $

  #v(0.6em)

  *The Gabriel Biplot (1971):*
  Simultaneously displays:
  - *Observation Points:* $Y_(i 1), Y_(i 2)$ mapped in component score space.
  - *Variable Vectors:* Directed arrows showing loading magnitude and pairwise cosine correlation.
]

// ============================================================================
// PART 3: CASE STUDY & SUMMARY
// ============================================================================

#slide[
  = Case Study: Multi-Asset Portfolio Decomposition

  *Setting:* 10 global asset classes and industry sector returns (600 trading sessions).

  #v(0.4em)

  *3 Retained Macroeconomic Factors ($76.8%$ Variance):*
  1. *PC1 (Market Equity Beta - $45.2%$):* Broad stock market risk (high loadings across US, Global, Tech).
  2. *PC2 (Duration / Interest Rate - $19.4%$):* Positive for Financials; negative for Treasuries & REITs.
  3. *PC3 (Commodity Inflation - $12.2%$):* Driven by Gold & Energy sector returns.

  #v(0.4em)

  *Takeaway:* Compress 10 risk dimensions into 3 uncorrelated macro factors for hedging.
]

#slide[
  = Summary & Key Takeaways

  - PCA is the cornerstone of unsupervised multivariate feature extraction.
  - Formulated via the *spectral decomposition* of $bold(S)$ or $bold(R)$.
  - Always standardize data ($z$-scores) when variables have different units or variances.
  - Validate component retention using multiple criteria (Kaiser, Scree, Parallel Analysis).
  - Use *Biplots* for holistic interpretation of observations and features.

  #v(1em)

  #align(center)[
    *Next Chapter: Chapter 4 — Factor Analysis (Latent Variable Models)*
  ]
]
