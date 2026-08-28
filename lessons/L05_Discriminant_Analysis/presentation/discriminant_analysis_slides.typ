// Discriminant Analysis Presentation using Touying
#import "@preview/touying:0.5.3": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Discriminant Analysis],
    subtitle: [Classification with Statistical Foundations],
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
// PART 1: OVERVIEW & THE CLASSIFICATION PROBLEM
// ============================================================================

#slide[
  = Today's Agenda

  + The Supervised Classification Problem
  + Fisher's Linear Discriminant (2 Populations)
  + Prior Probabilities & Misclassification Costs (ECM)
  + Multi-Group Discriminant Analysis ($k > 2$)
  + Canonical Discriminant Functions & Wilks' Lambda
  + Linear (LDA) vs. Quadratic (QDA) Discriminant Analysis
  + Python Implementation & Marketing Segmentation Case Study
]

#slide[
  = The Classification Problem

  *Scenario:* Given an observation $bold(x) in bb(R)^p$, assign it to one of $k$ predefined mutually exclusive groups $Pi_1, Pi_2, dots.h, Pi_k$.

  #v(0.8em)

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.2em,
    [
      *Marketing*
      - High-Value
      - Loyal
      - Occasional
    ],
    [
      *Finance & Risk*
      - Low Credit Risk
      - Medium Risk
      - Default / High Risk
    ],
    [
      *Medical Diagnostics*
      - Healthy Control
      - Benign Tumor
      - Malignant Tumor
    ]
  )

  #v(0.8em)
  *Goal:* Construct discriminant score functions $d_i(bold(x))$ that maximize separation between groups while minimizing within-group scatter.
]

// ============================================================================
// PART 2: TWO-GROUP DISCRIMINANT ANALYSIS
// ============================================================================

#slide[
  = Fisher's Linear Discriminant (Two Populations)

  Let $bold(x) in bb(R)^p$ from population $Pi_1 tilde.op (bold(mu)_1, bold(Sigma))$ or $Pi_2 tilde.op (bold(mu)_2, bold(Sigma))$ with shared covariance $bold(Sigma)$.

  #v(0.5em)

  *Fisher's Criterion:* Find linear combination $y = bold(a)^T bold(x)$ maximizing the ratio of between-group to within-group variance:
  $ J(bold(a)) = frac((bold(a)^T bold(mu)_1 - bold(a)^T bold(mu)_2)^2, bold(a)^T bold(Sigma) bold(a)) $

  #v(0.5em)

  *Optimal Coefficient Vector:*
  $ bold(a)^* = bold(Sigma)^(-1)(bold(mu)_1 - bold(mu)_2) quad ("sample estimator: " hat(bold(a)) = bold(S)_("pooled")^(-1)(overline(bold(x))_1 - overline(bold(x))_2)) $

  *Decision Rule (Equal Priors):*
  Assign to $Pi_1$ if $bold(a)^T bold(x) gt.eq frac(1, 2) bold(a)^T(bold(mu)_1 + bold(mu)_2) = m$; otherwise assign to $Pi_2$.
]

#slide[
  = Prior Probabilities & Misclassification Costs (ECM)

  In real business applications, groups have unequal base rates $p_1, p_2$ and misclassification penalties $C(2|1) eq.not C(1|2)$.

  #v(0.6em)

  *Expected Cost of Misclassification (ECM):*
  $ op("ECM") = C(2|1) p_1 P(2|1) + C(1|2) p_2 P(1|2) $

  #v(0.6em)

  *Bayes Optimal Decision Rule (Minimize ECM):*
  Assign $bold(x)$ to $Pi_1$ if:
  $ frac(f_1(bold(x)), f_2(bold(x))) gt.eq (frac(C(1|2), C(2|1))) (frac(p_2, p_1)) $

  Taking logarithms under multivariate normality yields an adjusted linear cutoff:
  $ d_L(bold(x)) = (bold(mu)_1 - bold(mu)_2)^T bold(Sigma)^(-1) bold(x) - frac(1, 2)(bold(mu)_1 - bold(mu)_2)^T bold(Sigma)^(-1)(bold(mu)_1 + bold(mu)_2) gt.eq ln[ frac(C(1|2) p_2, C(2|1) p_1) ] $
]

// ============================================================================
// PART 3: MULTI-GROUP & CANONICAL DISCRIMINATION
// ============================================================================

#slide[
  = Multi-Group Discrimination ($k > 2$ Populations)

  When classifying among $k$ groups, we define between-group matrix $bold(B)$ and within-group matrix $bold(W)$:
  $ bold(W) = sum_(i=1)^k sum_(j=1)^(n_i) (bold(x)_(i j) - overline(bold(x))_i)(bold(x)_(i j) - overline(bold(x))_i)^T, quad bold(B) = sum_(i=1)^k n_i (overline(bold(x))_i - overline(bold(x)))(overline(bold(x))_i - overline(bold(x)))^T $

  #v(0.6em)

  *Canonical Discriminant Functions:*
  Eigenvalue problem: $bold(W)^(-1) bold(B) bold(v)_r = lambda_r bold(v)_r$.
  - Number of non-zero canonical functions: $s = min(k - 1, p)$.
  - $r$-th canonical score: $Z_r = bold(v)_r^T bold(x)$.
  - Canonical correlations: $rho_r = sqrt(frac(lambda_r, 1 + lambda_r))$.
]

#slide[
  = Wilks' Lambda & Stepwise Variable Selection

  *Wilks' Lambda ($Lambda$):*
  $ Lambda = frac(|bold(W)|, |bold(W) + bold(B)|) = product_(r=1)^s frac(1, 1 + lambda_r), quad Lambda in (0, 1] $

  - $Lambda arrow.r 0$: High group separation (predictors strongly discriminate).
  - $Lambda arrow.r 1$: No group separation (group centroids are identical).

  #v(0.8em)

  *Stepwise Variable Selection:*
  - *Forward Selection:* Add predictor that produces largest reduction in $Lambda$ ($F_("to enter") > F_("in")$).
  - *Backward Elimination:* Remove least contributing predictor ($F_("to remove") < F_("out")$).
  - Prevents overfitting and maximizes model parsimony.
]

// ============================================================================
// PART 4: LDA VS QDA
// ============================================================================

#slide[
  = Linear (LDA) vs. Quadratic (QDA) Discriminant Analysis

  #table(
    columns: (1.5fr, 2fr, 2fr),
    align: (left, left, left),
    table.header([*Feature*], [*LDA (Linear)*], [*QDA (Quadratic)*]),
    [Covariance Assumption], [Equal: $bold(Sigma)_1 = bold(Sigma)_2 = dots.h = bold(Sigma)_k = bold(Sigma)$], [Unequal: $bold(Sigma)_i eq.not bold(Sigma)_j$ per class],
    [Decision Boundary], [Hyperplanes (Linear)], [Quadric surfaces (Hyperbolas / Parabolas)],
    [Parameters to Estimate], [$p$ means + 1 pooled $bold(Sigma)$ ($O(p^2)$)], [$k$ means + $k$ individual $bold(Sigma)_i$ ($O(k p^2)$)],
    [Sample Size Requirement], [Moderate ($n > p$)], [Large per class ($n_i > p$)],
    [Robustness], [More robust to small sample sizes], [Captures complex non-linear variance patterns]
  )
]

// ============================================================================
// PART 5: PYTHON IMPLEMENTATION & CASE STUDY
// ============================================================================

#slide[
  = Python Implementation: Scikit-Learn

  ```python
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report, confusion_matrix

  # 1. Train-test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # 2. Fit Linear Discriminant Analysis (with prior weights)
  lda = LinearDiscriminantAnalysis(priors=[0.30, 0.40, 0.30], store_covariance=True)
  lda.fit(X_train, y_train)

  # 3. Fit Quadratic Discriminant Analysis
  qda = QuadraticDiscriminantAnalysis(priors=[0.30, 0.40, 0.30])
  qda.fit(X_train, y_train)

  # 4. Evaluate predictions
  y_pred = lda.predict(X_test)
  print(classification_report(y_test, y_pred))
  ```
]

#slide[
  = Case Study: Customer Marketing Segmentation (Chapter 5)

  *Context:* 1,200 e-commerce customers classified into 3 segments using 8 behavioral metrics:

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.2em,
    [
      *High-Value (30%)*
      - High spend & order value
      - Low abandonment
      - High loyalty points
    ],
    [
      *Loyal (40%)*
      - Moderate order value
      - High purchase frequency
      - High email open rate
    ],
    [
      *Occasional (30%)*
      - Low frequency & spend
      - High cart abandonment
      - Low engagement
    ]
  )

  #v(0.8em)
  *Key Results:*
  - LDA achieves $> 94%$ classification accuracy on test split.
  - First canonical function captures $87.3%$ of between-group variance (`avg_order_value` and `purchase_freq` dominate).
  - Decision boundaries provide clear automated routing for CRM campaign triggers.
]

#slide[
  = Summary & Takeaways

  - *Discriminant Analysis* is a supervised classification technique grounded in multivariate probability distributions.
  - *LDA* assumes homogeneous covariance matrices and creates planar decision surfaces.
  - *QDA* accommodates class-specific covariance matrices at the cost of estimating more parameters.
  - Always evaluate misclassification costs and prior probabilities to minimize expected business loss.
  - Inspect canonical loadings and Wilks' Lambda to identify key discriminating features.

  #v(1em)

  #align(center)[
    *Next Chapter: Chapter 6 — Cluster Analysis (Unsupervised Discovery)*
  ]
]
