// Multivariate Analysis - Lecture Notes
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
  title: "Multivariate Analysis - Lecture Notes",
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
  #text(24pt, weight: "bold")[Multivariate Analysis Foundations]

  #text(16pt)[Theoretical Principles, Probability Models, and Diagnostics]

  #v(0.5em)

  #text(12pt)[MA2003B — Application of Multivariate Methods in Data Science]

  #v(0.5em)

  #text(11pt)[Tecnológico de Monterrey]

  #v(2em)
]

#pagebreak()

= Course and Chapter Overview

== Course Context
Multivariate statistical methods extend classical single-variable inference to systems where $p gt.eq 2$ interconnected random variables are observed simultaneously. In *MA2003B*, this module provides the fundamental matrix algebra, probability theory, and diagnostic toolset required for all downstream dimensionality reduction (PCA, Factor Analysis) and supervised classification (Discriminant Analysis) techniques.

== Learning Objectives
By the end of this chapter, students will be able to:
- Formulate random vectors, joint probability density functions, and marginal/conditional expectations.
- Compute and interpret sample mean vectors ($overline(bold(x))$), sample covariance matrices ($bold(S)$), and sample correlation matrices ($bold(R)$).
- Analyze the mathematical properties and geometry of the Multivariate Normal (MVN) distribution.
- Construct constant-density contour ellipsoids and compute Mahalanobis generalized distances.
- Identify missing data mechanisms (MCAR, MAR, MNAR) and apply modern imputation algorithms (KNN, MICE).
- Detect multivariate outliers using Mahalanobis distances and Robust Minimum Covariance Determinant (MCD) estimators.
- Derive Fisher $z$-transformations and construct confidence intervals for correlation parameters ($rho$).
- Apply multidimensional visualization techniques including scatter matrices, correlation heatmaps, and Andrews curves.

---

= 2.1 Multivariate Distributions & Random Vectors

== Random Vectors
Let $bold(X) = [X_1, X_2, dots.h, X_p]^T$ be a $p$-dimensional column random vector where each component $X_j$ is a real-valued random variable.

== Joint Cumulative Distribution Function (CDF)
$ F(bold(x)) = P(X_1 lt.eq x_1, X_2 lt.eq x_2, dots.h, X_p lt.eq x_p) $

== Joint Probability Density Function (PDF)
For continuous random vectors with density $f(bold(x)) = f(x_1, dots.h, x_p)$:
$ P(bold(X) in A) = integral dots.h integral_A f(x_1, dots.h, x_p) \, d x_1 dots.h d x_p $

Marginal density for subset $bold(X)_1$:
$ f_1(bold(x)_1) = integral_(bb(R)^(p - k)) f(bold(x)_1, bold(x)_2) \, d bold(x)_2 $

Conditional density of $bold(X)_1$ given $bold(X)_2 = bold(x)_2$:
$ f(bold(x)_1 | bold(x)_2) = frac(f(bold(x)_1, bold(x)_2), f_2(bold(x)_2)), quad "provided " f_2(bold(x)_2) > 0 $

---

= 2.2 Mean Vectors and Covariance Matrices

== Population Mean Vector
$ bold(mu) = bb(E)[bold(X)] = mat(delim: "[", bb(E)[X_1]; bb(E)[X_2]; dots.v; bb(E)[X_p]) = mat(delim: "[", mu_1; mu_2; dots.v; mu_p) $

== Population Covariance Matrix
$ bold(Sigma) = op("Cov")(bold(X)) = bb(E)[(bold(X) - bold(mu))(bold(X) - bold(mu))^T] = mat(delim: "[", sigma_(11), sigma_(12), dots.h, sigma_(1p); sigma_(21), sigma_(22), dots.h, sigma_(2p); dots.v, dots.v, dots.down, dots.v; sigma_(p 1), sigma_(p 2), dots.h, sigma_(p p)) $

where $sigma_(j j) = op("Var")(X_j) = sigma_j^2$ and $sigma_(j k) = op("Cov")(X_j, X_k) = sigma_(k j)$.

#info[
  *Positive Semi-Definiteness:* For any non-zero constant vector $bold(a) in bb(R)^p$, $op("Var")(bold(a)^T bold(X)) = bold(a)^T bold(Sigma) bold(a) gt.eq 0$. Hence, all valid covariance matrices $bold(Sigma)$ are symmetric positive semi-definite ($bold(Sigma) succ.eq 0$).
]

== Sample Statistics Matrix Operations
Given a data matrix $bold(X) in bb(R)^(n times p)$ with $n$ observations:

- *Sample Mean Vector:*
  $ overline(bold(x)) = frac(1, n) bold(X)^T bold(1)_n = mat(delim: "[", overline(x)_1, overline(x)_2, dots.h, overline(x)_p)^T $
- *Sample Covariance Matrix:*
  $ bold(S) = frac(1, n - 1) sum_(i=1)^n (bold(x)_i - overline(bold(x)))(bold(x)_i - overline(bold(x)))^T = frac(1, n - 1) bold(X)^T ( bold(I)_n - frac(1, n) bold(1)_n bold(1)_n^T ) bold(X) $

---

= 2.3 Correlations and Correlation Matrices

== Population and Sample Correlation
Let $bold(D) = op("diag")(sigma_(11), sigma_(22), dots.h, sigma_(p p))$ be the diagonal *variance* matrix ($sigma_(j j) = op("Var")(X_j)$, not a standard deviation). Its inverse square root $bold(D)^(-1/2) = op("diag")(1/sigma_1, dots.h, 1/sigma_p)$ contains reciprocal standard deviations, which is what rescales $bold(Sigma)$ to a correlation matrix below.

- *Population Correlation Matrix:*
  $ bold(P) = bold(D)^(-1/2) bold(Sigma) bold(D)^(-1/2) = mat(delim: "[", 1, rho_(12), dots.h, rho_(1p); rho_(21), 1, dots.h, rho_(2p); dots.v, dots.v, dots.down, dots.v; rho_(p 1), rho_(p 2), dots.h, 1), quad rho_(j k) = frac(sigma_(j k), sigma_j sigma_k) in [-1, 1] $
- *Sample Correlation Matrix:*
  $ bold(R) = bold(D)_(bold(S))^(-1/2) bold(S) bold(D)_(bold(S))^(-1/2), quad r_(j k) = frac(s_(j k), s_j s_k) $

== Geometric Interpretation
In sample space $bb(R)^n$, the sample correlation coefficient $r_(j k)$ equals the *cosine of the angle* $theta$ between the mean-centered observation vectors $bold(d)_j = bold(x)_j - overline(x)_j bold(1)$ and $bold(d)_k = bold(x)_k - overline(x)_k bold(1)$:
$ r_(j k) = cos(theta) = frac(bold(d)_j^T bold(d)_k, \|bold(d)_j\| \|bold(d)_k\|) $

---

= 2.4 The Multivariate Normal (MVN) Distribution

== Probability Density Function
A $p$-dimensional random vector $bold(X)$ follows a Multivariate Normal distribution $bold(X) tilde.op cal(N)_p(bold(mu), bold(Sigma))$ (with positive definite $bold(Sigma) succ 0$) if its joint PDF is:

$ f(bold(x)) = frac(1, (2pi)^(p/2) |bold(Sigma)|^(1/2)) exp( -frac(1, 2) (bold(x) - bold(mu))^T bold(Sigma)^(-1) (bold(x) - bold(mu)) ) $

== Key Properties of MVN
1. *Linear Combinations:* If $bold(X) tilde.op cal(N)_p(bold(mu), bold(Sigma))$ and $bold(A) in bb(R)^(q times p)$, then $bold(A) bold(X) + bold(b) tilde.op cal(N)_q(bold(A) bold(mu) + bold(b), bold(A) bold(Sigma) bold(A)^T)$.
2. *Marginals and Conditionals:* All marginal and conditional distributions of an MVN vector are themselves Multivariate Normal.
3. *Zero Covariance Implies Independence:* For MVN distributions, $op("Cov")(X_j, X_k) = 0 <==> X_j perp X_k$.

== Geometry of Contour Ellipsoids & Mahalanobis Distance
The exponent in the MVN density defines the *squared Mahalanobis distance*:
$ D^2(bold(x), bold(mu)) = (bold(x) - bold(mu))^T bold(Sigma)^(-1) (bold(x) - bold(mu)) $

- The surfaces of constant probability density are hyper-ellipsoids centered at $bold(mu)$:
  $ \{ bold(x) in bb(R)^p : (bold(x) - bold(mu))^T bold(Sigma)^(-1) (bold(x) - bold(mu)) = c^2 \} $
- The principal axes of the ellipsoid align with the eigenvectors $bold(v)_1, dots.h, bold(v)_p$ of $bold(Sigma)$, with half-lengths proportional to $plus.minus c sqrt(lambda_j)$.
- Under multivariate normality:
  $ D^2(bold(X), bold(mu)) tilde.op chi_p^2 $

---

= 2.5 Lost, Null, and Incorrect Values (Data Quality & Imputation)

== Missing Data Mechanisms
- *Missing Completely at Random (MCAR):* The probability of missingness is independent of both observed and unobserved data: $P(M | Y_("obs"), Y_("mis")) = P(M)$.
- *Missing at Random (MAR):* Missingness depends on observed variables but not unobserved values: $P(M | Y_("obs"), Y_("mis")) = P(M | Y_("obs"))$.
- *Missing Not at Random (MNAR):* Missingness depends directly on the unobserved value itself.

== Imputation Strategies
1. *Mean/Median Imputation:* Simple but distorts variance and attenuates covariance structures.
2. *K-Nearest Neighbors (KNN) Imputation:* Replaces missing values with the weighted average of the $K$ closest complete records using Euclidean/Gower distance.
3. *Iterative Imputer (MICE - Multivariate Imputation by Chained Equations):* Models each feature with missing values as a function of all other features in a round-robin Gibbs sampling scheme.

---

= 2.6 Multivariate Aberrant Data (Outlier Detection)

== Univariate vs. Multivariate Outliers
An observation may have perfectly normal marginal values on each individual coordinate $X_j$ while being a severe multivariate outlier because it violates the joint correlation structure $bold(Sigma)$.

== Diagnostic Metrics
- *Sample Mahalanobis Squared Distance:*
  $ D_i^2 = (bold(x)_i - overline(bold(x)))^T bold(S)^(-1) (bold(x)_i - overline(bold(x))) $
- *Chi-Square Outlier Rule:* Flag observation $i$ as an outlier if $D_i^2 > chi_(p, 1 - alpha)^2$ (typically $alpha = 0.001$).
- *Robust Covariance Estimation (MCD):* Classical $overline(bold(x))$ and $bold(S)$ are sensitive to outliers (breakdown point $= 1/n$). The *Minimum Covariance Determinant (MCD)* finds the subset of $h lt.eq n$ observations whose sample covariance matrix has the smallest determinant, providing robust location $hat(bold(mu))_("MCD")$ and scatter $hat(bold(Sigma))_("MCD")$.

---

= 2.7 Sample Correlations: Fisher and Ruben Confidence Intervals

== Fisher's $z$-Transformation
The sampling distribution of Pearson's $r$ is highly skewed when $|rho| > 0$. Fisher's transformation normalizes the distribution:

$ z = frac(1, 2) ln( frac(1 + r, 1 - r) ) = op("arctanh")(r) $

Asymptotically:
$ z attach(tilde.op, t: "approx") cal(N)( frac(1, 2)ln(frac(1+rho, 1-rho)), frac(1, n - 3) ) $

$100(1 - alpha)%$ Confidence Interval for $zeta = op("arctanh")(rho)$:
$ [z_L, z_U] = z plus.minus z_(1 - alpha/2) frac(1, sqrt(n - 3)) $

Back-transforming to correlation scale:
$ rho_L = tanh(z_L) = frac(exp(2 z_L) - 1, exp(2 z_L) + 1), quad quad rho_U = tanh(z_U) = frac(exp(2 z_U) - 1, exp(2 z_U) + 1) $

---

= 2.8 Multivariate Descriptive Analytics & Visualization

== Multidimensional Plotting Techniques
- *Scatter Plot Matrix (SPLOM):* Visualizes all $p(p-1)/2$ pairwise bivariate relationships simultaneously with diagonal univariate kernel density estimates.
- *Correlation Heatmaps:* Displays $bold(R)$ with color intensity gradients and optional hierarchical clustering dendrograms.
- *Andrews Curves:* Maps each $p$-dimensional observation $bold(x)_i = [x_(i 1), dots.h, x_(i p)]^T$ to a continuous trigonometric function over $t in [-pi, pi]$:
  $ f_(bold(x)_i)(t) = frac(x_(i 1), sqrt(2)) + x_(i 2)sin(t) + x_(i 3)cos(t) + x_(i 4)sin(2t) + x_(i 5)cos(2t) + dots.h $
  Similar multivariate observations appear as closely clustered curves in function space.

---

= Practical Python Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.covariance import MinCovDet

# 1. Imputation of Missing Values (KNN)
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])

# 2. Sample Mean Vector & Covariance Matrix
mean_vec = X_imputed.mean()
cov_mat = X_imputed.cov()
corr_mat = X_imputed.corr()

# 3. Mahalanobis Distance Outlier Detection
inv_cov = np.linalg.inv(cov_mat)
diff = X_imputed - mean_vec
mahal_d2 = np.sum(diff.dot(inv_cov) * diff, axis=1)
cutoff = stats.chi2.ppf(0.999, df=X_imputed.shape[1])
outliers = X_imputed[mahal_d2 > cutoff]
print(f"Detected {len(outliers)} multivariate outliers (cutoff = {cutoff:.2f})")

# 4. Fisher z Confidence Interval for Correlation
r = corr_mat.loc["pm25", "pm10"]
n = len(X_imputed)
z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)
ci_z = (z - 1.96 * se, z + 1.96 * se)
ci_r = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
print(f"PM2.5 vs PM10: r = {r:.4f}, 95% CI = [{ci_r[0]:.4f}, {ci_r[1]:.4f}]")
```
