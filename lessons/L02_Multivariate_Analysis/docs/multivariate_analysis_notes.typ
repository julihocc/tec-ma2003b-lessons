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
Multivariate statistical methods extend classical single-variable inference to systems where $p \ge 2$ interconnected random variables are observed simultaneously. In *MA2003B*, this module provides the fundamental matrix algebra, probability theory, and diagnostic toolset required for all downstream dimensionality reduction (PCA, Factor Analysis) and supervised classification (Discriminant Analysis) techniques.

== Learning Objectives
By the end of this chapter, students will be able to:
- Formulate random vectors, joint probability density functions, and marginal/conditional expectations.
- Compute and interpret sample mean vectors ($\bar{\mathbf{x}}$), sample covariance matrices ($\mathbf{S}$), and sample correlation matrices ($\mathbf{R}$).
- Analyze the mathematical properties and geometry of the Multivariate Normal (MVN) distribution.
- Construct constant-density contour ellipsoids and compute Mahalanobis generalized distances.
- Identify missing data mechanisms (MCAR, MAR, MNAR) and apply modern imputation algorithms (KNN, MICE).
- Detect multivariate outliers using Mahalanobis distances and Robust Minimum Covariance Determinant (MCD) estimators.
- Derive Fisher $z$-transformations and construct confidence intervals for correlation parameters ($\rho$).
- Apply multidimensional visualization techniques including scatter matrices, correlation heatmaps, and Andrews curves.

---

= 2.1 Multivariate Distributions & Random Vectors

== Random Vectors
Let $\mathbf{X} = [X_1, X_2, \dots, X_p]^T$ be a $p$-dimensional column random vector where each component $X_j$ is a real-valued random variable.

== Joint Cumulative Distribution Function (CDF)
$ F(\mathbf{x}) = P(X_1 \le x_1, X_2 \le x_2, \dots, X_p \le x_p) $

== Joint Probability Density Function (PDF)
For continuous random vectors with density $f(\mathbf{x}) = f(x_1, \dots, x_p)$:
$ P(\mathbf{X} \in A) = \int \dots \int_A f(x_1, \dots, x_p) \, d x_1 \dots d x_p $

Marginal density for subset $\mathbf{X}_1$:
$ f_1(\mathbf{x}_1) = \int_{\mathbb{R}^{p - k}} f(\mathbf{x}_1, \mathbf{x}_2) \, d \mathbf{x}_2 $

Conditional density of $\mathbf{X}_1$ given $\mathbf{X}_2 = \mathbf{x}_2$:
$ f(\mathbf{x}_1 | \mathbf{x}_2) = \frac{f(\mathbf{x}_1, \mathbf{x}_2)}{f_2(\mathbf{x}_2)}, \quad \text{provided } f_2(\mathbf{x}_2) > 0 $

---

= 2.2 Mean Vectors and Covariance Matrices

== Population Mean Vector
$ \boldsymbol{\mu} = \mathbb{E}[\mathbf{X}] = \begin{bmatrix} \mathbb{E}[X_1] \\ \mathbb{E}[X_2] \\ \vdots \\ \mathbb{E}[X_p] \end{bmatrix} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_p \end{bmatrix} $

== Population Covariance Matrix
$ \mathbf{\Sigma} = \operatorname{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T] = \begin{bmatrix} \sigma_{11} & \sigma_{12} & \dots & \sigma_{1p} \\ \sigma_{21} & \sigma_{22} & \dots & \sigma_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{p 1} & \sigma_{p 2} & \dots & \sigma_{p p} \end{bmatrix} $

where $\sigma_{j j} = \operatorname{Var}(X_j) = \sigma_j^2$ and $\sigma_{j k} = \operatorname{Cov}(X_j, X_k) = \sigma_{k j}$.

#info[
  *Positive Semi-Definiteness:* For any non-zero constant vector $\mathbf{a} \in \mathbb{R}^p$, $\operatorname{Var}(\mathbf{a}^T \mathbf{X}) = \mathbf{a}^T \mathbf{\Sigma} \mathbf{a} \ge 0$. Hence, all valid covariance matrices $\mathbf{\Sigma}$ are symmetric positive semi-definite ($\mathbf{\Sigma} \succeq 0$).
]

== Sample Statistics Matrix Operations
Given a data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ with $n$ observations:

- *Sample Mean Vector:*
  $ \bar{\mathbf{x}} = \frac{1}{n} \mathbf{X}^T \mathbf{1}_n = \begin{bmatrix} \bar{x}_1 & \bar{x}_2 & \dots & \bar{x}_p \end{bmatrix}^T $
- *Sample Covariance Matrix:*
  $ \mathbf{S} = \frac{1}{n - 1} \sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T = \frac{1}{n - 1} \mathbf{X}^T \left( \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n\mathbf{1}_n^T \right) \mathbf{X} $

---

= 2.3 Correlations and Correlation Matrices

== Population and Sample Correlation
Let $\mathbf{D} = \operatorname{diag}(\sigma_{11}, \sigma_{22}, \dots, \sigma_{p p})$ be the diagonal standard deviation matrix.

- *Population Correlation Matrix:*
  $ \mathbf{P} = \mathbf{D}^{-1/2} \mathbf{\Sigma} \mathbf{D}^{-1/2} = \begin{bmatrix} 1 & \rho_{12} & \dots & \rho_{1p} \\ \rho_{21} & 1 & \dots & \rho_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ \rho_{p 1} & \rho_{p 2} & \dots & 1 \end{bmatrix}, \quad \rho_{j k} = \frac{\sigma_{j k}}{\sigma_j \sigma_k} \in [-1, 1] $
- *Sample Correlation Matrix:*
  $ \mathbf{R} = \mathbf{D}_{\mathbf{S}}^{-1/2} \mathbf{S} \mathbf{D}_{\mathbf{S}}^{-1/2}, \quad r_{j k} = \frac{s_{j k}}{s_j s_k} $

== Geometric Interpretation
In sample space $\mathbb{R}^n$, the sample correlation coefficient $r_{j k}$ equals the *cosine of the angle* $\theta$ between the mean-centered observation vectors $\mathbf{d}_j = \mathbf{x}_j - \bar{x}_j \mathbf{1}$ and $\mathbf{d}_k = \mathbf{x}_k - \bar{x}_k \mathbf{1}$:
$ r_{j k} = \cos(\theta) = \frac{\mathbf{d}_j^T \mathbf{d}_k}{\|\mathbf{d}_j\| \|\mathbf{d}_k\|} $

---

= 2.4 The Multivariate Normal (MVN) Distribution

== Probability Density Function
A $p$-dimensional random vector $\mathbf{X}$ follows a Multivariate Normal distribution $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \mathbf{\Sigma})$ (with positive definite $\mathbf{\Sigma} \succ 0$) if its joint PDF is:

$ f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\mathbf{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) $

== Key Properties of MVN
1. *Linear Combinations:* If $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \mathbf{\Sigma})$ and $\mathbf{A} \in \mathbb{R}^{q \times p}$, then $\mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}_q(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\mathbf{\Sigma}\mathbf{A}^T)$.
2. *Marginals and Conditionals:* All marginal and conditional distributions of an MVN vector are themselves Multivariate Normal.
3. *Zero Covariance Implies Independence:* For MVN distributions, $\operatorname{Cov}(X_j, X_k) = 0 \iff X_j \perp X_k$.

== Geometry of Contour Ellipsoids & Mahalanobis Distance
The exponent in the MVN density defines the *squared Mahalanobis distance*:
$ D^2(\mathbf{x}, \boldsymbol{\mu}) = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) $

- The surfaces of constant probability density are hyper-ellipsoids centered at $\boldsymbol{\mu}$:
  $ \{ \mathbf{x} \in \mathbb{R}^p : (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = c^2 \} $
- The principal axes of the ellipsoid align with the eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_p$ of $\mathbf{\Sigma}$, with half-lengths proportional to $\pm c \sqrt{\lambda_j}$.
- Under multivariate normality:
  $ D^2(\mathbf{X}, \boldsymbol{\mu}) \sim \chi_p^2 $

---

= 2.5 Lost, Null, and Incorrect Values (Data Quality & Imputation)

== Missing Data Mechanisms
- *Missing Completely at Random (MCAR):* The probability of missingness is independent of both observed and unobserved data: $P(M | Y_{\text{obs}}, Y_{\text{mis}}) = P(M)$.
- *Missing at Random (MAR):* Missingness depends on observed variables but not unobserved values: $P(M | Y_{\text{obs}}, Y_{\text{mis}}) = P(M | Y_{\text{obs}})$.
- *Missing Not at Random (MNAR):* Missingness depends directly on the unobserved value itself.

== Imputation Strategies
1. *Mean/Median Imputation:* Simple but distorts variance and attenuates covariance structures.
2. *K-Nearest Neighbors (KNN) Imputation:* Replaces missing values with the weighted average of the $K$ closest complete records using Euclidean/Gower distance.
3. *Iterative Imputer (MICE - Multivariate Imputation by Chained Equations):* Models each feature with missing values as a function of all other features in a round-robin Gibbs sampling scheme.

---

= 2.6 Multivariate Aberrant Data (Outlier Detection)

== Univariate vs. Multivariate Outliers
An observation may have perfectly normal marginal values on each individual coordinate $X_j$ while being a severe multivariate outlier because it violates the joint correlation structure $\mathbf{\Sigma}$.

== Diagnostic Metrics
- *Sample Mahalanobis Squared Distance:*
  $ D_i^2 = (\mathbf{x}_i - \bar{\mathbf{x}})^T \mathbf{S}^{-1} (\mathbf{x}_i - \bar{\mathbf{x}}) $
- *Chi-Square Outlier Rule:* Flag observation $i$ as an outlier if $D_i^2 > \chi_{p, 1 - \alpha}^2$ (typically $\alpha = 0.001$).
- *Robust Covariance Estimation (MCD):* Classical $\bar{\mathbf{x}}$ and $\mathbf{S}$ are sensitive to outliers (breakdown point $= 1/n$). The *Minimum Covariance Determinant (MCD)* finds the subset of $h \le n$ observations whose sample covariance matrix has the smallest determinant, providing robust location $\hat{\boldsymbol{\mu}}_{\text{MCD}}$ and scatter $\hat{\mathbf{\Sigma}}_{\text{MCD}}$.

---

= 2.7 Sample Correlations: Fisher and Ruben Confidence Intervals

== Fisher's $z$-Transformation
The sampling distribution of Pearson's $r$ is highly skewed when $|\rho| > 0$. Fisher's transformation normalizes the distribution:

$ z = \frac{1}{2} \ln\left( \frac{1 + r}{1 - r} \right) = \operatorname{arctanh}(r) $

Asymptotically:
$ z \stackrel{\text{approx}}{\sim} \mathcal{N}\left( \frac{1}{2}\ln\left(\frac{1+\rho}{1-\rho}\right), \frac{1}{n - 3} \right) $

$100(1 - \alpha)\%$ Confidence Interval for $\zeta = \operatorname{arctanh}(\rho)$:
$ [z_L, z_U] = z \pm z_{1 - \alpha/2} \frac{1}{\sqrt{n - 3}} $

Back-transforming to correlation scale:
$ \rho_L = \tanh(z_L) = \frac{\exp(2 z_L) - 1}{\exp(2 z_L) + 1}, \qquad \rho_U = \tanh(z_U) = \frac{\exp(2 z_U) - 1}{\exp(2 z_U) + 1} $

---

= 2.8 Multivariate Descriptive Analytics & Visualization

== Multidimensional Plotting Techniques
- *Scatter Plot Matrix (SPLOM):* Visualizes all $p(p-1)/2$ pairwise bivariate relationships simultaneously with diagonal univariate kernel density estimates.
- *Correlation Heatmaps:* Displays $\mathbf{R}$ with color intensity gradients and optional hierarchical clustering dendrograms.
- *Andrews Curves:* Maps each $p$-dimensional observation $\mathbf{x}_i = [x_{i 1}, \dots, x_{i p}]^T$ to a continuous trigonometric function over $t \in [-\pi, \pi]$:
  $ f_{\mathbf{x}_i}(t) = \frac{x_{i 1}}{\sqrt{2}} + x_{i 2}\sin(t) + x_{i 3}\cos(t) + x_{i 4}\sin(2t) + x_{i 5}\cos(2t) + \dots $
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
