// Principal Component Analysis (PCA) - Lecture Notes
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
  title: "Principal Component Analysis - Lecture Notes",
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
  #text(24pt, weight: "bold")[Principal Component Analysis (PCA)]

  #text(16pt)[Dimensionality Reduction, Spectral Decomposition, and Geometry]

  #v(0.5em)

  #text(12pt)[MA2003B — Application of Multivariate Methods in Data Science]

  #v(0.5em)

  #text(11pt)[Tecnológico de Monterrey]

  #v(2em)
]

#pagebreak()

= Course and Chapter Overview

== Course Context
Principal Component Analysis (PCA) is the primary unsupervised dimensionality reduction technique in multivariate data science. Introduced by Karl Pearson (1901) and generalized by Harold Hotelling (1933), PCA transforms a system of $p$ correlated variables into $k lt.double p$ mutually orthogonal components that capture the maximum possible variance.

== Learning Objectives
By the end of this chapter, students will be able to:
- Identify appropriate practical and scientific use cases for PCA.
- Understand the geometric interpretation of PCA as an orthogonal rotation along the principal axes of the covariance ellipsoid.
- Derive the principal component formulation via spectral decomposition of $bold(S)$ and $bold(R)$ using Lagrange multipliers.
- Calculate component loadings, individual scores, and proportions of variance explained.
- Apply rigorous component retention criteria (Kaiser criterion, Scree plots, Horn's Parallel Analysis).
- Construct and interpret 2D/3D PCA Biplots.
- Implement end-to-end PCA pipelines in Python using `scikit-learn`.

---

= 3.1 Cases Where PCA is Used

== Primary Applications
1. **Multicollinearity Elimination:** Replaces highly correlated regression predictors with uncorrelated principal components (Principal Component Regression, PCR).
2. **High-Dimensional Data Compression:** Reduces memory footprint and computational cost in downstream machine learning algorithms while retaining $>80%$ of systemic variance.
3. **Exploratory 2D/3D Visualization:** Projects high-dimensional datasets ($p gt.eq 10$) onto principal score planes to identify natural clustering, trajectories, and outliers.
4. **Macroeconomic & Financial Risk Factor Modeling:** Extracts systematic market drivers (e.g. Yield curve level, slope, curvature) from multi-asset return streams.

#info[
  *PCA vs. Factor Analysis:* PCA is a variance-focused mathematical transformation of all observed variance (common + unique). In contrast, Factor Analysis posits an underlying measurement model focusing strictly on shared common variance ($bold(Sigma) = bold(Lambda) bold(Lambda)^T + bold(Psi)$).
]

---

= 3.2 Geometrical Description of Major Components

== Geometric Rotation of Axes
Let $bold(x)_1, dots.h, bold(x)_n in bb(R)^p$ be a sample of observations centered at $overline(bold(x)) = bold(0)$.
- The *first principal component* $Y_1 = bold(a)_1^T bold(X)$ defines the direction in $bb(R)^p$ along which the orthogonal projection of the data points exhibits *maximum dispersion (variance)*.
- The *second principal component* $Y_2 = bold(a)_2^T bold(X)$ defines the direction of maximum remaining variance, subject to being *orthogonal* to $bold(a)_1$ ($bold(a)_2^T bold(a)_1 = 0$).
- Geometrically, the eigenvectors of $bold(S)$ specify the orientation of the principal axes of the constant-density ellipsoid $\{bold(x) : bold(x)^T bold(S)^(-1) bold(x) = c^2\}$.

---

= 3.3 Mathematical Formulation & Spectral Decomposition

== Derivation via Lagrange Multipliers
Let $bold(X)$ be a random vector with covariance matrix $bold(Sigma)$ (or sample covariance $bold(S)$).
We seek a linear combination $Y_1 = bold(a)_1^T bold(X)$ that maximizes:

$ op("Var")(Y_1) = bold(a)_1^T bold(Sigma) bold(a)_1 quad "subject to " bold(a)_1^T bold(a)_1 = 1 $

The Lagrangian is:
$ cal(L)(bold(a)_1, lambda_1) = bold(a)_1^T bold(Sigma) bold(a)_1 - lambda_1 (bold(a)_1^T bold(a)_1 - 1) $

Setting the gradient to zero:
$ frac(partial cal(L), partial bold(a)_1) = 2 bold(Sigma) bold(a)_1 - 2 lambda_1 bold(a)_1 = bold(0) ==> bold(Sigma) bold(a)_1 = lambda_1 bold(a)_1 $

Thus:
- $lambda_1$ is the *largest eigenvalue* of $bold(Sigma)$.
- $bold(a)_1$ is the corresponding *normalized eigenvector* ($\|bold(a)_1\| = 1$).
- Maximum variance equals the eigenvalue: $op("Var")(Y_1) = bold(a)_1^T bold(Sigma) bold(a)_1 = lambda_1$.

For the $j$-th component:
$ bold(Sigma) bold(a)_j = lambda_j bold(a)_j, quad lambda_1 gt.eq lambda_2 gt.eq dots.h gt.eq lambda_p gt.eq 0 $

== Spectral Decomposition of Covariance ($bold(S)$) and Correlation ($bold(R)$)
$ bold(S) = bold(V) bold(Lambda) bold(V)^T = sum_(j=1)^p lambda_j bold(v)_j bold(v)_j^T $

where $bold(V) = [bold(v)_1, dots.h, bold(v)_p]$ is the orthogonal eigenvector matrix ($bold(V)^T bold(V) = bold(I)$), and $bold(Lambda) = op("diag")(lambda_1, dots.h, lambda_p)$.

== Total Variance and Variance Explained
$ "Total Variance" = op("tr")(bold(S)) = sum_(j=1)^p s_(j j) = sum_(j=1)^p lambda_j $

The proportion of total sample variance explained by the $j$-th principal component is:
$ "Variance Proportion"_j = frac(lambda_j, sum_(k=1)^p lambda_k) $

== Component Loadings (Correlations)
The correlation between the original variable $X_k$ and the $j$-th component $Y_j$ is:
$ op("Corr")(X_k, Y_j) = frac(a_(j k) sqrt(lambda_j), sqrt(s_(k k))) $

When PCA is performed on the *correlation matrix* $bold(R)$ (where $s_(k k) = 1$):
$ op("Corr")(X_k, Y_j) = a_(j k) sqrt(lambda_j) $

---

= 3.4 Determination of the Number of Major Components

Several complementary rules guide component retention:

#table(
  columns: (2fr, 3fr, 2fr),
  align: (left, left, left),
  table.header([*Criterion*], [*Mathematical Rule*], [*Best Use Context*]),
  [Kaiser-Guttman Rule], [Retain components where $lambda_j > 1.0$ (for $bold(R)$) or $lambda_j > overline(lambda)$ (for $bold(S)$)], [Standard correlation matrix PCA],
  [Cattell Scree Plot], [Visually locate the "elbow" (inflection point) where eigenvalues level off], [Exploratory visual inspection],
  [Cumulative Variance], [Retain minimum $k$ such that $sum_(j=1)^k lambda_j / sum lambda_m gt.eq 70% - 85%$], [Data compression goals],
  [Horn's Parallel Analysis], [Retain $lambda_j$ that exceeds the 95th percentile of eigenvalues from random uncorrelated noise], [Rigorous statistical benchmark]
)

---

= 3.5 Biplots and Interpretation in Python

== The PCA Biplot (Gabriel 1971)
A Biplot simultaneously displays two sets of coordinates on the first two principal axes:
1. **Observation Scores ($Y_(i 1), Y_(i 2)$):** Plotted as scatter points showing sample clustering.
2. **Variable Loading Vectors ($bold(a)_1, bold(a)_2$):** Plotted as vectors emanating from the origin.
   - Vector length indicates the strength of representation in the 2D subspace.
   - The cosine of the angle between two variable vectors approximates their correlation.

---

= Practical Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:, 1:])

# 2. Fit Full PCA
pca = PCA()
scores = pca.fit_transform(X_scaled)
eigenvalues = pca.explained_variance_
variance_ratio = pca.explained_variance_ratio_

# 3. Scree Plot & Kaiser Criterion
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', color='navy', lw=2)
plt.axhline(1.0, color='red', ls='--', label='Kaiser Threshold (λ = 1.0)')
plt.xlabel('Principal Component Index')
plt.ylabel('Eigenvalue (Variance Explained)')
plt.title('PCA Scree Plot')
plt.legend()
plt.show()

# 4. Component Loadings DataFrame
loadings = pd.DataFrame(
    pca.components_[:3].T * np.sqrt(pca.explained_variance_[:3]),
    index=df.columns[1:],
    columns=['PC1', 'PC2', 'PC3']
)
print("=== Component Loadings Matrix ===")
print(loadings.round(3))
```
