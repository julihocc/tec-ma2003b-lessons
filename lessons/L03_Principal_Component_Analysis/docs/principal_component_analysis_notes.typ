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
Principal Component Analysis (PCA) is the primary unsupervised dimensionality reduction technique in multivariate data science. Introduced by Karl Pearson (1901) and generalized by Harold Hotelling (1933), PCA transforms a system of $p$ correlated variables into $k \ll p$ mutually orthogonal components that capture the maximum possible variance.

== Learning Objectives
By the end of this chapter, students will be able to:
- Identify appropriate practical and scientific use cases for PCA.
- Understand the geometric interpretation of PCA as an orthogonal rotation along the principal axes of the covariance ellipsoid.
- Derive the principal component formulation via spectral decomposition of $\mathbf{S}$ and $\mathbf{R}$ using Lagrange multipliers.
- Calculate component loadings, individual scores, and proportions of variance explained.
- Apply rigorous component retention criteria (Kaiser criterion, Scree plots, Horn's Parallel Analysis).
- Construct and interpret 2D/3D PCA Biplots.
- Implement end-to-end PCA pipelines in Python using `scikit-learn`.

---

= 3.1 Cases Where PCA is Used

== Primary Applications
1. **Multicollinearity Elimination:** Replaces highly correlated regression predictors with uncorrelated principal components (Principal Component Regression, PCR).
2. **High-Dimensional Data Compression:** Reduces memory footprint and computational cost in downstream machine learning algorithms while retaining $>80\%$ of systemic variance.
3. **Exploratory 2D/3D Visualization:** Projects high-dimensional datasets ($p \ge 10$) onto principal score planes to identify natural clustering, trajectories, and outliers.
4. **Macroeconomic & Financial Risk Factor Modeling:** Extracts systematic market drivers (e.g. Yield curve level, slope, curvature) from multi-asset return streams.

#info[
  *PCA vs. Factor Analysis:* PCA is a variance-focused mathematical transformation of all observed variance (common + unique). In contrast, Factor Analysis posits an underlying measurement model focusing strictly on shared common variance ($\mathbf{\Sigma} = \mathbf{\Lambda}\mathbf{\Lambda}^T + \mathbf{\Psi}$).
]

---

= 3.2 Geometrical Description of Major Components

== Geometric Rotation of Axes
Let $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^p$ be a sample of observations centered at $\bar{\mathbf{x}} = \mathbf{0}$.
- The *first principal component* $Y_1 = \mathbf{a}_1^T \mathbf{X}$ defines the direction in $\mathbb{R}^p$ along which the orthogonal projection of the data points exhibits *maximum dispersion (variance)*.
- The *second principal component* $Y_2 = \mathbf{a}_2^T \mathbf{X}$ defines the direction of maximum remaining variance, subject to being *orthogonal* to $\mathbf{a}_1$ ($\mathbf{a}_2^T \mathbf{a}_1 = 0$).
- Geometrically, the eigenvectors of $\mathbf{S}$ specify the orientation of the principal axes of the constant-density ellipsoid $\{\mathbf{x} : \mathbf{x}^T \mathbf{S}^{-1} \mathbf{x} = c^2\}$.

---

= 3.3 Mathematical Formulation & Spectral Decomposition

== Derivation via Lagrange Multipliers
Let $\mathbf{X}$ be a random vector with covariance matrix $\mathbf{\Sigma}$ (or sample covariance $\mathbf{S}$).
We seek a linear combination $Y_1 = \mathbf{a}_1^T \mathbf{X}$ that maximizes:

$ \operatorname{Var}(Y_1) = \mathbf{a}_1^T \mathbf{\Sigma} \mathbf{a}_1 \quad \text{subject to } \mathbf{a}_1^T \mathbf{a}_1 = 1 $

The Lagrangian is:
$ \mathcal{L}(\mathbf{a}_1, \lambda_1) = \mathbf{a}_1^T \mathbf{\Sigma} \mathbf{a}_1 - \lambda_1 (\mathbf{a}_1^T \mathbf{a}_1 - 1) $

Setting the gradient to zero:
$ \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = 2 \mathbf{\Sigma} \mathbf{a}_1 - 2 \lambda_1 \mathbf{a}_1 = \mathbf{0} \implies \mathbf{\Sigma} \mathbf{a}_1 = \lambda_1 \mathbf{a}_1 $

Thus:
- $\lambda_1$ is the *largest eigenvalue* of $\mathbf{\Sigma}$.
- $\mathbf{a}_1$ is the corresponding *normalized eigenvector* ($\|\mathbf{a}_1\| = 1$).
- Maximum variance equals the eigenvalue: $\operatorname{Var}(Y_1) = \mathbf{a}_1^T \mathbf{\Sigma} \mathbf{a}_1 = \lambda_1$.

For the $j$-th component:
$ \mathbf{\Sigma} \mathbf{a}_j = \lambda_j \mathbf{a}_j, \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0 $

== Spectral Decomposition of Covariance ($\mathbf{S}$) and Correlation ($\mathbf{R}$)
$ \mathbf{S} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T = \sum_{j=1}^p \lambda_j \mathbf{v}_j \mathbf{v}_j^T $

where $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_p]$ is the orthogonal eigenvector matrix ($\mathbf{V}^T \mathbf{V} = \mathbf{I}$), and $\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_p)$.

== Total Variance and Variance Explained
$ \text{Total Variance} = \operatorname{tr}(\mathbf{S}) = \sum_{j=1}^p s_{j j} = \sum_{j=1}^p \lambda_j $

The proportion of total sample variance explained by the $j$-th principal component is:
$ \text{Variance Proportion}_j = \frac{\lambda_j}{\sum_{k=1}^p \lambda_k} $

== Component Loadings (Correlations)
The correlation between the original variable $X_k$ and the $j$-th component $Y_j$ is:
$ \operatorname{Corr}(X_k, Y_j) = \frac{a_{j k} \sqrt{\lambda_j}}{\sqrt{s_{k k}}} $

When PCA is performed on the *correlation matrix* $\mathbf{R}$ (where $s_{k k} = 1$):
$ \operatorname{Corr}(X_k, Y_j) = a_{j k} \sqrt{\lambda_j} $

---

= 3.4 Determination of the Number of Major Components

Several complementary rules guide component retention:

#table(
  columns: (2fr, 3fr, 2fr),
  align: (left, left, left),
  table.header([*Criterion*], [*Mathematical Rule*], [*Best Use Context*]),
  [Kaiser-Guttman Rule], [Retain components where $\lambda_j > 1.0$ (for $\mathbf{R}$) or $\lambda_j > \bar{\lambda}$ (for $\mathbf{S}$)], [Standard correlation matrix PCA],
  [Cattell Scree Plot], [Visually locate the "elbow" (inflection point) where eigenvalues level off], [Exploratory visual inspection],
  [Cumulative Variance], [Retain minimum $k$ such that $\sum_{j=1}^k \lambda_j / \sum \lambda_m \ge 70\% - 85\%$], [Data compression goals],
  [Horn's Parallel Analysis], [Retain $\lambda_j$ that exceeds the 95th percentile of eigenvalues from random uncorrelated noise], [Rigorous statistical benchmark]
)

---

= 3.5 Biplots and Interpretation in Python

== The PCA Biplot (Gabriel 1971)
A Biplot simultaneously displays two sets of coordinates on the first two principal axes:
1. **Observation Scores ($Y_{i 1}, Y_{i 2}$):** Plotted as scatter points showing sample clustering.
2. **Variable Loading Vectors ($\mathbf{a}_1, \mathbf{a}_2$):** Plotted as vectors emanating from the origin.
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
