# Chapter 2 Lecture Notes (`lessons/L02_Multivariate_Analysis/notes/`)

This directory contains the theoretical lecture notes and study guides for **Chapter 2: Multivariate Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`multivariate_analysis_notes.typ`](multivariate_analysis_notes.typ) | Typst Source | Comprehensive study guide covering all syllabus subtopics (2.1–2.8) with random vector algebra, MVN density functions, contour ellipsoids, Mahalanobis distances, imputation theory, and Python code blocks. |
| [`multivariate_analysis_notes.pdf`](multivariate_analysis_notes.pdf) | PDF (Git LFS) | Fully compiled, publication-ready PDF document typeset with Typst. |

---

## 📖 Theoretical Topics Covered

1. **Multivariate Distributions & Random Vectors (2.1):** Column random vectors $\mathbf{X} \in \mathbb{R}^p$, joint CDF, joint PDF, marginal and conditional density functions.
2. **Mean Vectors & Covariance Matrices (2.2):** Population parameters $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$, $\mathbf{\Sigma} = \operatorname{Cov}(\mathbf{X})$, sample statistics $\bar{\mathbf{x}}$, $\mathbf{S}$, positive semi-definiteness ($\mathbf{\Sigma} \succeq 0$).
3. **Correlations & Correlation Matrices (2.3):** Matrix normalization $\mathbf{R} = \mathbf{D}_{\mathbf{S}}^{-1/2}\mathbf{S}\mathbf{D}_{\mathbf{S}}^{-1/2}$, geometric interpretation ($r_{jk} = \cos \theta$).
4. **Multivariate Normal (MVN) Distribution (2.4):** Joint PDF $f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\mathbf{\Sigma}|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$, linear combinations, constant-density contour ellipsoids, Mahalanobis distance $D^2 \sim \chi_p^2$.
5. **Missing Value Mechanisms & Imputation (2.5):** MCAR, MAR, MNAR definitions, Listwise deletion pitfalls, KNN imputation, MICE chained equations.
6. **Multivariate Aberrant Data (2.6):** Univariate vs. multivariate outliers, sample Mahalanobis distance cutoff ($D_i^2 > \chi_{p, 1-\alpha}^2$), Robust Minimum Covariance Determinant (MCD).
7. **Sample Correlations: Fisher & Ruben Intervals (2.7):** Fisher's $z = \operatorname{arctanh}(r)$, asymptotic normality $\mathcal{N}(\operatorname{arctanh}(\rho), \frac{1}{n-3})$, back-transformation to correlation confidence interval $[\rho_L, \rho_U]$.
8. **Multivariate Visualization (2.8):** Scatter plot matrices (SPLOM), correlation heatmaps, and Andrews curves.

---

## 🛠️ Compilation Command

```bash
cd lessons/L02_Multivariate_Analysis/notes
typst compile multivariate_analysis_notes.typ
```
