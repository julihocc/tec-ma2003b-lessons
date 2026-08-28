# Chapter 7 Lecture Notes (`lessons/L07_Multivariate_Regression/notes/`)

This directory contains the theoretical lecture notes and study guides for **Chapter 7: Multivariate Regression**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`multivariate_regression_notes.typ`](multivariate_regression_notes.typ) | Typst Source | Comprehensive study guide covering all syllabus subtopics (7.1–7.7) with Logistic regression, inference for covariance matrices (Wishart distribution, Box's M), Hotelling's $T^2$, one-way/two-way MANOVA test statistics, and Canonical Correlation Analysis (CCA) matrix derivations. |
| [`multivariate_regression_notes.pdf`](multivariate_regression_notes.pdf) | PDF (Git LFS) | Fully compiled, publication-ready PDF document typeset with Typst. |

---

## 📖 Theoretical Topics Covered

1. **Logistic Regression Model (7.1):** Log-odds formulation $\ln\left(\frac{p}{1-p}\right) = \mathbf{x}^T\boldsymbol{\beta}$, Maximum Likelihood Estimation (MLE), Fisher scoring/Newton-Raphson, Wald tests, Deviance and Likelihood Ratio tests.
2. **Inferences for Covariance Matrices (7.2):** Wishart distribution $\mathcal{W}_p(n-1, \mathbf{\Sigma})$, likelihood ratio tests for $\mathbf{\Sigma} = \mathbf{\Sigma}_0$, sphericity test ($\mathbf{\Sigma} = \sigma^2 \mathbf{I}$), and Box's M test for equality of $K$ covariance matrices.
3. **Inferences for Mean Vectors (7.3):** Single-sample Hotelling's $T^2 = n(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^T\mathbf{S}^{-1}(\bar{\mathbf{x}} - \boldsymbol{\mu}_0) \sim \frac{p(n-1)}{n-p} F_{p, n-p}$, simultaneous confidence intervals (Bonferroni, $T^2$ ellipsoids).
4. **MANOVA (7.4):** Multivariate Analysis of Variance:
   - Decomposition: $\mathbf{T} = \mathbf{B} + \mathbf{W}$ (Total, Between, Within scatter matrices).
   - Test statistics: Wilks' Lambda $\Lambda = \frac{|\mathbf{W}|}{|\mathbf{B}+\mathbf{W}|}$, Pillai's trace $V = \operatorname{tr}(\mathbf{B}(\mathbf{B}+\mathbf{W})^{-1})$, Hotelling-Lawley trace $U = \operatorname{tr}(\mathbf{W}^{-1}\mathbf{B})$, Roy's largest root $\theta = \frac{\lambda_1}{1+\lambda_1}$.
5. **Canonical Correlation Analysis - CCA (7.5):** Joint linear combinations $U_1 = \mathbf{a}_1^T \mathbf{X}_1$ and $V_1 = \mathbf{b}_1^T \mathbf{X}_2$ maximizing correlation $\operatorname{Corr}(U_1, V_1)$, generalized eigenvalue problem $\mathbf{\Sigma}_{11}^{-1}\mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}\mathbf{\Sigma}_{21}\mathbf{a} = \rho^2 \mathbf{a}$.
6. **Factor & Regression Synthesis (7.6):** Combining dimensionality reduction with predictive modeling (Reduced-Rank Regression).
7. **Software & Coding (7.7):** Implementation in Python using `statsmodels.multivariate.manova.MANOVA` and `sklearn.cross_decomposition.CCA`.

---

## 🛠️ Compilation Command

```bash
cd lessons/L07_Multivariate_Regression/notes
typst compile multivariate_regression_notes.typ
```
