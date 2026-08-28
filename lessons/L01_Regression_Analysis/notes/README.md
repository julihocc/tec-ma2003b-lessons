# Chapter 1 Lecture Notes (`lessons/L01_Regression_Analysis/notes/`)

This directory contains the theoretical lecture notes and study guides for **Chapter 1: Regression Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`regression_analysis_notes.typ`](regression_analysis_notes.typ) | Typst Source | Comprehensive study guide covering all syllabus subtopics (1.1–1.8) with mathematical rigor, OLS derivations, Gauss-Markov theorem, ANOVA tables, and Python code blocks. |
| [`regression_analysis_notes.pdf`](regression_analysis_notes.pdf) | PDF (Git LFS) | Fully compiled, publication-ready PDF document typeset with Typst. |

---

## 📖 Theoretical Topics Covered

1. **Simple Linear Regression (1.1):** Model formulation $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$, OLS derivation via normal equations, Gauss-Markov theorem (BLUE property).
2. **ANOVA & Intervals (1.2):** Total variation decomposition ($\text{SST} = \text{SSR} + \text{SSE}$), $F$-test for overall significance, Confidence Interval for $\mathbb{E}[Y|X_0]$ vs. Prediction Interval for $Y_{\text{new}}$.
3. **Residual Analysis & Normality (1.3):** Raw, standardized, and studentized residuals, Q-Q plots, Shapiro-Wilk test.
4. **Variable Selection (1.4 & 1.5):** Forward selection and Backward elimination algorithms, partial $F$-tests, AIC/BIC penalization.
5. **Non-Linear Models (1.6):** Polynomial regression, Log-Linear, Linear-Log, and Log-Log elasticity models.
6. **Multiple Linear Regression (1.7):** Matrix formulation $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$, $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, Variance Inflation Factor (VIF).
7. **Aberrant Data & Heteroskedasticity (1.8):** Hat matrix leverage ($h_{ii}$), Cook's distance ($D_i$), Breusch-Pagan test, Weighted Least Squares (WLS), Huber-White HC3 robust standard errors.

---

## 🛠️ Compilation Command

```bash
cd lessons/L01_Regression_Analysis/notes
typst compile regression_analysis_notes.typ
```
