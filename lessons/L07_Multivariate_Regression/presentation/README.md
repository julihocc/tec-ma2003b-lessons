# Chapter 7 Presentation — Multivariate Regression Slides (`lessons/L07_Multivariate_Regression/presentation/`)

This directory contains the classroom slide presentation for **Chapter 7: Multivariate Regression**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`multivariate_regression_slides.typ`](multivariate_regression_slides.typ) | Typst Source | Slide deck built with [Touying](https://github.com/touying-typ/touying) using the `university-theme` and Tecnológico de Monterrey branding (`#003366` / `#1E88E5`). |
| [`multivariate_regression_slides.pdf`](multivariate_regression_slides.pdf) | PDF (Git LFS) | Fully compiled, 16:9 widescreen presentation PDF. |

---

## 📽️ Slide Deck Outline

1. **Foundations of Multi-Response Models:** Multiple predictors vs. Multiple responses ($\mathbf{Y} \in \mathbb{R}^m$).
2. **Multivariate Linear Regression:** Matrix formulation $\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}$ and cross-equation error covariance $\mathbf{\Sigma}_{\mathbf{E}}$.
3. **Hotelling's $T^2$ Inference:** Generalizing the Student's $t$-test to multivariate mean vector hypotheses.
4. **Multivariate Analysis of Variance (MANOVA):** Total, Between, and Within scatter decomposition ($\mathbf{T} = \mathbf{B} + \mathbf{W}$).
5. **The 4 Classic MANOVA Test Statistics:** Wilks' $\Lambda$, Pillai's trace, Hotelling-Lawley trace, and Roy's largest root.
6. **Logistic Regression & Odds Ratios:** Maximum Likelihood estimation for binary categorical responses.
7. **Canonical Correlation Analysis (CCA):** Finding maximal cross-set correlations between two battery sets of variables.
8. **Cardiovascular Risk Case Study:** Multi-response clinical biomarker predictions.
9. **Course Synthesis & Capstone Overview:** Connecting all 7 multivariate techniques for real-world decision systems.

---

## 🛠️ Compilation Command

```bash
cd lessons/L07_Multivariate_Regression/presentation
typst compile multivariate_regression_slides.typ
```
