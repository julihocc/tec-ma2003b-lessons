# Chapter 1 Presentation — Regression Analysis Slides (`lessons/L01_Regression_Analysis/presentation/`)

This directory contains the classroom slide presentation for **Chapter 1: Regression Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`regression_analysis_slides.typ`](regression_analysis_slides.typ) | Typst Source | Slide deck built with [Touying](https://github.com/touying-typ/touying) using the `university-theme` and Tecnológico de Monterrey branding (`#003366` / `#1E88E5`). |
| [`regression_analysis_slides.pdf`](regression_analysis_slides.pdf) | PDF (Git LFS) | Fully compiled, 16:9 widescreen presentation PDF. |

---

## 📽️ Slide Deck Outline

1. **Foundations of Simple Linear Regression:** Population equations, Gauss-Markov BLUE theorem.
2. **ANOVA & Variation Decomposition:** Sum of squares table, $R^2$, and overall $F$-test.
3. **Confidence vs. Prediction Intervals:** Comparison and mathematical interpretation.
4. **Multiple Linear Regression:** Matrix formulation, $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, and hat matrix leverage.
5. **Multicollinearity:** Variance Inflation Factor (VIF) interpretation rules.
6. **Outlier Diagnostics & Influence:** Studentized residuals, leverage, and Cook's distance formula.
7. **Heteroskedasticity:** Breusch-Pagan test and HC3 robust standard errors.
8. **Case Study Walkthrough & Summary:** Property valuation results and key takeaways.

---

## 🛠️ Compilation Command

```bash
cd lessons/L01_Regression_Analysis/presentation
typst compile regression_analysis_slides.typ
```
