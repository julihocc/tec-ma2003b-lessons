# Chapter 2 Presentation — Multivariate Analysis Slides (`lessons/L02_Multivariate_Analysis/presentation/`)

This directory contains the classroom slide presentation for **Chapter 2: Multivariate Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`multivariate_analysis_slides.typ`](multivariate_analysis_slides.typ) | Typst Source | Slide deck built with [Touying](https://github.com/touying-typ/touying) using the `university-theme` and Tecnológico de Monterrey branding (`#003366` / `#1E88E5`). |
| [`multivariate_analysis_slides.pdf`](multivariate_analysis_slides.pdf) | PDF (Git LFS) | Fully compiled, 16:9 widescreen presentation PDF. |

---

## 📽️ Slide Deck Outline

1. **Random Vectors & Mean/Covariance Parameters:** Population definitions and positive semi-definiteness.
2. **Sample Statistics:** Sample mean vector $\bar{\mathbf{x}}$, covariance $\mathbf{S}$, and correlation matrix $\mathbf{R}$.
3. **The Multivariate Normal Distribution (MVN):** Joint density formula and core properties.
4. **Geometry of Covariance:** Contour ellipsoids, Mahalanobis distance, and Chi-Square Q-Q plots.
5. **Missing Values & Imputation:** MCAR/MAR diagnostics and KNN imputation algorithm.
6. **Multivariate Outlier Diagnostics:** Classical Mahalanobis vs. Robust Minimum Covariance Determinant (MCD).
7. **Fisher $z$-Transformation:** Normalizing sample correlations for confidence intervals.
8. **Case Study & Takeaways:** Environmental air quality monitoring insights.

---

## 🛠️ Compilation Command

```bash
cd lessons/L02_Multivariate_Analysis/presentation
typst compile multivariate_analysis_slides.typ
```
