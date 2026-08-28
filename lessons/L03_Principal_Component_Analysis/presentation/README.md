# Chapter 3 Presentation — PCA Slides (`lessons/L03_Principal_Component_Analysis/presentation/`)

This directory contains the classroom slide presentation for **Chapter 3: Principal Component Analysis (PCA)**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`principal_component_analysis_slides.typ`](principal_component_analysis_slides.typ) | Typst Source | Slide deck built with [Touying](https://github.com/touying-typ/touying) using the `university-theme` and Tecnológico de Monterrey branding (`#003366` / `#1E88E5`). |
| [`principal_component_analysis_slides.pdf`](principal_component_analysis_slides.pdf) | PDF (Git LFS) | Fully compiled, 16:9 widescreen presentation PDF. |

---

## 📽️ Slide Deck Outline

1. **Motivations for PCA:** Data compression, multicollinearity, and exploratory visualization.
2. **Geometric Foundations:** Orthogonal coordinate rotation along principal axes of inertia.
3. **Mathematical Derivation:** Lagrange multiplier derivation of $\mathbf{\Sigma}\mathbf{a} = \lambda\mathbf{a}$.
4. **Spectral Decomposition:** Preserving total variance $\operatorname{tr}(\mathbf{S}) = \sum \lambda_j$.
5. **Component Retention Criteria:** Kaiser rule, Scree plot, cumulative variance, and Parallel Analysis.
6. **Loadings, Scores & Biplots:** Gabriel biplot geometry and variable vectors.
7. **Financial Case Study:** Macroeconomic factor decomposition across 10 asset classes.
8. **Summary & Bridge to Factor Analysis:** Contrasting total variance reduction with latent construct modeling.

---

## 🛠️ Compilation Command

```bash
cd lessons/L03_Principal_Component_Analysis/presentation
typst compile principal_component_analysis_slides.typ
```
