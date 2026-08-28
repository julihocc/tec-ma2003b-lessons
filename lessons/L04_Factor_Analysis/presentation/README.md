# Chapter 4 Presentation — Factor Analysis Slides (`lessons/L04_Factor_Analysis/presentation/`)

This directory contains classroom slide presentations for **Chapter 4: Factor Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`factor_analysis_slides.typ`](factor_analysis_slides.typ) | Typst Source | Primary slide deck built with [Touying](https://github.com/touying-typ/touying) using the `university-theme` and Tecnológico de Monterrey branding (`#003366` / `#1E88E5`). |
| [`factor_analysis_slides.pdf`](factor_analysis_slides.pdf) | PDF (Git LFS) | Fully compiled 16:9 Touying slide deck PDF. |
| [`fa_presentation.typ`](fa_presentation.typ) | Typst Source | Supplementary classroom presentation source. |
| [`fa_presentation.pdf`](fa_presentation.pdf) | PDF (Git LFS) | Compiled supplementary presentation PDF. |

---

## 📽️ Primary Slide Deck Outline

1. **Foundations of Latent Variable Modeling:** When and why to use Factor Analysis vs. PCA.
2. **The Common Factor Model:** Matrix formulation $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{F} + \boldsymbol{\epsilon}$.
3. **Variance Decomposition:** Communalities ($h^2$) and Uniqueness ($\psi$).
4. **Extraction Methods:** Principal Axis Factoring vs. Maximum Likelihood.
5. **Factor Rotation:** Orthogonal Varimax vs. Oblique Promax for Thurstone's simple structure.
6. **Factor Scores:** Computing Bartlett and Regression factor scores.
7. **Educational Assessment Case Study:** Step-by-step interpretation of 3 student ability constructs.

---

## 🛠️ Compilation Command

```bash
cd lessons/L04_Factor_Analysis/presentation
typst compile factor_analysis_slides.typ
```
