# Chapter 5 Presentation — Discriminant Analysis Slides (`lessons/L05_Discriminant_Analysis/presentation/`)

This directory contains the classroom slide presentation for **Chapter 5: Discriminant Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`discriminant_analysis_slides.typ`](discriminant_analysis_slides.typ) | Typst Source | Slide deck built with [Touying](https://github.com/touying-typ/touying) using the `university-theme` and Tecnológico de Monterrey branding (`#003366` / `#1E88E5`). |
| [`discriminant_analysis_slides.pdf`](discriminant_analysis_slides.pdf) | PDF (Git LFS) | Fully compiled, 16:9 widescreen presentation PDF. |

---

## 📽️ Slide Deck Outline

1. **Supervised Classification Foundations:** Group separation vs. regression.
2. **Two-Group Fisher's LDA:** Maximizing between-to-within variance ratio.
3. **Multi-Group Canonical Discriminant Functions:** Eigendecomposition of $\mathbf{W}^{-1}\mathbf{B}$.
4. **Bayesian Decision Theory:** Incorporating prior probabilities $p_k$ and misclassification costs $c(i|j)$.
5. **Linear vs. Quadratic Boundaries:** When to switch from LDA to QDA.
6. **Wilks' $\Lambda$ & Significance Testing:** Assessing canonical dimension strength.
7. **Customer Segmentation Case Study:** Projecting 3 consumer tiers onto canonical space.
8. **Summary & Decision Workflow:** Practical guidelines for classifier selection.

---

## 🛠️ Compilation Command

```bash
cd lessons/L05_Discriminant_Analysis/presentation
typst compile discriminant_analysis_slides.typ
```
