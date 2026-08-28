# Chapter 4 Lecture Notes & Reports (`lessons/L04_Factor_Analysis/notes/`)

This directory contains the theoretical lecture notes, study guides, and executive case reports for **Chapter 4: Factor Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`factor_analysis_notes.typ`](factor_analysis_notes.typ) | Typst Source | Comprehensive study guide covering all syllabus subtopics (4.1–4.6) with the Common Factor Model $\mathbf{X} = \mathbf{\mu} + \mathbf{L}\mathbf{F} + \boldsymbol{\epsilon}$, covariance decomposition $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T + \mathbf{\Psi}$, extraction algorithms, rotation geometry, and factor score estimation. |
| [`factor_analysis_notes.pdf`](factor_analysis_notes.pdf) | PDF (Git LFS) | Fully compiled study guide PDF. |
| [`executive_report.typ`](executive_report.typ) | Typst Source | Executive-level analytical report summarizing findings from the educational assessment study for institutional stakeholders. |
| [`executive_report.pdf`](executive_report.pdf) | PDF (Git LFS) | Fully compiled executive report PDF. |

---

## 📖 Theoretical Topics Covered

1. **Objectives of Factor Analysis (4.1):** Latent construct discovery, separating common variance from unique variance, contrasting FA with PCA.
2. **Factor Analysis Equations (4.2):** Mathematical model $\mathbf{X} - \boldsymbol{\mu} = \mathbf{L}\mathbf{F} + \boldsymbol{\epsilon}$, fundamental covariance identity $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T + \mathbf{\Psi}$, communality $h_i^2$, uniqueness $\psi_i$.
3. **Choice of Number of Factors (4.3):** Kaiser rule, Scree plot inflection, residual correlation matrix test ($|\mathbf{S} - \hat{\mathbf{\Sigma}}| \approx \mathbf{0}$).
4. **Factor Rotation (4.4):** Indeterminacy of orthogonal factor solutions, Kaiser's Varimax criterion for achieving Thurstone's simple structure.
5. **Oblique Rotation (4.5):** Promax and Quartimin rotations, reference axes, pattern matrix vs. structure matrix.
6. **Factor Scores & Coding (4.6):** Weighted least squares (Bartlett) and regression (Thomson) factor score estimation, `factor_analyzer` Python library workflow.

---

## 🛠️ Compilation Commands

```bash
cd lessons/L04_Factor_Analysis/notes

# Compile lecture notes
typst compile factor_analysis_notes.typ

# Compile executive report
typst compile executive_report.typ
```
