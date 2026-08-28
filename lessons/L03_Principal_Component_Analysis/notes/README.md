# Chapter 3 Lecture Notes (`lessons/L03_Principal_Component_Analysis/notes/`)

This directory contains the theoretical lecture notes and study guides for **Chapter 3: Principal Component Analysis (PCA)**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`principal_component_analysis_notes.typ`](principal_component_analysis_notes.typ) | Typst Source | Comprehensive study guide covering all syllabus subtopics (3.1–3.5) with algebraic derivations via Lagrange multipliers, spectral decomposition of $\mathbf{S}$ and $\mathbf{R}$, retention criteria comparisons, biplot theory, and Python code blocks. |
| [`principal_component_analysis_notes.pdf`](principal_component_analysis_notes.pdf) | PDF (Git LFS) | Fully compiled, publication-ready PDF document typeset with Typst. |

---

## 📖 Theoretical Topics Covered

1. **Cases Where PCA is Used (3.1):** High-dimensional data compression, multicollinearity remediation in regression, exploratory visualization, macroeconomic factor extraction. Contrasting PCA with Factor Analysis.
2. **Geometrical Description of Components (3.2):** Orthogonal rotation of coordinates in $\mathbb{R}^p$, finding principal axes of inertia that maximize variance of projected data.
3. **Estimation & Spectral Decomposition (3.3):**
   - Mathematical derivation using Lagrange multipliers: $\max \mathbf{a}_1^T\mathbf{\Sigma}\mathbf{a}_1$ subject to $\mathbf{a}_1^T\mathbf{a}_1 = 1 \implies \mathbf{\Sigma}\mathbf{a}_1 = \lambda_1\mathbf{a}_1$.
   - Spectral theorem: $\mathbf{S} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T = \sum_{j=1}^p \lambda_j \mathbf{v}_j\mathbf{v}_j^T$.
   - Total variance preservation: $\operatorname{tr}(\mathbf{S}) = \sum \lambda_j$.
   - Component loadings: $L_{jk} = a_{jk}\sqrt{\lambda_j}$.
   - Individual component scores: $Y_{ij} = \mathbf{a}_j^T(\mathbf{x}_i - \bar{\mathbf{x}})$.
4. **Component Retention Rules (3.4):** Kaiser-Guttman criterion ($\lambda > 1.0$), Cattell's Scree plot elbow rule, cumulative variance threshold ($70\%-85\%$), Horn's Parallel Analysis.
5. **Biplots & Python Coding (3.5):** Gabriel (1971) 2D/3D biplots, interpreting vector angles as approximate correlations, `sklearn.decomposition.PCA` workflow.

---

## 🛠️ Compilation Command

```bash
cd lessons/L03_Principal_Component_Analysis/notes
typst compile principal_component_analysis_notes.typ
```
