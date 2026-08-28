# Chapter 5 Lecture Notes (`lessons/L05_Discriminant_Analysis/notes/`)

This directory contains the theoretical lecture notes and study guides for **Chapter 5: Discriminant Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`discriminant_analysis_notes.typ`](discriminant_analysis_notes.typ) | Typst Source | Comprehensive study guide covering all syllabus subtopics (5.1–5.6) with two-group Fisher LDA derivation, multi-group canonical discriminant functions, optimal Bayes classification rules, cost functions, Wilks' $\Lambda$, and Python code blocks. |
| [`discriminant_analysis_notes.pdf`](discriminant_analysis_notes.pdf) | PDF (Git LFS) | Fully compiled, publication-ready PDF document typeset with Typst. |

---

## 📖 Theoretical Topics Covered

1. **Two-Group Discrimination (5.1):** Fisher's linear discriminant function maximizing between-group to within-group variance ratio $\frac{\mathbf{a}^T\mathbf{B}\mathbf{a}}{\mathbf{a}^T\mathbf{W}\mathbf{a}}$, solution $\mathbf{a} \propto \mathbf{S}_{\text{pooled}}^{-1}(\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)$.
2. **Cost Functions & Prior Probabilities (5.2):** Bayes classification rule minimizing Expected Cost of Misclassification (ECM): assign to $\Pi_1$ if $\frac{f_1(\mathbf{x})}{f_2(\mathbf{x})} > \frac{c(1|2)p_2}{c(2|1)p_1}$.
3. **Basic Discrimination & QDA (5.3):** Quadratic Discriminant Analysis (QDA) when covariance matrices are unequal ($\mathbf{\Sigma}_1 \neq \mathbf{\Sigma}_2$), Box's M test.
4. **Stepwise Feature Selection (5.4):** Forward and backward variable selection using partial $F$-to-enter and Wilks' $\Lambda$ minimization.
5. **Canonical Discriminant Functions (5.5):** Generalization to $K \ge 3$ groups, spectral decomposition of $\mathbf{W}^{-1}\mathbf{B}$, eigenvalue-eigenvector solutions, Wilks' Lambda test statistic $\Lambda = \prod_{i=1}^s \frac{1}{1 + \lambda_i}$.
6. **Coding & Software (5.6):** Implementation using `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` and `QuadraticDiscriminantAnalysis`.

---

## 🛠️ Compilation Command

```bash
cd lessons/L05_Discriminant_Analysis/notes
typst compile discriminant_analysis_notes.typ
```
