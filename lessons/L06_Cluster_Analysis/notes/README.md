# Chapter 6 Lecture Notes (`lessons/L06_Cluster_Analysis/notes/`)

This directory contains the theoretical lecture notes and study guides for **Chapter 6: Analysis by Conglomerates (Cluster Analysis)**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`cluster_analysis_notes.typ`](cluster_analysis_notes.typ) | Typst Source | Comprehensive study guide covering all syllabus subtopics (6.1–6.6) with distance metric metrics, agglomerative clustering algorithms, Ward's method, K-Means convergence theory, silhouette widths, and Python code blocks. |
| [`cluster_analysis_notes.pdf`](cluster_analysis_notes.pdf) | PDF (Git LFS) | Fully compiled, publication-ready PDF document typeset with Typst. |

---

## 📖 Theoretical Topics Covered

1. **Similarity & Dissimilarity Measures (6.1):** Euclidean ($L_2$), Manhattan ($L_1$), Minkowski ($L_p$), Mahalanobis, Canberra, and cosine distances for quantitative variables; Jaccard and simple matching coefficients for binary attributes.
2. **Graphical Methods (6.2):** Scatter plot matrices, principal component projections, Andrews trigonometric curves, and dendrogram branch geometry.
3. **Non-Hierarchical Grouping (6.3):** K-Means clustering algorithm (Lloyd-Forgy), objective function minimizing within-cluster inertia $W(C) = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$, K-Means++ centroid initialization.
4. **Hierarchical Grouping (6.4):** Agglomerative nesting (AGNES), Lance-Williams recurrence formula for updating dissimilarity matrices, Cophenetic correlation coefficient.
5. **Nearest Neighbor & Linkage Methods (6.5):** Single linkage (chaining effect), Complete linkage (compact spheres), Average linkage (UPGMA), Ward's minimum variance criterion ($\Delta \text{ESS}$).
6. **Cluster Validation & Coding (6.6):** Silhouette score $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, Calinski-Harabasz index, Davies-Bouldin index, `scipy.cluster.hierarchy` and `sklearn.cluster` implementation.

---

## 🛠️ Compilation Command

```bash
cd lessons/L06_Cluster_Analysis/notes
typst compile cluster_analysis_notes.typ
```
