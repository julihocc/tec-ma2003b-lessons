# Chapter 6 Presentation — Cluster Analysis Slides (`lessons/L06_Cluster_Analysis/presentation/`)

This directory contains classroom slide presentations for **Chapter 6: Analysis by Conglomerates (Cluster Analysis)**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`cluster_analysis_slides.typ`](cluster_analysis_slides.typ) | Typst Source | Slide deck built with [Touying](https://github.com/touying-typ/touying) using the `university-theme` and Tecnológico de Monterrey branding (`#003366` / `#1E88E5`). |
| [`cluster_analysis_slides.pdf`](cluster_analysis_slides.pdf) | PDF (Git LFS) | Fully compiled, 16:9 widescreen presentation PDF. |

---

## 📽️ Slide Deck Outline

1. **Foundations of Unsupervised Learning:** Clustering vs. Classification.
2. **Proximity & Distance Metrics:** Euclidean, Manhattan, Mahalanobis, and Gower distances.
3. **Hierarchical Agglomerative Clustering:** Linkage strategies (Single, Complete, Average, Ward's).
4. **Dendrogram Interpretation:** Branch heights, cophenetic correlation, and cutting trees.
5. **Partitioning Methods (K-Means):** Lloyd's algorithm, inertia minimization, and K-Means++.
6. **Determining Optimal Cluster Count ($k$):** Elbow method and Silhouette width analysis.
7. **Customer Behavioral Case Study:** 2,000 shopper profiles segmented into 4 actionable personas.
8. **Summary & Guidelines:** Selecting the appropriate clustering algorithm for specific data geometries.

---

## 🛠️ Compilation Command

```bash
cd lessons/L06_Cluster_Analysis/presentation
typst compile cluster_analysis_slides.typ
```
