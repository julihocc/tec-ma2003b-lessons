# Chapter 6 Notebook — Customer Cluster Analysis Case Study (`lessons/L06_Cluster_Analysis/notebook/`)

This directory contains the interactive Jupyter case study for **Chapter 6: Analysis by Conglomerates (Cluster Analysis)**.

---

## 📂 Directory Contents

| Notebook | Purpose | Status |
| :--- | :--- | :---: |
| [`customer_clustering_analysis.ipynb`](customer_clustering_analysis.ipynb) | Comprehensive unsupervised segmentation case study comparing Hierarchical Clustering (Ward's, Complete, Single), K-Means, Elbow method, Silhouette analysis, Dendrogram interpretation, and 2D PCA cluster projections. | ✅ Verified |

---

## 🔬 Notebook Cell Progression & Methodology

1. **Exploratory Data Analysis (EDA):** Feature distributions, pairwise scatterplots, and scaling sensitivity analysis.
2. **Data Standardization:** Applying `StandardScaler` to prevent high-magnitude features (`annual_income`) from distorting Euclidean distances.
3. **Hierarchical Agglomerative Clustering:**
   - Computing distance matrices and evaluating linkage methods: Single, Complete, Average, and Ward's Minimum Variance.
   - Plotting hierarchical **Dendrograms** with horizontal truncation threshold lines.
   - Extracting cluster assignments at optimal cophenetic distance cutoffs.
4. **Non-Hierarchical Clustering (K-Means):**
   - Running K-Means across cluster ranges $k \in [2, 10]$.
   - Calculating Within-Cluster Sum of Squares (Inertia) for the **Elbow Curve**.
   - Calculating and plotting **Silhouette Scores** and silhouette width profiles.
5. **Dimensionality Reduction & Cluster Visualization:**
   - Fitting 2D Principal Component Analysis (PCA) on normalized features.
   - Plotting convex hulls and centroid locations for the 4 identified customer personas.
6. **Cluster Persona Profiling:** Radar charts and summary statistics describing behavioral personas for targeted marketing campaigns.

---

## 🚀 Execution Instructions

```bash
# Launch via Jupyter Notebook interface
uv run jupyter notebook lessons/L06_Cluster_Analysis/notebook/customer_clustering_analysis.ipynb

# Or run non-interactively via the test harness
uv run python scripts/run_notebooks.py lessons/L06_Cluster_Analysis/notebook/customer_clustering_analysis.ipynb
```
