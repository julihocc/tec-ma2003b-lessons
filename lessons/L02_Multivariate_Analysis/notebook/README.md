# Chapter 2 Notebook — Environmental Multivariate Analysis Case Study (`lessons/L02_Multivariate_Analysis/notebook/`)

This directory contains the interactive Jupyter case study for **Chapter 2: Multivariate Analysis**.

---

## 📂 Directory Contents

| Notebook | Purpose | Status |
| :--- | :--- | :---: |
| [`environmental_multivariate_analysis.ipynb`](environmental_multivariate_analysis.ipynb) | End-to-end multivariate data analysis case study covering missing data imputation (KNN), mean vectors & covariance matrices, Chi-square Q-Q normality test, Robust MCD outlier detection, Fisher $z$ confidence intervals, and Andrews curves. | ✅ Verified |

---

## 🔬 Notebook Cell Progression & Methodology

1. **Data Loading & Missing Pattern Inspection:** Inspecting null sensor readings across the 8 atmospheric metrics using a missingness heatmap.
2. **Missing Data Imputation:** Applying K-Nearest Neighbors ($K=5$) imputation to restore complete multivariate records while preserving cross-feature covariance.
3. **Multivariate Summary Statistics:** Computing sample mean vector ($\bar{\mathbf{x}}$), sample covariance matrix ($\mathbf{S}$), correlation matrix ($\mathbf{R}$), generalized variance ($|\mathbf{S}|$), and total variance ($\operatorname{tr}(\mathbf{S})$).
4. **Multivariate Normality Assessment:** Constructing a Chi-Square Q-Q plot comparing sample Mahalanobis squared distances $D^2$ to theoretical quantiles ($\chi_8^2$).
5. **Multivariate Outlier Detection:** Comparing classical Mahalanobis distance against the Robust Minimum Covariance Determinant (MCD) estimator to identify masked sensor anomalies.
6. **Fisher $z$-Transformation:** Calculating 95% confidence intervals for primary pollutant correlation pairs (e.g., PM2.5 vs. PM10, O3 vs. Temperature).
7. **Multidimensional Visualizations:** Mapping 8D air quality profiles onto continuous function space using Andrews Curves grouped by pollution severity tiers.

---

## 🚀 Execution Instructions

```bash
# Launch via Jupyter Notebook interface
uv run jupyter notebook lessons/L02_Multivariate_Analysis/notebook/environmental_multivariate_analysis.ipynb

# Or run non-interactively via the test harness
uv run python scripts/run_notebooks.py lessons/L02_Multivariate_Analysis/notebook/environmental_multivariate_analysis.ipynb
```
