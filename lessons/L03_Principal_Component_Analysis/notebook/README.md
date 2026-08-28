# Chapter 3 Notebook — Financial Market PCA Case Study (`lessons/L03_Principal_Component_Analysis/notebook/`)

This directory contains the interactive Jupyter case study for **Chapter 3: Principal Component Analysis (PCA)**.

---

## 📂 Directory Contents

| Notebook | Purpose | Status |
| :--- | :--- | :---: |
| [`financial_pca_analysis.ipynb`](financial_pca_analysis.ipynb) | End-to-end PCA case study covering data standardization, spectral decomposition of the correlation matrix, Kaiser criterion, Scree plots, Horn's Parallel Analysis, component loadings interpretation, 2D/3D biplots, and factor score dynamics. | ✅ Verified |

---

## 🔬 Notebook Cell Progression & Methodology

1. **Exploratory Data Analysis (EDA):** Asset returns distributions, volatility rankings, and correlation matrix heatmap ($\mathbf{R}$).
2. **Fitting Principal Component Analysis:** Standardizing returns ($z$-scores) and decomposing the sample correlation matrix via `sklearn.decomposition.PCA`.
3. **Component Retention Analysis:**
   - Evaluating individual and cumulative variance explained.
   - Kaiser-Guttman rule ($\lambda_j > 1.0$).
   - Constructing Scree Plot with elbow identification.
   - Performing **Horn's Parallel Analysis** (simulating 200 Gaussian noise datasets to establish empirical 95th percentile noise thresholds).
4. **Loadings Interpretation Heatmap:** Extracting economic interpretations:
   - **PC1:** Global Market Growth / Equity Beta.
   - **PC2:** Interest Rate / Duration Sensitivity.
   - **PC3:** Energy & Commodity Inflation Factor.
5. **The Gabriel 2D Biplot:** Projecting trading session observation scores and directed asset loading vectors simultaneously.
6. **Time Series Factor Dynamics:** Plotting temporal scores across trading sessions to track market risk regimes and macroeconomic shocks.

---

## 🚀 Execution Instructions

```bash
# Launch via Jupyter Notebook interface
uv run jupyter notebook lessons/L03_Principal_Component_Analysis/notebook/financial_pca_analysis.ipynb

# Or run non-interactively via the test harness
uv run python scripts/run_notebooks.py lessons/L03_Principal_Component_Analysis/notebook/financial_pca_analysis.ipynb
```
