# Chapter 5 Notebook — Marketing Discriminant Analysis Case Study (`lessons/L05_Discriminant_Analysis/notebook/`)

This directory contains the interactive Jupyter case study for **Chapter 5: Discriminant Analysis**.

---

## 📂 Directory Contents

| Notebook | Purpose | Status |
| :--- | :--- | :---: |
| [`marketing_discriminant_analysis.ipynb`](marketing_discriminant_analysis.ipynb) | End-to-end supervised classification case study covering Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), Box's M test for equality of covariance matrices, canonical discriminant functions, Wilks' $\Lambda$, decision boundary visualization, and confusion matrix evaluation. | ✅ Verified |

---

## 🔬 Notebook Cell Progression & Methodology

1. **Exploratory Data Analysis by Group:** Visualizing 3 consumer segments (`Budget`, `Mainstream`, `Premium`) across spending and income distributions.
2. **Assumption Testing:**
   - Group-wise multivariate normality.
   - **Box's M Test** for equality of group covariance matrices ($\mathbf{\Sigma}_1 = \mathbf{\Sigma}_2 = \mathbf{\Sigma}_3$).
3. **Linear Discriminant Analysis (LDA):**
   - Estimating pooled covariance matrix $\mathbf{S}_{\text{pooled}}$ and between-group scatter $\mathbf{B}$.
   - Extracting $K-1 = 2$ canonical discriminant functions via eigendecomposition of $\mathbf{W}^{-1}\mathbf{B}$.
   - Evaluating Wilks' Lambda ($\Lambda$) significance and canonical correlations.
4. **Quadratic Discriminant Analysis (QDA):**
   - Relaxing equal covariance assumptions to model group-specific curvatures.
5. **Decision Boundaries & Canonical Projections:**
   - Projecting 7D feature space onto 2D canonical variate space ($LD_1$ vs. $LD_2$).
   - Plotting classification decision boundaries and group centroids.
6. **Model Evaluation:** Confusion matrices, cross-validation classification accuracy, precision, recall, and Expected Cost of Misclassification (ECM).

---

## 🚀 Execution Instructions

```bash
# Launch via Jupyter Notebook interface
uv run jupyter notebook lessons/L05_Discriminant_Analysis/notebook/marketing_discriminant_analysis.ipynb

# Or run non-interactively via the test harness
uv run python scripts/run_notebooks.py lessons/L05_Discriminant_Analysis/notebook/marketing_discriminant_analysis.ipynb
```
