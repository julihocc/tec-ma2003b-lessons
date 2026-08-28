# Chapter 7 Notebook — Health Risk Multivariate Regression Case Study (`lessons/L07_Multivariate_Regression/notebook/`)

This directory contains the interactive Jupyter case study for **Chapter 7: Multivariate Regression**.

---

## 📂 Directory Contents

| Notebook | Purpose | Status |
| :--- | :--- | :---: |
| [`health_risk_analysis.ipynb`](health_risk_analysis.ipynb) | End-to-end clinical case study covering Multivariate Linear Regression ($\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}$), Hotelling's $T^2$ two-sample test, Multivariate Analysis of Variance (MANOVA with Wilks' $\Lambda$, Pillai's trace, Hotelling-Lawley, Roy's largest root), Logistic Regression, and Canonical Correlation Analysis (CCA). | ✅ Verified |

---

## 🔬 Notebook Cell Progression & Methodology

1. **Exploratory Data Analysis (EDA):** Patient demographics, physiological correlations, and joint response scatter plots (`systolic_bp` vs. `fasting_glucose`).
2. **Two-Sample Hotelling's $T^2$ Test:** Testing difference in joint biomarker mean vectors between CVD risk groups:
   $ T^2 = \frac{n_1 n_2}{n_1 + n_2} (\bar{\mathbf{y}}_1 - \bar{\mathbf{y}}_2)^T \mathbf{S}_{\text{pooled}}^{-1} (\bar{\mathbf{y}}_1 - \bar{\mathbf{y}}_2) $
3. **Multivariate Analysis of Variance (MANOVA):**
   - Evaluating multi-group differences across lifestyle categories.
   - Reporting Wilks' $\Lambda$, Pillai's Trace, Hotelling-Lawley Trace, and Roy's Largest Root with corresponding $F$-approximations.
4. **Multivariate Linear Regression Model:**
   - Estimating coefficient matrix $\hat{\mathbf{B}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$.
   - Decomposing residual error covariance matrix $\hat{\mathbf{\Sigma}}_{\mathbf{E}} = \frac{1}{n - k - 1} \hat{\mathbf{E}}^T \hat{\mathbf{E}}$.
   - Evaluating cross-equation error correlation.
5. **Multivariate Logistic Regression:** Modeling binary cardiac risk diagnosis (`cvd_risk_flag`) with odds ratios and ROC-AUC curve.
6. **Canonical Correlation Analysis (CCA):**
   - Identifying maximal linear associations between the Lifestyle Set ($\mathbf{X}_1$) and Physiological Biomarker Set ($\mathbf{X}_2$).
   - Computing canonical correlation coefficients ($\rho_1^*, \rho_2^*$) and canonical variate loadings.

---

## 🚀 Execution Instructions

```bash
# Launch via Jupyter Notebook interface
uv run jupyter notebook lessons/L07_Multivariate_Regression/notebook/health_risk_analysis.ipynb

# Or run non-interactively via the test harness
uv run python scripts/run_notebooks.py lessons/L07_Multivariate_Regression/notebook/health_risk_analysis.ipynb
```
