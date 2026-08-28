# Chapter 1 Notebook — Housing Regression Case Study (`lessons/L01_Regression_Analysis/notebook/`)

This directory contains the hands-on interactive Jupyter case study for **Chapter 1: Regression Analysis**.

---

## 📂 Directory Contents

| Notebook | Purpose | Status |
| :--- | :--- | :---: |
| [`housing_regression_analysis.ipynb`](housing_regression_analysis.ipynb) | End-to-end regression modeling case study evaluating single/multiple linear regression, ANOVA tables, prediction bands, polynomial terms, VIF, Cook's distance, and HC3 robust standard errors. | ✅ Verified |

---

## 🔬 Notebook Cell Progression & Methodology

1. **Exploratory Data Analysis (EDA):** Summary statistics, distribution shapes, and correlation heatmap across the 7 housing predictors and `sale_price`.
2. **Simple Linear Regression:** Fitting `sale_price ~ sqft_living`, estimating OLS parameters ($\hat{\beta}_0, \hat{\beta}_1$), computing $R^2$, and plotting 95% confidence bands for mean response vs. 95% prediction intervals for individual homes.
3. **Multiple Linear Regression:** Adding structural and geographic predictors (`bedrooms`, `bathrooms`, `dist_city_center_km`, `building_grade`, `house_age_years`).
4. **Polynomial & Non-Linear Transformations:** Fitting quadratic age effect ($\text{Age}^2$) capturing depreciation of middle-aged homes and historical renovation premiums.
5. **Multicollinearity Diagnostics:** Calculating Variance Inflation Factors (VIF) to confirm no severe collinearity ($\text{VIF} < 5$).
6. **Residual Diagnostics (4-Panel Plot):**
   - Residuals vs. Fitted values (evaluating linearity and variance stability).
   - Normal Q-Q plot of studentized residuals (evaluating normality).
   - Cook's Distance plot (evaluating point influence against $4/n$ cutoff).
   - Leverage vs. Studentized Residuals plot.
7. **Heteroskedasticity Testing & Robust SE:**
   - Breusch-Pagan Lagrange Multiplier test.
   - Refitting with Heteroskedasticity-Consistent standard errors (HC3) to ensure valid inference.
8. **Business Conclusions:** Translating statistical coefficients into real estate appraisal insights.

---

## 🚀 Execution Instructions

```bash
# Launch via Jupyter Notebook interface
uv run jupyter notebook lessons/L01_Regression_Analysis/notebook/housing_regression_analysis.ipynb

# Or run non-interactively via the test harness
uv run python scripts/run_notebooks.py lessons/L01_Regression_Analysis/notebook/housing_regression_analysis.ipynb
```
