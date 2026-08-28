# Environmental Monitoring & Multivariate Analysis Data Dictionary

**Dataset Name:** `environmental_data.csv`  
**Records:** 800 monitoring stations  
**Features:** 9 columns (1 Identifier, 8 Continuous Atmospheric Metrics)  
**Topic:** Chapter 2 — Multivariate Analysis (MA2003B)

---

## 📊 Variable Descriptions

| Variable Name | Type | Unit | Range / Values | Description | Role in Multivariate Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `station_id` | Integer | ID | `101` - `900` | Unique sensor station identifier | Identifier |
| `pm25` | Float | $\mu\text{g/m}^3$ | `2.0` - `75.0` | Fine particulate matter ($\le 2.5\mu\text{m}$) | Primary pollutant ($X_1$) |
| `pm10` | Float | $\mu\text{g/m}^3$ | `5.0` - `140.0` | Coarse inhalable particulate matter ($\le 10\mu\text{m}$) | High collinearity with PM2.5 ($X_2$) |
| `no2` | Float | ppb | `1.0` - `65.0` | Nitrogen dioxide concentration | Traffic / combustion marker ($X_3$) |
| `so2` | Float | ppb | `0.5` - `30.0` | Sulfur dioxide concentration | Industrial emissions marker ($X_4$) |
| `co` | Float | ppm | `0.10` - `3.20` | Carbon monoxide concentration | Incomplete combustion marker ($X_5$) |
| `o3` | Float | $\mu\text{g/m}^3$ | `1.0` - `90.0` | Ground-level tropospheric ozone | Photochemical oxidant ($X_6$) |
| `temperature` | Float | $^\circ\text{C}$ | `2.0` - `42.0` | Ambient surface air temperature | Meteorological covariate ($X_7$) |
| `humidity` | Float | $\%$ | `10.0` - `100.0` | Relative ambient atmospheric humidity | Meteorological covariate ($X_8$) |

---

## 🔬 Pedagogical Highlights for Chapter 2

1. **Multivariate Distribution Properties:** Joint mean vector ($\bar{\mathbf{x}}$) and covariance matrix ($\mathbf{S}$) capturing cross-pollutant interactions.
2. **Multivariate Normality Assessment:** Chi-square Q-Q plot comparing Mahalanobis squared distances $D^2$ to $\chi_8^2$ theoretical distribution.
3. **Missing Value Diagnostics & Imputation:** Contains ~3% simulated missing values (MCAR/MAR) to practice mean, KNN, and Iterative (MICE) imputation methods.
4. **Multivariate Outlier Detection:** Contains subtle joint outliers identifiable through Mahalanobis distance thresholds ($\chi_{8, 0.999}^2 = 26.12$) that remain undetected by standard 1D univariate limits.
5. **Fisher & Ruben Correlation Intervals:** Evaluation of confidence intervals for sample correlation coefficients (e.g. $\rho(\text{PM2.5}, \text{PM10})$ and $\rho(\text{O}_3, \text{Temp})$).
6. **High-Dimensional Visualizations:** Pairwise scatter plot matrices, correlation heatmaps, and Andrews curves for multivariate pattern recognition.
