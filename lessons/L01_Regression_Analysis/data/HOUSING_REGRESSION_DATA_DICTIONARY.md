# Housing Valuation & Regression Analysis Data Dictionary

**Dataset Name:** `housing_regression.csv`  
**Records:** 1,000 observations  
**Features:** 9 columns (1 Identifier, 7 Predictors, 1 Continuous Target)  
**Topic:** Chapter 1 — Regression Analysis (MA2003B)

---

## 📊 Variable Descriptions

| Variable Name | Type | Unit | Range / Values | Description | Role in Regression |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `property_id` | Integer | ID | `1001` - `2000` | Unique property observation identifier | Identifier (exclude from models) |
| `sqft_living` | Float | Sq. Feet | `600.0` - `4800.0` | Interior living area square footage | Primary linear predictor ($X_1$) |
| `bedrooms` | Integer | Count | `1` - `6` | Total number of bedrooms | Discrete predictor ($X_2$) |
| `bathrooms` | Float | Count | `1.0` - `4.5` | Number of full & half bathrooms (0.5 increments) | Continuous/discrete predictor ($X_3$) |
| `house_age_years` | Float | Years | `0.0` - `65.0` | Age of residential structure since construction | Non-linear quadratic term ($X_4, X_4^2$) |
| `dist_city_center_km` | Float | Kilometers | `1.00` - `40.00` | Straight-line distance to primary downtown urban hub | Negative slope predictor ($X_5$) |
| `building_grade` | Integer | Rating | `4` - `10` | Structural build quality & finishes rating scale | Ordinal/categorical predictor ($X_6$) |
| `energy_efficiency_rating` | Float | kWh/m²/yr | `45.0` - `260.0` | Estimated annual residential energy demand index | Multi-collinear predictor ($X_7$) |
| `sale_price` | Float | USD (\$) | `\$75,000` - `\$950,000` | Final verified residential transaction sale price | **Target Response Variable ($Y$)** |

---

## 🔬 Pedagogical Features for Chapter 1

1. **Simple vs. Multiple Linear Regression:** Strong linear association between `sqft_living` and `sale_price` ($R^2 \approx 0.65$), expanding to multiple predictors ($R^2 > 0.85$).
2. **ANOVA & Model Comparison:** Nested $F$-tests comparing simple model vs. full feature matrix.
3. **Non-Linear Relationships:** Quadratic effect on `house_age_years` capturing depreciation of mid-aged homes and premium for historic renovated homes.
4. **Heteroskedasticity:** Error variance expands proportionally with larger `sqft_living` values (verifiable via Breusch-Pagan test and addressable with WLS and HC3 standard errors).
5. **Outlier Diagnostics:** Includes simulated high-leverage and high-residual observations for Cook's distance and studentized residual analysis.
