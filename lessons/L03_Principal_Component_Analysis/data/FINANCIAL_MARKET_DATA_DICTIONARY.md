# Global Financial Market & PCA Analysis Data Dictionary

**Dataset Name:** `financial_market_data.csv`  
**Records:** 600 daily trading periods  
**Features:** 11 columns (1 Identifier, 10 Asset Class Daily Percentage Returns)  
**Topic:** Chapter 3 — Principal Component Analysis (MA2003B)

---

## 📊 Variable Descriptions

| Variable Name | Type | Unit | Typical Range | Description | Expected Factor Sensitivity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `trading_day` | Integer | Day | `1` - `600` | Chronological trading session index | Sequence ID (exclude from PCA) |
| `us_equities` | Float | $\%$ | `-3.5` to `+3.5` | S&P 500 broad US equity benchmark daily return | High Market Beta ($F_1$) |
| `global_equities` | Float | $\%$ | `-3.8` to `+3.8` | MSCI World ex-USA developed equities return | High Market Beta ($F_1$) |
| `emerging_markets` | Float | $\%$ | `-4.5` to `+4.5` | MSCI Emerging Markets equity index return | High Market Beta + Commodity ($F_1, F_3$) |
| `tech_sector` | Float | $\%$ | `-5.0` to `+5.0` | Global Information Technology sector return | High Beta + Interest Rate sensitive ($F_1, -F_2$) |
| `energy_sector` | Float | $\%$ | `-4.8` to `+4.8` | Global Energy (Oil & Gas) sector index return | High Commodity / Inflation Beta ($F_3$) |
| `financials_sector`| Float | $\%$ | `-4.2` to `+4.2` | Global Banking & Financial Services index return | Market Beta + Positive Rate sensitive ($F_1, +F_2$) |
| `treasury_bonds` | Float | $\%$ | `-2.0` to `+2.0` | 10-Year Benchmark Government Treasury bond return | Negative Interest Rate / Duration ($ -F_2$) |
| `corporate_bonds` | Float | $\%$ | `-2.5` to `+2.5` | Investment-grade corporate credit bond return | Hybrid Market Beta + Duration ($F_1, -F_2$) |
| `commodities_gold`| Float | $\%$ | `-3.5` to `+3.5` | Gold & Precious Metals spot daily return | Safe-haven / Commodity Inflation ($F_3$) |
| `real_estate_reits`| Float | $\%$ | `-4.0` to `+4.0` | Global Real Estate Investment Trusts return | Equity Beta + Strong Duration ($F_1, -F_2$) |

---

## 🔬 Pedagogical Highlights for Chapter 3

1. **Dimensionality Reduction:** Compressing 10 correlated asset return streams into 3 interpretable macro risk components accounting for $>75\%$ of total market variance.
2. **Covariance vs. Correlation PCA:** Demonstrates the necessity of standardization when variables exhibit differing volatilities (e.g. Treasury bonds $\sigma \approx 0.9\%$ vs. Tech sector $\sigma \approx 1.8\%$).
3. **Component Retention Criteria:**
   - **Kaiser Criterion:** Number of eigenvalues $\lambda_j > 1.0$.
   - **Cattell Scree Plot:** Visual inflection / elbow test.
   - **Cumulative Variance:** Retaining $75\%-85\%$ explained variance.
   - **Horn's Parallel Analysis:** Comparing sample eigenvalues against simulated random noise permutations.
4. **2D & 3D PCA Biplots:** Jointly visualizing asset loading vectors and trading day scores to identify risk regimes (e.g., inflationary sell-offs vs. risk-on rallies).
