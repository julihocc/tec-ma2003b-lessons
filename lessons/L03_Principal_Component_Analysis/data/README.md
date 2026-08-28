# Chapter 3 Data — Financial Market PCA Dataset (`lessons/L03_Principal_Component_Analysis/data/`)

This directory contains the dataset, generation script, and variable dictionary for **Chapter 3: Principal Component Analysis (PCA)**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`fetch_financial_pca.py`](fetch_financial_pca.py) | Python Script | Deterministic synthetic generator producing daily return time series across 10 global asset classes and sectors, driven by 3 latent macroeconomic factors (Market Beta, Duration/Rates, Commodity Inflation). |
| [`financial_market_data.csv`](financial_market_data.csv) | CSV (Git LFS) | Generated dataset containing 600 trading session returns across 10 asset classes. |
| [`FINANCIAL_MARKET_DATA_DICTIONARY.md`](FINANCIAL_MARKET_DATA_DICTIONARY.md) | Markdown | Detailed data dictionary with asset descriptions, benchmarks, volatility profiles, and factor sensitivities. |

---

## 📊 Dataset Summary

- **Observations:** 600 trading sessions
- **Asset Returns ($p=10$):** `us_equities`, `global_equities`, `emerging_markets`, `tech_sector`, `energy_sector`, `financials_sector`, `treasury_bonds`, `corporate_bonds`, `commodities_gold`, `real_estate_reits`.
- **Target Dimensions:** Dimensionality reduction from 10 correlated asset streams to 3 orthogonal principal components explaining $>75\%$ of systemic variance.

---

## 🚀 Regeneration Command

```bash
uv run python lessons/L03_Principal_Component_Analysis/data/fetch_financial_pca.py
```
