# Chapter 1 Data — Housing Regression Dataset (`lessons/L01_Regression_Analysis/data/`)

This directory contains the dataset, generation script, and variable dictionary for **Chapter 1: Regression Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`fetch_housing_regression.py`](fetch_housing_regression.py) | Python Script | Deterministic synthetic data generator creating realistic residential property transaction records with non-linear effects and heteroskedasticity. |
| [`housing_regression.csv`](housing_regression.csv) | CSV (Git LFS) | Generated dataset containing 1,000 residential transactions across 9 metrics. |
| [`HOUSING_REGRESSION_DATA_DICTIONARY.md`](HOUSING_REGRESSION_DATA_DICTIONARY.md) | Markdown | Detailed data dictionary with variable definitions, units, value ranges, and modeling roles. |

---

## 📊 Dataset Summary

- **Observations:** 1,000 residential transactions
- **Target Variable ($Y$):** `sale_price` (USD)
- **Predictors ($X$):** `sqft_living`, `bedrooms`, `bathrooms`, `house_age_years`, `dist_city_center_km`, `building_grade`, `energy_efficiency_rating`.

---

## 🚀 Regeneration Command

To regenerate the dataset from scratch:

```bash
uv run python lessons/L01_Regression_Analysis/data/fetch_housing_regression.py
```
