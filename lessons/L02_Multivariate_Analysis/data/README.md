# Chapter 2 Data — Environmental Air Quality Dataset (`lessons/L02_Multivariate_Analysis/data/`)

This directory contains the dataset, generation script, and variable dictionary for **Chapter 2: Multivariate Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`fetch_environmental_data.py`](fetch_environmental_data.py) | Python Script | Deterministic synthetic generator producing realistic atmospheric sensor records with cross-pollutant covariance, missing sensor readings (~3%), and subtle multivariate outliers (~2.5%). |
| [`environmental_data.csv`](environmental_data.csv) | CSV (Git LFS) | Generated dataset containing 800 station observations across 8 atmospheric metrics. |
| [`ENVIRONMENTAL_DATA_DICTIONARY.md`](ENVIRONMENTAL_DATA_DICTIONARY.md) | Markdown | Detailed data dictionary describing pollutant types, physical units, ranges, and multivariate roles. |

---

## 📊 Dataset Summary

- **Observations:** 800 environmental monitoring stations
- **Atmospheric Metrics ($p=8$):** `pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `temperature`, `humidity`.
- **Special Characteristics:** Includes missing values (for MCAR/MAR imputation) and multivariate outliers (for Mahalanobis and Robust MCD detection).

---

## 🚀 Regeneration Command

```bash
uv run python lessons/L02_Multivariate_Analysis/data/fetch_environmental_data.py
```
