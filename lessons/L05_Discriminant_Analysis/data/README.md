# Chapter 5 Data — Marketing Segmentation Dataset (`lessons/L05_Discriminant_Analysis/data/`)

This directory contains the dataset, generation script, and variable dictionary for **Chapter 5: Discriminant Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`fetch_marketing.py`](fetch_marketing.py) | Python Script | Deterministic data generator simulating 1,200 consumer profiles categorized into 3 distinct purchasing tiers (*Budget*, *Mainstream*, *Premium*) based on behavioral and financial features. |
| [`marketing.csv`](marketing.csv) | CSV (Git LFS) | Generated dataset containing 1,200 customer records across 7 features and 1 categorical segment target. |
| [`MARKETING_DATA_DICTIONARY.md`](MARKETING_DATA_DICTIONARY.md) | Markdown | Detailed data dictionary with variable definitions, units, target class distributions, and group covariance structures. |

---

## 📊 Dataset Summary

- **Observations:** 1,200 retail banking & e-commerce customers
- **Target Classes ($K=3$):** `Budget` (400), `Mainstream` (500), `Premium` (300).
- **Predictor Features ($p=7$):** `annual_income_k`, `spending_score`, `savings_k`, `credit_score`, `avg_transaction_amt`, `purchase_frequency_monthly`, `digital_engagement_score`.

---

## 🚀 Regeneration Command

```bash
uv run python lessons/L05_Discriminant_Analysis/data/fetch_marketing.py
```
