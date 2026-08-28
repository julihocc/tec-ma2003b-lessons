# Chapter 6 Data — E-Commerce Customer Clustering Dataset (`lessons/L06_Cluster_Analysis/data/`)

This directory contains the dataset, generation script, and data dictionary for **Chapter 6: Analysis by Conglomerates (Cluster Analysis)**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`fetch_customer_data.py`](fetch_customer_data.py) | Python Script | Deterministic data generator producing realistic multi-channel e-commerce customer transaction profiles with 4 natural behavioral clusters (*Bargain Hunters*, *Loyal Everyday*, *High-Value VIPs*, *Occasional Spenders*). |
| [`customer_data.csv`](customer_data.csv) | CSV (Git LFS) | Unlabeled raw dataset containing 2,000 customer observations across 7 behavioral metrics. |
| [`customer_data_with_labels.csv`](customer_data_with_labels.csv) | CSV (Git LFS) | Ground-truth labeled dataset for benchmarking clustering accuracy and external validation metrics (ARI, NMI). |
| [`CUSTOMER_DATA_DICTIONARY.md`](CUSTOMER_DATA_DICTIONARY.md) | Markdown | Detailed data dictionary with variable definitions, units, value ranges, and cluster profile characteristics. |

---

## 📊 Dataset Summary

- **Observations:** 2,000 online shoppers
- **Behavioral Features ($p=7$):** `annual_income`, `spending_score`, `total_purchases_annual`, `avg_order_value`, `website_visits_monthly`, `return_rate`, `discount_usage_pct`.
- **True Underlying Clusters ($k=4$):** Budget/Bargain, Steady Loyal, High-Value VIP, Low-Frequency Browser.

---

## 🚀 Regeneration Command

```bash
uv run python lessons/L06_Cluster_Analysis/data/fetch_customer_data.py
```
