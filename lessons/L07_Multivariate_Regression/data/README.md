# Chapter 7 Data — Cardiovascular Health Risk Dataset (`lessons/L07_Multivariate_Regression/data/`)

This directory contains the dataset, generation script, and data dictionaries for **Chapter 7: Multivariate Regression**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`fetch_health_data.py`](fetch_health_data.py) | Python Script | Deterministic data generator simulating 1,000 patient clinical health profiles across demographic, lifestyle, physiological biomarkers, and dual continuous response variables (`systolic_bp`, `fasting_glucose`) and binary diagnosis (`cvd_risk_flag`). |
| [`health_data.csv`](health_data.csv) | CSV (Git LFS) | Generated dataset containing 1,000 clinical records across 13 health metrics. |
| [`HEALTH_DATA_DICTIONARY.md`](HEALTH_DATA_DICTIONARY.md) | Markdown | Detailed data dictionary with variable definitions, units, physiological normal ranges, and multivariate modeling roles. |
| [`HEALTH_DATA_DICTIONARY.pdf`](HEALTH_DATA_DICTIONARY.pdf) | PDF (Git LFS) | Publication-formatted data dictionary PDF document. |

---

## 📊 Dataset Summary

- **Observations:** 1,000 adult patients
- **Joint Continuous Responses ($\mathbf{Y} \in \mathbb{R}^2$):** `systolic_bp`, `fasting_glucose`.
- **Categorical Risk Classification:** `cvd_risk_flag` (0: Low/Moderate, 1: High Risk).
- **Predictor Variables ($\mathbf{X} \in \mathbb{R}^{10}$):** `age`, `bmi`, `cholesterol_total`, `ldl`, `hdl`, `triglycerides`, `physical_activity_hrs`, `smoking_pack_years`, `diet_quality_score`, `stress_index`.

---

## 🚀 Regeneration Command

```bash
uv run python lessons/L07_Multivariate_Regression/data/fetch_health_data.py
```
