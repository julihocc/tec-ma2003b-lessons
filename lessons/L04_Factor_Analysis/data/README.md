# Chapter 4 Data — Educational Assessment Dataset (`lessons/L04_Factor_Analysis/data/`)

This directory contains the dataset, generation script, and data dictionary for **Chapter 4: Factor Analysis**.

---

## 📂 Directory Contents

| File | Type | Description |
| :--- | :--- | :--- |
| [`fetch_educational.py`](fetch_educational.py) | Python Script | Deterministic data generator producing standardized student performance scores across 9 academic subject evaluations, driven by 3 latent factors (STEM/Analytical, Language/Humanities, Spatial/Visual Reasoning). |
| [`educational.csv`](educational.csv) | CSV (Git LFS) | Generated dataset containing 200 student assessment records across 9 academic metrics. |
| [`EDUCATIONAL_ASSESSMENT_DATA_DICTIONARY.md`](EDUCATIONAL_ASSESSMENT_DATA_DICTIONARY.md) | Markdown | Detailed data dictionary with variable definitions, scoring scales, latent factor structures, and uniqueness variances. |

---

## 📊 Dataset Summary

- **Observations:** 200 secondary/university students
- **Subject Scores ($p=9$):** `Math_Algebra`, `Math_Geometry`, `Physics`, `Language_Grammar`, `Language_Literature`, `History`, `Spatial_Visual`, `Mechanical_Reasoning`, `Abstract_Logic`.
- **Target Latent Factors ($m=3$):** Quantitative/STEM, Verbal/Humanities, Perceptual/Spatial.

---

## 🚀 Regeneration Command

```bash
uv run python lessons/L04_Factor_Analysis/data/fetch_educational.py
```
