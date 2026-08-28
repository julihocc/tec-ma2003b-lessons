# Chapter 4 Notebooks & Snippets (`lessons/L04_Factor_Analysis/notebook/`)

This directory contains the primary case study notebook and modular concept snippets for **Chapter 4: Factor Analysis**.

---

## 📂 Directory Contents

| Notebook / Subfolder | Type | Purpose | Status |
| :--- | :--- | :--- | :---: |
| [`educational_analysis.ipynb`](educational_analysis.ipynb) | Primary Notebook | End-to-end Factor Analysis case study evaluating factorability (KMO, Bartlett's test), factor extraction (Principal Axis & ML), Varimax/Promax rotations, communalities ($h^2$), and student factor scores. | ✅ Verified |
| [`snippets/`](snippets/) | Subfolder | 5 focused pedagogical notebooks isolating specific mathematical concepts with an automated non-interactive testing harness ([`test_all_snippets.py`](snippets/test_all_snippets.py)). | ✅ 5/5 Passed |

---

## 🔬 Primary Notebook Workflow

1. **Exploratory Assessment & Factorability:**
   - Visualizing the 9-subject academic correlation matrix.
   - Kaiser-Meyer-Olkin (KMO) Measure of Sampling Adequacy ($\text{KMO} > 0.70$).
   - Bartlett's Test of Sphericity ($p < 0.001$, rejecting uncorrelated identity matrix).
2. **Factor Extraction:**
   - Principal Axis Factoring vs. Maximum Likelihood extraction.
   - Scree Plot and Kaiser eigenvalue criterion ($\lambda > 1.0$) confirming 3 latent factors.
3. **Factor Rotation:**
   - Orthogonal **Varimax** rotation (maximizing variance of squared loadings per factor).
   - Oblique **Promax** rotation (allowing realistic inter-factor correlation).
4. **Communalities & Uniqueness:**
   - Calculating shared variance $h_i^2 = \sum l_{ij}^2$ and specific uniqueness $\psi_i = 1 - h_i^2$.
5. **Factor Scoring & Interpretation:**
   - Computing Bartlett and Thomson-Thurstone factor scores for individual student profiling.

---

## 🚀 Execution Instructions

```bash
# Launch primary notebook in Jupyter
uv run jupyter notebook lessons/L04_Factor_Analysis/notebook/educational_analysis.ipynb

# Run snippet test harness
uv run python lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py
```
