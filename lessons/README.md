# Course Lessons & Instructional Modules (`lessons/`)

This directory houses all 7 instructional modules for **MA2003B — Application of Multivariate Methods in Data Science** at **Tecnológico de Monterrey**.

---

## 🏛️ Standard 4-Folder Module Architecture

Every lesson folder (`L01` through `L07`) follows an identical, standardized **4-folder structure**:

```text
L##_Topic_Name/
├── data/                                         # Datasets, generators, and data dictionaries
│   ├── fetch_<dataset>.py                        # Reproducible data generation script
│   ├── <dataset>.csv                             # Cleaned dataset (tracked via Git LFS)
│   ├── <DATASET>_DATA_DICTIONARY.md              # Variable descriptions, units, and ranges
│   └── README.md                                 # Data folder documentation & usage
├── notebook/                                     # Interactive Jupyter notebooks & case studies
│   ├── <topic>_analysis.ipynb                    # Primary end-to-end case study notebook
│   └── README.md                                 # Notebook walkthrough, cell flow, and outputs
├── notes/                                        # Lecture notes & theoretical study guides
│   ├── <topic>_notes.typ                         # Typst markup source for lecture notes
│   ├── <topic>_notes.pdf                         # Compiled lecture note PDF (Git LFS)
│   └── README.md                                 # Mathematical outline & compilation commands
├── presentation/                                 # Classroom slide decks
│   ├── <topic>_slides.typ                        # Touying university-theme presentation source
│   ├── <topic>_slides.pdf                        # Compiled slide deck PDF (Git LFS)
│   └── README.md                                 # Slide agenda & presentation compilation guide
└── README.md                                     # Chapter overview, syllabus subtopics, and quickstart
```

---

## 📚 Curriculum Navigation & Topic Overview

| Module | Chapter Title | Syllabus Section | Primary Methods & Diagnostics | Pedagogical Dataset |
| :---: | :--- | :---: | :--- | :--- |
| **[L01](L01_Regression_Analysis/)** | **Regression Analysis** | Section 1 | Simple/Multiple Linear Regression, ANOVA, Residual Q-Q, Stepwise Selection, Polynomial terms, VIF, Cook's D, Breusch-Pagan, HC3 Robust SE | Residential Property Valuation (1,000 homes × 9 features) |
| **[L02](L02_Multivariate_Analysis/)** | **Multivariate Analysis** | Section 2 | Random vectors, Mean/Covariance matrices, MVN geometry, KNN Imputation, Mahalanobis Distance, Robust MCD Outliers, Fisher $z$ CIs, Andrews Curves | Environmental Air Quality (800 stations × 8 features) |
| **[L03](L03_Principal_Component_Analysis/)** | **Principal Component Analysis (PCA)** | Section 3 | Spectral Decomposition of $\mathbf{S}$ and $\mathbf{R}$, Eigenvalues/Eigenvectors, Kaiser Rule, Scree Plots, Horn's Parallel Analysis, Loadings Heatmap, 2D/3D Biplots | Global Financial Asset Returns (600 trading days × 10 assets) |
| **[L04](L04_Factor_Analysis/)** | **Factor Analysis** | Section 4 | Exploratory Factor Analysis (EFA), Principal Axis Factoring, Maximum Likelihood, Varimax/Promax Rotation, Communalities, Latent Factor Scores | Educational Assessment (200 students × 9 metrics) |
| **[L05](L05_Discriminant_Analysis/)** | **Discriminant Analysis** | Section 5 | Fisher's Linear Discriminant (LDA), Quadratic Discriminant (QDA), Canonical Discriminant Functions, Wilks' $\Lambda$, Expected Cost of Misclassification (ECM) | Customer Marketing Segmentation (1,200 customers × 7 features) |
| **[L06](L06_Cluster_Analysis/)** | **Analysis by Conglomerates (Cluster Analysis)** | Section 6 | Hierarchical Clustering (Ward's, Complete, Single), K-Means, Silhouette Analysis, Elbow Method, Dendrogram Visualization, 2D PCA Projections | E-Commerce Shopper Behavior (2,000 customers × 7 features) |
| **[L07](L07_Multivariate_Regression/)** | **Multivariate Regression** | Section 7 | Multivariate Linear Regression $\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}$, Hotelling's $T^2$, MANOVA (Wilks, Pillai, Hotelling-Lawley, Roy), Canonical Correlation Analysis (CCA) | Cardiovascular Health Risk Assessment (1,000 patients × 13 features) |

---

## ⚡ Unified Execution Commands with `uv`

All lesson assets can be executed directly using the modern `uv` workspace environment:

```bash
# 1. Regenerate any lesson dataset
uv run python lessons/L01_Regression_Analysis/data/fetch_housing_regression.py

# 2. Run all 7 primary notebooks from clean kernels
uv run python scripts/run_notebooks.py

# 3. Run factor analysis snippet validation suite
uv run python lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py

# 4. Launch Jupyter Notebook interface
uv run jupyter notebook
```
