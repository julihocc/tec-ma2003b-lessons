# MA2003B — Application of Multivariate Methods in Data Science

**Tecnológico de Monterrey**  
**Instructor:** Dr. Juliho Castillo Colmenares  
**Course Code:** MA2003B  
**Academic Program:** 5 IDM19 (Ingeniería en Ciencia de Datos y Matemáticas)  
**Syllabus Documentation:** [Plan Analítico (PDF)](docs/MA2003B%20-%20Anal%C3%ADtico.pdf) | [Syllabus Summary (Markdown)](docs/SYLLABUS.md)

This repository contains teaching materials, lecture notes, slide presentations, synthetic datasets, and interactive Jupyter notebooks for the course *Application of Multivariate Methods in Data Science* (MA2003B).

---

## 📂 Repository Architecture

```text
tec-ma2003b-lessons/
├── .gitattributes                                 # Git LFS tracking configuration (*.ipynb, *.csv, *.pdf)
├── .python-version                                # Pinned Python runtime (3.11)
├── docs/                                          # Course-wide documentation & syllabus
│   ├── SYLLABUS.md                                # Complete syllabus transcription
│   └── MA2003B - Analítico.pdf                    # Official Plan Analítico PDF (LFS)
├── lessons/                                       # Course lesson modules (Chapters 1 to 7)
│   ├── L01_Regression_Analysis/                   # Section 1: Regression Analysis
│   ├── L02_Multivariate_Analysis/                 # Section 2: Multivariate Analysis
│   ├── L03_Principal_Component_Analysis/          # Section 3: Principal Component Analysis (PCA)
│   ├── L04_Factor_Analysis/                       # Section 4: Factor Analysis
│   ├── L05_Discriminant_Analysis/                 # Section 5: Discriminant Analysis
│   ├── L06_Cluster_Analysis/                      # Section 6: Analysis by Conglomerates
│   └── L07_Multivariate_Regression/               # Section 7: Multivariate Regression
├── pyproject.toml                                 # uv project configuration
├── uv.lock                                        # Deterministic dependency lockfile
├── requirements.txt                               # Pip-compatible dependency export
└── README.md                                      # Repository overview
```

Every lesson module inside `lessons/` follows a standardized **4-folder layout**:
- `data/`: Dataset files (`.csv`), synthetic data generators (`fetch_*.py`), and data dictionaries (`*.md`).
- `docs/`: Module lecture notes and theoretical study guides (`<topic>_notes.typ` / `.pdf`).
- `notebook/`: Hands-on case studies and interactive Jupyter notebooks (`.ipynb`).
- `presentation/`: Slide presentations built with [Typst](https://typst.app/) + [Touying](https://github.com/touying-typ/touying) (`<topic>_slides.typ` / `.pdf`).

---

## 📚 Curriculum & Module Overview

| Section | Module Folder | Official Topic Title | Primary Methods | Case Study Dataset |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **[L01](lessons/L01_Regression_Analysis/)** | **Regression Analysis** | Simple/Multiple Linear, ANOVA, Residuals, Stepwise selection, Heteroskedasticity, HC3 | Residential Property Valuation (1,000 homes × 9 metrics) |
| **2** | **[L02](lessons/L02_Multivariate_Analysis/)** | **Multivariate Analysis** | MVN distributions, Mean vectors, Covariance/Correlation, Robust MCD Outliers, Fisher $z$ | Environmental Air Quality Monitoring (800 stations × 8 metrics) |
| **3** | **[L03](lessons/L03_Principal_Component_Analysis/)** | **Principal Component Analysis (PCA)** | Spectral decomposition, Eigenvalues, Scree plots, Horn's Parallel Analysis, Biplots | Global Financial Asset Returns (600 trading days × 10 assets) |
| **4** | **[L04](lessons/L04_Factor_Analysis/)** | **Factor Analysis** | Common Factor Model, Principal Axis, ML extraction, Varimax/Promax rotation | Educational Assessment (200 students × 9 metrics) |
| **5** | **[L05](lessons/L05_Discriminant_Analysis/)** | **Discriminant Analysis** | Fisher's LDA, QDA, Canonical discriminant functions, Wilks' Lambda, ECM | Customer Marketing Segmentation (1,200 customers) |
| **6** | **[L06](lessons/L06_Cluster_Analysis/)** | **Analysis by Conglomerates (Cluster Analysis)** | Hierarchical (Ward's), K-Means, Elbow & Silhouette metrics, 2D PCA | E-Commerce Shopper Behavior (2,000 customers) |
| **7** | **[L07](lessons/L07_Multivariate_Regression/)** | **Multivariate Regression** | Logistic Regression, Hotelling's $T^2$, MANOVA, Canonical Correlation (CCA) | Cardiovascular Health Risk (1,000 patients) |

---

## 🚀 Quickstart with `uv`

This project uses [`uv`](https://docs.astral.sh/uv/) for Python dependency and environment management.

### 1. Install & Sync Environment

```bash
# Clone the repository
git clone https://github.com/julihocc/tec-ma2003b-lessons.git
cd tec-ma2003b-lessons

# Install dependencies and create .venv automatically
uv sync
```

### 2. Generating Datasets

Each module includes a self-contained data generation script:

```bash
# Chapter 1: Regression Analysis
uv run python lessons/L01_Regression_Analysis/data/fetch_housing_regression.py

# Chapter 2: Multivariate Analysis
uv run python lessons/L02_Multivariate_Analysis/data/fetch_environmental_data.py

# Chapter 3: Principal Component Analysis
uv run python lessons/L03_Principal_Component_Analysis/data/fetch_financial_pca.py

# Chapter 4: Factor Analysis
uv run python lessons/L04_Factor_Analysis/data/fetch_educational.py

# Chapter 5: Discriminant Analysis
uv run python lessons/L05_Discriminant_Analysis/data/fetch_marketing.py

# Chapter 6: Cluster Analysis
uv run python lessons/L06_Cluster_Analysis/data/fetch_customer_data.py

# Chapter 7: Multivariate Regression
uv run python lessons/L07_Multivariate_Regression/data/fetch_health_data.py
```

### 3. Launching Jupyter Notebooks

```bash
uv run jupyter notebook
```

Navigate to any lesson `notebook/` subfolder (e.g. `lessons/L01_Regression_Analysis/notebook/housing_regression_analysis.ipynb`) to run the interactive analyses.

### 4. Running Snippet Tests

```bash
uv run python lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py
```

### 5. Compiling Slides & Documents (Typst)

Presentations and lecture notes are typeset using [Typst](https://typst.app/):

```bash
# Compile slides (example for L01)
cd lessons/L01_Regression_Analysis/presentation
typst compile regression_analysis_slides.typ

# Compile lecture notes (example for L01)
cd ../docs
typst compile regression_analysis_notes.typ
```

---

## 🛠️ Technology Stack

- **Environment & Package Manager:** [uv](https://docs.astral.sh/uv/) with `pyproject.toml` and `uv.lock`
- **Computing & Statistics:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `factor_analyzer`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Interactive Notebooks:** Jupyter / JupyterLab
- **Typesetting & Slides:** [Typst](https://typst.app/) with `@preview/touying:0.5.3` (University Theme, Tec de Monterrey branding)
- **Asset Storage:** Git LFS (`*.ipynb`, `*.csv`, `*.pdf`)

---

## 📄 License & Attribution

Materials developed for academic instruction at **Tecnológico de Monterrey**.  
© Dr. Juliho Castillo Colmenares. All rights reserved.
