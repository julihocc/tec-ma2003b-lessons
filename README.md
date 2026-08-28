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
├── docs/                                          # Course-wide documentation & syllabus
│   ├── SYLLABUS.md                                # Complete syllabus transcription
│   └── MA2003B - Analítico.pdf                    # Official Plan Analítico PDF
├── lessons/                                       # Course lesson modules (Chapters 1 to 7)
│   ├── L01_Regression_Analysis/
│   ├── L02_Multivariate_Analysis/
│   ├── L03_Principal_Component_Analysis/
│   ├── L04_Factor_Analysis/
│   ├── L05_Discriminant_Analysis/
│   ├── L06_Cluster_Analysis/
│   └── L07_Multivariate_Regression/
├── requirements.txt                               # Unified course dependencies
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
| **1** | **[L01](lessons/L01_Regression_Analysis/)** | **Regression Analysis** | Simple/Multiple Linear, ANOVA, Residuals, Stepwise selection, Heteroskedasticity | Predictive regression modeling |
| **2** | **[L02](lessons/L02_Multivariate_Analysis/)** | **Multivariate Analysis** | MVN distributions, Mean vectors, Covariance/Correlation, Outliers, Fisher/Ruben intervals | Multivariate exploratory diagnostics |
| **3** | **[L03](lessons/L03_Principal_Component_Analysis/)** | **Principal Component Analysis (PCA)** | Spectral decomposition, Eigenvalues, Scree plots, Biplots | Dimensionality reduction |
| **4** | **[L04](lessons/L04_Factor_Analysis/)** | **Factor Analysis** | Common Factor Model, Principal Axis, ML extraction, Varimax/Promax rotation | Educational Assessment (200 students × 9 metrics) |
| **5** | **[L05](lessons/L05_Discriminant_Analysis/)** | **Discriminant Analysis** | Fisher's LDA, QDA, Canonical discriminant functions, Wilks' Lambda, ECM | Customer Marketing Segmentation (1,200 customers) |
| **6** | **[L06](lessons/L06_Cluster_Analysis/)** | **Analysis by Conglomerates (Cluster Analysis)** | Hierarchical (Ward's), K-Means, Elbow & Silhouette metrics, 2D PCA | E-Commerce Shopper Behavior (2,000 customers) |
| **7** | **[L07](lessons/L07_Multivariate_Regression/)** | **Multivariate Regression** | Logistic Regression, Hotelling's $T^2$, MANOVA, Canonical Correlation (CCA) | Cardiovascular Health Risk (1,000 patients) |

---

## 🚀 Quickstart & Setup

### 1. Environment Setup

Clone the repository and install the unified dependencies:

```bash
# Clone the repository
git clone https://github.com/julihocc/tec-ma2003b-lessons.git
cd tec-ma2003b-lessons

# Create and activate a virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Generating Datasets

Each module includes a self-contained data generation script:

```bash
# Chapter 4: Factor Analysis
python lessons/L04_Factor_Analysis/data/fetch_educational.py

# Chapter 5: Discriminant Analysis
python lessons/L05_Discriminant_Analysis/data/fetch_marketing.py

# Chapter 6: Cluster Analysis
python lessons/L06_Cluster_Analysis/data/fetch_customer_data.py

# Chapter 7: Multivariate Regression
python lessons/L07_Multivariate_Regression/data/fetch_health_data.py
```

### 3. Launching Notebooks

```bash
jupyter notebook
```

Navigate to any lesson `notebook/` subfolder (e.g. `lessons/L06_Cluster_Analysis/notebook/customer_clustering_analysis.ipynb`) to run the interactive analyses.

### 4. Compiling Slides & Documents (Typst)

Presentations and lecture notes are typeset using [Typst](https://typst.app/):

```bash
# Compile slides (example for L06)
cd lessons/L06_Cluster_Analysis/presentation
typst compile cluster_analysis_slides.typ

# Compile lecture notes (example for L06)
cd ../docs
typst compile cluster_analysis_notes.typ
```

---

## 🛠️ Technology Stack

- **Computing & Statistics:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `factor_analyzer`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Interactive Notebooks:** Jupyter / JupyterLab
- **Typesetting & Slides:** [Typst](https://typst.app/) with `@preview/touying:0.5.3` (University Theme, Tec de Monterrey branding)

---

## 📄 License & Attribution

Materials developed for academic instruction at **Tecnológico de Monterrey**.  
© Dr. Juliho Castillo Colmenares. All rights reserved.
