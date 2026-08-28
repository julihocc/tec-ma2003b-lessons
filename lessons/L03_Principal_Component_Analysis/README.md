# Chapter 3 — Principal Component Analysis (PCA)

This chapter covers Principal Component Analysis (PCA) for dimensionality reduction, feature orthogonalization, and multivariate visualization.

## Chapter Overview (Syllabus Section 3)

Principal Component Analysis transforms a set of correlated variables into a smaller set of uncorrelated variables (principal components) that capture the maximum possible variance. It forms the foundation for exploratory dimensionality reduction and serves as the contrastive baseline for Factor Analysis.

## Syllabus Topics & Subtopics

- **3.1** Cases where PCA is used
- **3.2** Geometrical description and classification of major components
- **3.3** Estimation of major components
- **3.4** Determination of the appropriate number of major components
- **3.5** Coding and commercial programs

## Directory Structure

```text
L03_Principal_Component_Analysis/
├── data/                                         # Datasets and generation scripts
│   ├── fetch_financial_pca.py                    # Multi-asset returns market data generator
│   ├── financial_market_data.csv                 # Generated dataset (600 trading days × 10 assets)
│   └── FINANCIAL_MARKET_DATA_DICTIONARY.md       # Detailed variable descriptions
├── docs/                                         # Lecture notes and study guides
│   └── principal_component_analysis_notes.typ    # Comprehensive study guide (Typst)
├── notebook/                                     # Interactive Jupyter notebooks
│   └── financial_pca_analysis.ipynb              # Global market factor decomposition case study
├── presentation/                                 # Presentation slides
│   └── principal_component_analysis_slides.typ   # Typst presentation (Touying university theme)
└── README.md                                     # Module documentation
```

## Usage & Execution

### 1. Generate Dataset
```bash
uv run python lessons/L03_Principal_Component_Analysis/data/fetch_financial_pca.py
```

### 2. Run Interactive Notebook
```bash
uv run jupyter notebook lessons/L03_Principal_Component_Analysis/notebook/financial_pca_analysis.ipynb
```

### 3. Compile Slides & Documents (Typst)
```bash
# Presentation
cd lessons/L03_Principal_Component_Analysis/presentation/
typst compile principal_component_analysis_slides.typ

# Lecture Notes
cd ../docs/
typst compile principal_component_analysis_notes.typ
```
