# Chapter 7 — Multivariate Regression

This chapter covers multivariate linear regression with multiple response variables, inference for covariance matrices, Hotelling's $T^2$ hypothesis testing, Multivariate Analysis of Variance (MANOVA), Logistic regression, and Canonical Correlation Analysis (CCA).

## Chapter Overview (Syllabus Section 7)

Multivariate Regression models the simultaneous joint behavior of multiple response variables ($\mathbf{Y} \in \mathbb{R}^m$) as a function of multiple explanatory variables ($\mathbf{X} \in \mathbb{R}^{k+1}$). This module establishes cross-equation covariance estimation, simultaneous multivariate hypothesis testing (MANOVA, Hotelling's $T^2$), categorical binary classification (Logistic Regression), and dual-battery association analysis (Canonical Correlation Analysis).

## Syllabus Topics & Subtopics

- **7.1** Logistic regression model
- **7.2** Inferences for variances and covariances matrices
- **7.3** Inferences for a vector of means
- **7.4** MANOVA
- **7.5** Canonical correlation analysis
- **7.6** Analysis by factors and regression
- **7.7** Programming and commercial systems

## Directory Structure

```text
L07_Multivariate_Regression/
├── data/                                         # Datasets and generation scripts
│   ├── fetch_health_data.py                      # Clinical cardiovascular health generator
│   ├── health_data.csv                           # Generated dataset (1,000 patients × 13 metrics)
│   ├── HEALTH_DATA_DICTIONARY.md                 # Detailed variable descriptions
│   ├── HEALTH_DATA_DICTIONARY.pdf                # Formatted data dictionary PDF
│   └── README.md                                 # Data documentation & regeneration guide
├── notebook/                                     # Interactive Jupyter notebooks
│   ├── health_risk_analysis.ipynb                # End-to-end clinical multivariate case study
│   └── README.md                                 # Notebook walkthrough & cell progression
├── notes/                                        # Lecture notes and study guides
│   ├── multivariate_regression_notes.typ         # Comprehensive study guide (Typst)
│   ├── multivariate_regression_notes.pdf         # Compiled lecture note PDF
│   └── README.md                                 # Notes outline & compilation guide
├── presentation/                                 # Presentation slides
│   ├── multivariate_regression_slides.typ        # Typst presentation (Touying university theme)
│   ├── multivariate_regression_slides.pdf        # Compiled presentation PDF
│   └── README.md                                 # Presentation agenda & compilation guide
└── README.md                                     # Module documentation
```

## Usage & Quickstart

### 1. Generate Dataset
```bash
uv run python lessons/L07_Multivariate_Regression/data/fetch_health_data.py
```

### 2. Run Interactive Notebook
```bash
uv run jupyter notebook lessons/L07_Multivariate_Regression/notebook/health_risk_analysis.ipynb
```

### 3. Compile Slides & Documents (Typst)
```bash
# Presentation
cd lessons/L07_Multivariate_Regression/presentation/
typst compile multivariate_regression_slides.typ

# Lecture Notes
cd ../notes/
typst compile multivariate_regression_notes.typ
```
