# Chapter 1 — Regression Analysis

This chapter covers univariate and multiple regression analysis, model diagnostics, variable selection methods, and assumption testing in statistical modeling.

## Chapter Overview (Syllabus Section 1)

Regression analysis explores functional and predictive relationships between explanatory variables and outcome metrics. This module establishes core modeling principles including estimation, ANOVA tables, residual diagnostics, automated variable selection, and remedies for heteroskedasticity.

## Syllabus Topics & Subtopics

- **1.1** Simple linear regression
- **1.2** ANOVA and confidence intervals for predictions
- **1.3** Residual analysis and condition of normality
- **1.4** Forward selection
- **1.5** Backward selection
- **1.6** Non-linear regression
- **1.7** Multiple linear regression
- **1.8** Aberrant data and heteroskedasticity problems

## Directory Structure

```text
L01_Regression_Analysis/
├── data/                                         # Datasets and generation scripts
│   ├── fetch_housing_regression.py               # Synthetic property transaction generator
│   ├── housing_regression.csv                    # Generated dataset (1,000 homes × 9 metrics)
│   └── HOUSING_REGRESSION_DATA_DICTIONARY.md     # Detailed variable descriptions
├── notes/                                        # Lecture notes and study guides
│   └── regression_analysis_notes.tex             # Comprehensive study guide (LaTeX)
├── notebook/                                     # Interactive Jupyter notebooks
│   └── housing_regression_analysis.ipynb         # End-to-end residential valuation case study
├── presentation/                                 # Presentation slides
│   └── regression_analysis_slides.tex            # LaTeX Beamer presentation
└── README.md                                     # Module documentation
```

## Usage & Execution

### 1. Generate Dataset
```bash
uv run python lessons/L01_Regression_Analysis/data/fetch_housing_regression.py
```

### 2. Run Interactive Notebook
```bash
uv run jupyter notebook lessons/L01_Regression_Analysis/notebook/housing_regression_analysis.ipynb
```

### 3. Compile Slides & Documents (LaTeX)
```bash
# Presentation
cd lessons/L01_Regression_Analysis/presentation/
pdflatex regression_analysis_slides.tex

# Lecture Notes
cd ../notes/
pdflatex regression_analysis_notes.tex
```
