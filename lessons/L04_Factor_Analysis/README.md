# Chapter 4 — Factor Analysis

This chapter covers Factor Analysis (FA) techniques for latent variable modeling, common variance decomposition, and dimensionality reduction, along with pedagogical comparisons to Principal Component Analysis (PCA).

## Chapter Overview

Factor Analysis is a statistical method used to identify underlying latent variables (factors) that explain the patterns of correlations within a set of observed variables. Unlike Principal Component Analysis (PCA), which focuses on explaining maximum total variance, Factor Analysis specifically models the common variance shared among variables while treating unique variance as measurement error or specific factors.

The fundamental goal is to find a smaller number of latent factors that can adequately reproduce the observed correlation matrix, providing insights into the underlying structure of the data.

## Learning Objectives

By the end of this chapter, students will be able to:

- Distinguish between Factor Analysis and Principal Component Analysis
- Understand the common factor model ($\mathbf{X} - \boldsymbol{\mu} = \mathbf{\Lambda F} + \boldsymbol{\varepsilon}$)
- Apply different factor extraction methods (Principal Axis Factoring, Maximum Likelihood)
- Determine the optimal number of factors using multiple criteria (Kaiser, Scree plot, Parallel Analysis)
- Interpret factor loadings, communalities ($h^2$), and uniquenesses ($\psi^2$)
- Apply orthogonal (Varimax) and oblique (Promax) rotation techniques for simple structure
- Implement factor analysis workflows in Python (`factor_analyzer`, `scikit-learn`)

## Directory Structure

```text
L04_Factor_Analysis/
├── data/                                         # Datasets and generation scripts
│   ├── fetch_educational.py                      # Synthetic educational assessment data generator
│   ├── educational.csv                           # Generated student dataset (200 students × 9 metrics)
│   └── EDUCATIONAL_ASSESSMENT_DATA_DICTIONARY.md # Detailed variable descriptions
├── notes/                                        # Lecture notes and reports
│   ├── factor_analysis_notes.typ                 # Comprehensive study guide (Typst)
│   └── executive_report.typ                      # Executive assessment report (Typst)
├── notebook/                                     # Interactive Jupyter notebooks
│   ├── educational_analysis.ipynb                # End-to-end educational assessment case study
│   └── snippets/                                 # Hands-on tutorial notebook series
│       ├── 01_pca_basic_example.ipynb            # Basic PCA concepts
│       ├── 02_component_retention.ipynb          # Retention criteria (Kaiser, scree plot)
│       ├── 03_factor_analysis_basic.ipynb        # Basic factor analysis
│       ├── 04_factor_rotation.ipynb              # Rotation comparison (Varimax, Promax)
│       ├── 05_complete_workflow.ipynb            # Complete end-to-end factor analysis workflow
│       ├── README.md                             # Snippets guide
│       ├── TESTING_RESULTS.md                    # Test execution results
│       └── test_all_snippets.py                  # Automated test script
├── presentation/                                 # Presentation slides
│   ├── factor_analysis_slides.typ                # Typst presentation (Touying university theme)
│   └── fa_presentation.pdf                       # Reference compiled presentation
└── README.md                                     # Module documentation
```

## Usage & Execution

### 1. Generate Dataset
```bash
python data/fetch_educational.py
```

### 2. Run Interactive Notebook
```bash
jupyter notebook notebook/educational_analysis.ipynb
```

### 3. Run Snippet Tutorial Test Suite
```bash
python notebook/snippets/test_all_snippets.py
```

### 4. Compile Slides & Documents (Typst)
```bash
# Presentation
cd presentation/
typst compile factor_analysis_slides.typ

# Study Guide Notes
cd ../notes/
typst compile factor_analysis_notes.typ
```
