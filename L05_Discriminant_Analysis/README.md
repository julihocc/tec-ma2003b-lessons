# Chapter 5 — Discriminant Analysis

This chapter covers Discriminant Analysis techniques for classification and group separation in multivariate data, focusing on Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and canonical discriminant functions.

## Chapter Overview

Discriminant Analysis is a supervised statistical method used to classify observations into predefined groups based on multiple predictor variables. The goal is to find linear combinations of predictors (discriminant functions) that best separate the groups while maximizing between-group differences and minimizing within-group variation.

Unlike purely black-box machine learning classifiers, discriminant analysis provides interpretable score functions and structural loadings that reveal which variables contribute most to group separation, making it valuable for understanding underlying population differences.

## Learning Objectives

By the end of this chapter, students will be able to:

- Distinguish between descriptive and predictive discriminant analysis
- Apply Fisher's Linear Discriminant Analysis for two-group separation
- Derive multi-group canonical discriminant functions ($k > 2$ classes)
- Incorporate prior probabilities and misclassification cost matrices (Expected Cost of Misclassification - ECM)
- Understand and test assumptions: multivariate normality, covariance homogeneity (Box's M test)
- Compare LDA vs. Quadratic Discriminant Analysis (QDA)
- Perform stepwise variable selection using Wilks' Lambda ($\Lambda$)
- Implement discriminant classification workflows in Python (`scikit-learn`, `scipy`)

## Directory Structure

```text
L05_Discriminant_Analysis/
├── data/                                   # Datasets and generation scripts
│   ├── fetch_marketing.py                  # Synthetic customer behavior data generator
│   ├── marketing.csv                       # Generated customer dataset (1,200 customers × 8 metrics)
│   └── MARKETING_DATA_DICTIONARY.md        # Detailed variable descriptions
├── docs/                                   # Lecture notes and study guides
│   ├── discriminant_analysis_notes.typ     # Comprehensive study guide (Typst)
│   └── discriminant_analysis_notes.pdf     # Reference compiled study guide PDF
├── notebook/                               # Interactive Jupyter notebooks
│   └── marketing_discriminant_analysis.ipynb # End-to-end customer segmentation case study
├── presentation/                           # Presentation slides
│   ├── discriminant_analysis_slides.typ    # Typst presentation (Touying university theme)
│   └── discriminant_analysis_slides.pdf    # Reference compiled presentation PDF
└── README.md                               # Module documentation
```

## Usage & Execution

### 1. Generate Dataset
```bash
python data/fetch_marketing.py
```

### 2. Run Interactive Notebook
```bash
jupyter notebook notebook/marketing_discriminant_analysis.ipynb
```

### 3. Compile Slides & Documents (Typst)
```bash
# Presentation
cd presentation/
typst compile discriminant_analysis_slides.typ

# Study Guide Notes
cd ../docs/
typst compile discriminant_analysis_notes.typ
```
