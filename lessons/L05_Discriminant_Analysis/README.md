# Chapter 5 — Discriminant Analysis

This chapter covers supervised statistical classification, Fisher's Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), canonical discriminant functions, and Bayesian cost-optimal decision rules.

## Chapter Overview (Syllabus Section 5)

Discriminant Analysis seeks to separate and classify observations into distinct, known categorical groups based on a set of continuous predictor variables. This module establishes both the geometric variance-maximization perspective (Fisher) and the probabilistic likelihood perspective (Bayes), covering assumption testing, stepwise feature selection, and marketing segmentation case studies.

## Syllabus Topics & Subtopics

- **5.1** Discrimination for two normal multivariate populations
- **5.2** Function of costs and a priori probabilities
- **5.3** Basic discrimination
- **5.4** Stepwise selection
- **5.5** Canonical discriminant functions
- **5.6** Coding and commercial programs

## Directory Structure

```text
L05_Discriminant_Analysis/
├── data/                                         # Datasets and generation scripts
│   ├── fetch_marketing.py                        # Customer behavioral segmentation generator
│   ├── marketing.csv                             # Generated dataset (1,200 customers × 7 features)
│   ├── MARKETING_DATA_DICTIONARY.md              # Detailed variable descriptions
│   └── README.md                                 # Data documentation & regeneration guide
├── notebook/                                     # Interactive Jupyter notebooks
│   ├── marketing_discriminant_analysis.ipynb     # End-to-end customer classification case study
│   └── README.md                                 # Notebook walkthrough & cell progression
├── notes/                                        # Lecture notes and study guides
│   ├── discriminant_analysis_notes.typ           # Comprehensive study guide (Typst)
│   ├── discriminant_analysis_notes.pdf           # Compiled lecture note PDF
│   └── README.md                                 # Notes outline & compilation guide
├── presentation/                                 # Presentation slides
│   ├── discriminant_analysis_slides.typ          # Typst presentation (Touying university theme)
│   ├── discriminant_analysis_slides.pdf          # Compiled presentation PDF
│   └── README.md                                 # Presentation agenda & compilation guide
└── README.md                                     # Module documentation
```

## Usage & Quickstart

### 1. Generate Dataset
```bash
uv run python lessons/L05_Discriminant_Analysis/data/fetch_marketing.py
```

### 2. Run Interactive Notebook
```bash
uv run jupyter notebook lessons/L05_Discriminant_Analysis/notebook/marketing_discriminant_analysis.ipynb
```

### 3. Compile Slides & Documents (Typst)
```bash
# Presentation
cd lessons/L05_Discriminant_Analysis/presentation/
typst compile discriminant_analysis_slides.typ

# Lecture Notes
cd ../notes/
typst compile discriminant_analysis_notes.typ
```
