# Chapter 2 — Multivariate Analysis

This chapter covers foundational concepts in multivariate distributions, matrix representations, data cleaning, correlation inference, and multidimensional visualization.

## Chapter Overview (Syllabus Section 2)

Multivariate analysis involves the simultaneous observation and statistical treatment of multiple random variables. This module introduces joint probability distributions, mean vectors, covariance/correlation structures, multivariate normality, data quality inspection, and sample correlation intervals.

## Syllabus Topics & Subtopics

- **2.1** Multivariate distributions
- **2.2** Mean vectors and variance and covariance matrices
- **2.3** Correlations and correlation matrices
- **2.4** Multivariate normal probability density function
- **2.5** Lost, null, incorrect values and detection of discrepancies
- **2.6** Multivariate aberrant data
- **2.7** Sample correlations, Fisher and Ruben intervals
- **2.8** Multivariate descriptive analytics and visualization

## Directory Structure

```text
L02_Multivariate_Analysis/
├── data/                                         # Datasets and generation scripts
│   ├── fetch_environmental_data.py               # Atmospheric sensor monitoring generator
│   ├── environmental_data.csv                    # Generated dataset (800 stations × 8 metrics)
│   └── ENVIRONMENTAL_DATA_DICTIONARY.md          # Detailed variable descriptions
├── notes/                                        # Lecture notes and study guides
│   └── multivariate_analysis_notes.typ           # Comprehensive study guide (Typst)
├── notebook/                                     # Interactive Jupyter notebooks
│   └── environmental_multivariate_analysis.ipynb # Air quality multivariate case study
├── presentation/                                 # Presentation slides
│   └── multivariate_analysis_slides.typ          # Typst presentation (Touying university theme)
└── README.md                                     # Module documentation
```

## Usage & Execution

### 1. Generate Dataset
```bash
uv run python lessons/L02_Multivariate_Analysis/data/fetch_environmental_data.py
```

### 2. Run Interactive Notebook
```bash
uv run jupyter notebook lessons/L02_Multivariate_Analysis/notebook/environmental_multivariate_analysis.ipynb
```

### 3. Compile Slides & Documents (Typst)
```bash
# Presentation
cd lessons/L02_Multivariate_Analysis/presentation/
typst compile multivariate_analysis_slides.typ

# Lecture Notes
cd ../notes/
typst compile multivariate_analysis_notes.typ
```
