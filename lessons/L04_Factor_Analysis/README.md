# Chapter 4 — Factor Analysis

This chapter covers Exploratory Factor Analysis (EFA), the Common Factor Model, factor extraction algorithms, orthogonal and oblique rotations, and factor score estimation.

## Chapter Overview (Syllabus Section 4)

Unlike Principal Component Analysis (which reduces total observed variance), Factor Analysis postulates that observed correlations are driven by a smaller set of unobservable *latent constructs*. This module covers the mathematical framework, rotation algorithms, communality estimation, and pedagogical student assessment case studies.

## Syllabus Topics & Subtopics

- **4.1** Objectives of factor analysis
- **4.2** Factor analysis equations
- **4.3** Choice of the appropriate number of factors
- **4.4** Factor rotation
- **4.5** Oblique rotation method
- **4.6** Coding and commercial programs

## Directory Structure

```text
L04_Factor_Analysis/
├── data/                                         # Datasets and generation scripts
│   ├── fetch_educational.py                      # Student performance score generator
│   ├── educational.csv                           # Generated dataset (200 students × 9 metrics)
│   ├── EDUCATIONAL_ASSESSMENT_DATA_DICTIONARY.md # Detailed variable descriptions
│   └── README.md                                 # Data documentation & regeneration guide
├── notebook/                                     # Interactive Jupyter notebooks
│   ├── educational_analysis.ipynb                # Primary factor analysis case study
│   ├── snippets/                                 # 5 modular concept snippets + test harness
│   │   ├── 01_pca_basic_example.ipynb            # Snippet 1: Manual vs. sklearn PCA
│   │   ├── 02_component_retention.ipynb          # Snippet 2: Kaiser rule & Scree plots
│   │   ├── 03_factor_analysis_basic.ipynb        # Snippet 3: Single-factor model
│   │   ├── 04_factor_rotation.ipynb              # Snippet 4: Varimax vs. Promax rotation
│   │   ├── 05_complete_workflow.ipynb            # Snippet 5: End-to-end EFA workflow
│   │   ├── test_all_snippets.py                  # Automated non-interactive test runner
│   │   └── README.md                             # Snippets documentation
│   └── README.md                                 # Notebooks documentation
├── notes/                                        # Lecture notes and study guides
│   ├── factor_analysis_notes.typ                 # Comprehensive study guide (Typst)
│   ├── factor_analysis_notes.pdf                 # Compiled lecture note PDF
│   ├── executive_report.typ                      # Executive stakeholder case report (Typst)
│   ├── executive_report.pdf                      # Compiled executive report PDF
│   └── README.md                                 # Notes outline & compilation guide
├── presentation/                                 # Presentation slides
│   ├── factor_analysis_slides.typ                # Touying university-theme presentation
│   ├── factor_analysis_slides.pdf                # Compiled presentation PDF
│   └── README.md                                 # Presentation agenda & compilation guide
└── README.md                                     # Module documentation
```

## Usage & Quickstart

### 1. Generate Dataset
```bash
uv run python lessons/L04_Factor_Analysis/data/fetch_educational.py
```

### 2. Run Interactive Notebook
```bash
uv run jupyter notebook lessons/L04_Factor_Analysis/notebook/educational_analysis.ipynb
```

### 3. Run Factor Analysis Snippet Test Suite
```bash
uv run python lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py
```

### 4. Compile Slides & Documents (Typst)
```bash
# Presentation
cd lessons/L04_Factor_Analysis/presentation/
typst compile factor_analysis_slides.typ

# Lecture Notes
cd ../notes/
typst compile factor_analysis_notes.typ
```
