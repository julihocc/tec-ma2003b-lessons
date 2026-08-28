# Chapter 7 — Multivariate Regression & Modeling

This chapter demonstrates multivariate regression methods for analyzing relationships between lifestyle factors, physiological measurements, and cardiovascular disease (CVD) risk. The analysis showcases logistic regression, mean vector testing, MANOVA, and canonical correlation analysis.

## Business Context

A healthcare research institute wants to understand the complex relationships between patient lifestyle behaviors, physiological health markers, and cardiovascular disease risk. The goals are to:

- **Predict CVD Risk**: Identify patients at high risk based on lifestyle and health markers
- **Evaluate Interventions**: Test whether a lifestyle intervention program improves health outcomes
- **Understand Relationships**: Explore how lifestyle factors relate to physiological measurements
- **Develop Prevention Strategies**: Create targeted interventions based on identified risk factors

## Dataset Description

The synthetic dataset contains 1,000 patients with demographics, lifestyle factors, and physiological measurements across 14 variables:

- **Demographics**: `patient_id`, `age`, `bmi`
- **Lifestyle Factors**: `exercise_hours_week`, `smoking_years`, `alcohol_units_week`, `stress_score`, `sleep_hours`
- **Physiological Markers**: `systolic_bp`, `diastolic_bp`, `cholesterol`, `glucose`, `triglycerides`, `hdl`
- **Outcomes**: `cvd_risk_high` (0/1), `treatment_group` (Control / Intervention)

## Directory Structure

```text
L07_Multivariate_Regression/
├── data/                                         # Data files and generation scripts
│   ├── fetch_health_data.py                      # Data generation script
│   ├── health_data.csv                           # Generated patient dataset (1,000 × 14)
│   ├── HEALTH_DATA_DICTIONARY.md                 # Detailed variable descriptions
│   └── HEALTH_DATA_DICTIONARY.pdf                 # Reference data dictionary PDF
├── notes/                                        # Lecture notes and documentation
│   ├── multivariate_regression_notes.typ         # Typst source for lecture notes
│   └── multivariate_regression_notes.pdf         # Compiled lecture notes
├── notebook/                                     # Analysis notebooks
│   └── health_risk_analysis.ipynb                # Complete multivariate regression analysis
├── presentation/                                 # Presentation materials
│   ├── multivariate_regression_slides.typ        # Typst source for presentation (Touying)
│   └── multivariate_regression_slides.pdf        # Compiled presentation slides
└── README.md                                     # Module documentation
```

## Usage & Execution

### 1. Generate Dataset
```bash
python data/fetch_health_data.py
```

### 2. Run Interactive Notebook
```bash
jupyter notebook notebook/health_risk_analysis.ipynb
```

### 3. Compile Slides & Documents (Typst)
```bash
# Presentation
cd presentation/
typst compile multivariate_regression_slides.typ

# Lecture Notes
cd ../notes/
typst compile multivariate_regression_notes.typ
```
