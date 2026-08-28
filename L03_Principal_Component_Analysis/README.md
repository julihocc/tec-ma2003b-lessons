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

## Learning Objectives

- Formulate PCA via spectral decomposition of covariance ($\mathbf{S}$) and correlation ($\mathbf{R}$) matrices.
- Geometrically interpret major components as orthogonal axes of maximum variance.
- Estimate principal component loadings, scores, and variance explained proportions.
- Determine component retention using the Kaiser criterion, Scree plots, and parallel analysis.
- Implement PCA in Python (`scikit-learn`) and commercial software.

## Directory Structure

```text
L03_Principal_Component_Analysis/
├── data/           # Datasets and generation scripts
├── docs/           # Lecture notes and study guides
├── notebook/       # Interactive Jupyter notebooks
├── presentation/   # Presentation slides (Typst + Touying)
└── README.md       # Module documentation
```
