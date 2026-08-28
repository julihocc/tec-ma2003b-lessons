# Chapter 6 — Analysis by Conglomerates (Cluster Analysis)

This chapter covers unsupervised statistical learning, proximity and distance metrics, hierarchical agglomerative clustering (AGNES), partitioning algorithms (K-Means), and cluster validation metrics.

## Chapter Overview (Syllabus Section 6)

Cluster Analysis partitions a set of unlabeled multivariate observations into natural, homogeneous subgroups (*clusters*) such that objects within the same cluster are highly similar while objects across different clusters are distinctly separated. This module explores both hierarchical linkage methods (Ward's, Complete, Single) and non-hierarchical methods (K-Means, K-Means++), supported by validation metrics (Silhouette, Elbow) and e-commerce behavioral segmentation case studies.

## Syllabus Topics & Subtopics

- **6.1** Similarity and dissimilarity measures
- **6.2** Graphical methods (dispersion, main components, Andrews)
- **6.3** Non-hierarchical grouping methods
- **6.4** Hierarchical grouping
- **6.5** Nearest neighbor method
- **6.6** Coding and commercial programs

## Directory Structure

```text
L06_Cluster_Analysis/
├── data/                                         # Datasets and generation scripts
│   ├── fetch_customer_data.py                    # Multi-channel shopper behavioral generator
│   ├── customer_data.csv                         # Unlabeled dataset (2,000 customers × 7 features)
│   ├── customer_data_with_labels.csv             # Ground-truth labeled benchmark dataset
│   ├── CUSTOMER_DATA_DICTIONARY.md               # Detailed variable descriptions
│   └── README.md                                 # Data documentation & regeneration guide
├── notebook/                                     # Interactive Jupyter notebooks
│   ├── customer_clustering_analysis.ipynb        # End-to-end unsupervised segmentation case study
│   └── README.md                                 # Notebook walkthrough & cell progression
├── notes/                                        # Lecture notes and study guides
│   ├── cluster_analysis_notes.typ                # Comprehensive study guide (Typst)
│   ├── cluster_analysis_notes.pdf                # Compiled lecture note PDF
│   └── README.md                                 # Notes outline & compilation guide
├── presentation/                                 # Presentation slides
│   ├── cluster_analysis_slides.typ               # Typst presentation (Touying university theme)
│   ├── cluster_analysis_slides.pdf               # Compiled presentation PDF
│   └── README.md                                 # Presentation agenda & compilation guide
└── README.md                                     # Module documentation
```

## Usage & Quickstart

### 1. Generate Dataset
```bash
uv run python lessons/L06_Cluster_Analysis/data/fetch_customer_data.py
```

### 2. Run Interactive Notebook
```bash
uv run jupyter notebook lessons/L06_Cluster_Analysis/notebook/customer_clustering_analysis.ipynb
```

### 3. Compile Slides & Documents (Typst)
```bash
# Presentation
cd lessons/L06_Cluster_Analysis/presentation/
typst compile cluster_analysis_slides.typ

# Lecture Notes
cd ../notes/
typst compile cluster_analysis_notes.typ
```
