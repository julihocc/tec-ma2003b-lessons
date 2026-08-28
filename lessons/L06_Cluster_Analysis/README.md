# Chapter 6 — Cluster Analysis

This chapter covers Cluster Analysis techniques for discovering natural customer segments in e-commerce behavioral data. Unlike supervised segmentation, this analysis uses unsupervised learning to reveal hidden patterns without predefined categories.

## Business Context

An e-commerce company wants to understand their customer base better to optimize marketing strategies, personalize experiences, and allocate resources efficiently. Without predetermined customer categories, the company uses cluster analysis to:

- **Discover Natural Segments**: Identify groups based on behavioral patterns
- **Personalize Marketing**: Tailor campaigns to each discovered segment
- **Optimize Resources**: Focus efforts on high-potential customer groups
- **Improve Retention**: Develop segment-specific retention strategies

## Dataset Description

The synthetic dataset contains 2,000 customers with 7 behavioral metrics:

- **monthly_purchases**: Average purchases per month
- **avg_basket_size**: Average number of items per transaction
- **total_spend**: Total spending over observation period (dollars)
- **session_duration**: Average time spent per website visit (minutes)
- **email_clicks**: Average email marketing clicks per month
- **product_views**: Average product pages viewed per session
- **return_rate**: Proportion of purchased items returned

## Directory Structure

```text
L06_Cluster_Analysis/
├── data/                               # Data files and generation scripts
│   ├── fetch_customer_data.py          # Data generation script
│   ├── customer_data.csv               # Generated customer dataset (2,000 × 7)
│   ├── customer_data_with_labels.csv   # Dataset with cluster labels
│   └── CUSTOMER_DATA_DICTIONARY.md     # Detailed variable descriptions
├── notes/                              # Lecture notes and documentation
│   ├── cluster_analysis_notes.typ      # Typst source for lecture notes
│   └── cluster_analysis_notes.pdf      # Compiled lecture notes
├── notebook/                           # Analysis notebooks
│   └── customer_clustering_analysis.ipynb # Complete cluster analysis
├── presentation/                       # Presentation materials
│   ├── cluster_analysis_slides.typ     # Typst source for presentation (Touying)
│   └── cluster_analysis_slides.pdf     # Compiled presentation slides
└── README.md                           # Module documentation
```

## Usage & Execution

### 1. Generate Dataset
```bash
python data/fetch_customer_data.py
```

### 2. Run Interactive Notebook
```bash
jupyter notebook notebook/customer_clustering_analysis.ipynb
```

### 3. Compile Slides & Documents (Typst)
```bash
# Presentation
cd presentation/
typst compile cluster_analysis_slides.typ

# Lecture Notes
cd ../notes/
typst compile cluster_analysis_notes.typ
```
