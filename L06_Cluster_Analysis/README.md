# E-Commerce Customer Clustering Analysis

This example demonstrates cluster analysis for discovering natural customer segments in e-commerce behavioral data. Unlike supervised segmentation, this analysis uses unsupervised learning to reveal hidden patterns without predefined categories.

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

## Analysis Approach

### Hierarchical Clustering

- Build dendrogram to visualize cluster structure
- Test multiple linkage methods (Ward's, Complete, Average)
- Identify optimal number of clusters from dendrogram
- Understand hierarchical relationships between segments

### K-Means Clustering

- Determine optimal k using elbow method and silhouette analysis
- Compare multiple initializations for stability
- Fast clustering for large-scale customer base
- Validate cluster assignments

### Comparison and Validation

- Compare hierarchical vs. k-means results
- Use silhouette scores to assess cluster quality
- Validate stability using bootstrap samples
- Profile each discovered segment

## Key Results

### Optimal Number of Clusters

- **Elbow Method**: Suggests k = 4 (diminishing returns after k=4)
- **Silhouette Analysis**: k = 4 achieves highest average silhouette score (0.458)
- **Dendrogram (Ward's)**: Large vertical gap suggests 4 clusters (cut at distance ≈50)
- **Consensus**: Both hierarchical and k-means converge on 4 distinct segments

### Discovered Customer Segments

#### Cluster 0 - Engaged but Selective Shoppers (15.3% - 307 customers)

- **Distinctive Highs**: avg_basket_size (+72%), return_rate (+56%), email_clicks (+52%)
- **Distinctive Lows**: session_duration (-70%), product_views (-58%), monthly_purchases (-42%)
- **Profile**: High basket sizes with high returns, engaged with emails but low browsing
- **Strategy**: VIP loyalty program, free expedited shipping, improve product information to reduce returns, personalized recommendations

#### Cluster 1 - Low-Value Browsers (35.4% - 707 customers)

- **Distinctive Highs**: session_duration (+19%), return_rate (+15%)
- **Distinctive Lows**: total_spend (-84%), email_clicks (-78%), monthly_purchases (-76%)
- **Profile**: Spend time browsing but make very few purchases
- **Strategy**: Abandoned cart recovery, retargeting ads, limited-time discounts, improve product information and reviews

#### Cluster 2 - Premium High-Value Customers (21.6% - 432 customers)

- **Distinctive Highs**: total_spend (+168%), avg_basket_size (+128%), monthly_purchases (+124%)
- **Distinctive Lows**: return_rate (-24%)
- **Profile**: Highest spenders with frequent large purchases and low returns
- **Strategy**: Exclusive VIP treatment, premium customer service, early access to collections, referral incentives, focus on retention

#### Cluster 3 - Frequent Small-Basket Shoppers (27.7% - 554 customers)

- **Distinctive Highs**: monthly_purchases (+24%), product_views (+12%)
- **Distinctive Lows**: avg_basket_size (-49%), total_spend (-45%), return_rate (-32%)
- **Profile**: Regular purchases with lower average order values
- **Strategy**: Targeted discount codes, free shipping thresholds, value packs, multi-buy promotions

### Clustering Performance

- **Ward's Hierarchical Silhouette**: 0.458 (balanced clusters, hierarchical structure)
- **K-Means Silhouette**: 0.458 (identical to hierarchical - strong validation)
- **Method Convergence**: Both methods produce similar cluster sizes and quality
- **Cluster Quality**: Above 0.4 is acceptable for customer segmentation where boundaries are naturally fuzzy
- **PCA Visualization**: 2D projection shows reasonable cluster separation

### Business Insights

1. **Method Validation**: Convergence of hierarchical and k-means on same structure provides strong evidence of genuine segments
2. **Value Concentration**: Premium segment (21.6%) drives disproportionate revenue
3. **Conversion Opportunity**: Low-value browsers (35.4%) represent largest untapped potential
4. **Engagement Diversity**: Each segment requires distinct marketing approach
5. **Actionable Profiles**: All segments differ by >10% on multiple dimensions, enabling targeted strategies

## Presentation Structure

The presentation (`cluster_analysis_complete.typ`) integrates theoretical concepts with the practical customer segmentation case study:

**Part 1: Theoretical Foundations** - Core clustering concepts with case study references:

- Introduction to cluster analysis (with e-commerce segmentation preview)
- Distance and similarity measures (with customer data standardization example)
- Hierarchical clustering methods (with dendrogram interpretation example showing 4 segments)
- K-means and non-hierarchical methods (with elbow method results)
- Determining optimal clusters (with convergence of methods at k=4)
- Validation techniques (with silhouette analysis showing 0.458 score)
- Practical considerations and best practices

**Part 2: Practical Application** - Complete customer segmentation workflow:

- Business context and dataset overview
- Exploratory data analysis with correlations
- Data standardization (critical for mixed-scale variables)
- Hierarchical clustering with dendrogram analysis
- K-means with elbow method determination
- Cluster interpretation and profiling (4 distinct segments)
- Silhouette validation and quality assessment
- PCA visualization in 2D space
- Business recommendations for each segment

The presentation serves as a companion to the Jupyter notebook, showing how theoretical concepts apply to the real-world segmentation problem throughout.

## Directory Structure

```text
L06_Cluster_Analysis/
├── data/                           # Data files and generation scripts
│   ├── fetch_customer_data.py      # Data generation script
│   ├── customer_data.csv           # Generated customer dataset (2,000 × 7)
│   ├── customer_data_with_labels.csv  # Dataset with cluster labels
│   └── CUSTOMER_DATA_DICTIONARY.md # Detailed variable descriptions
├── notebook/                       # Analysis notebooks
│   └── customer_clustering_analysis.ipynb  # Complete cluster analysis
├── docs/                           # Lecture notes and documentation
│   ├── cluster_analysis_notes.typ  # Typst source for lecture notes
│   └── cluster_analysis_notes.pdf  # Compiled lecture notes
├── presentation/                   # Presentation materials
│   ├── cluster_analysis_complete.typ  # Typst source for presentation
│   └── cluster_analysis_complete.pdf  # Compiled presentation slides
└── README.md                       # This file
```

## Usage

```bash
# Generate the dataset
python data/fetch_customer_data.py

# Run the cluster analysis (Jupyter notebook)
jupyter notebook notebook/customer_clustering_analysis.ipynb
```

## Educational Value

This example illustrates:

- **Unsupervised Learning**: Discovering patterns without training labels
- **Method Comparison**: Hierarchical vs. k-means clustering
- **Optimal k Selection**: Using elbow, silhouette, and dendrogram methods
- **Validation**: Assessing cluster quality and stability
- **Business Translation**: Converting statistical clusters to actionable segments
- **Visualization**: Dendrograms, PCA plots, and radar charts
- **Interpretation**: Profiling and naming discovered segments

## Extensions

Students can extend this analysis by:

- Adding temporal features (recency, time since last purchase)
- Incorporating demographic data (age, location)
- Testing density-based clustering (DBSCAN) for outlier detection
- Implementing fuzzy clustering for soft segment membership
- Building predictive models based on discovered segments
- Developing CLV (Customer Lifetime Value) models per segment
- Creating dynamic segmentation that evolves over time

## Comparison with Discriminant Analysis

| Aspect | Cluster Analysis (This Example) | Discriminant Analysis (L05) |
|--------|----------------------------------|------------------------------|
| **Learning Type** | Unsupervised | Supervised |
| **Input** | Features only | Features + known labels |
| **Goal** | Discover segments | Classify to known segments |
| **Output** | Segment assignments + profiles | Classification rules |
| **When to Use** | Exploratory analysis | Predictive classification |
| **Validation** | Internal metrics (silhouette) | External metrics (accuracy) |

## Real-World Considerations

### Data Preprocessing
- Standardization is critical (variables on different scales)
- Outlier handling affects cluster centroids
- Missing value imputation may be necessary

### Business Implementation
- Segment profiles must be actionable
- Consider operational constraints (campaign capacity)
- Monitor segment drift over time
- Balance personalization with privacy

### Model Maintenance
- Re-cluster periodically as behavior evolves
- Track segment transitions (customers moving between segments)
- Validate business impact of segmentation strategy
- Refine based on A/B testing results
