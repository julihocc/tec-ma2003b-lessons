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

- **Elbow Method**: Suggests k = 4 (clear elbow at 4 clusters)
- **Silhouette Analysis**: k = 4 achieves highest average silhouette score (0.48)
- **Dendrogram**: Large fusion distance jump between 4 and 3 clusters
- **Consensus**: 4 distinct customer segments identified

### Discovered Customer Segments

**Segment 1 - Power Shoppers (22% of customers)**
- High purchase frequency and spending
- Long session durations
- High email engagement and product views
- Low return rates
- **Strategy**: Premium loyalty program, early access to products

**Segment 2 - Bargain Hunters (28% of customers)**
- Moderate purchase frequency
- Low average basket size but frequent purchases
- High product views (comparison shopping)
- Moderate email clicks
- **Strategy**: Promotional campaigns, bundle deals

**Segment 3 - Window Shoppers (35% of customers)**
- Low purchase frequency
- High session duration and product views
- Very low email engagement
- Moderate return rates
- **Strategy**: Cart abandonment emails, retargeting ads

**Segment 4 - Impulse Buyers (15% of customers)**
- Sporadic high-value purchases
- Short session durations
- Large basket sizes when they buy
- High email click rates
- **Strategy**: Limited-time offers, flash sales

### Clustering Performance

- **Ward's Hierarchical**: Clear, compact clusters
- **K-Means**: Comparable results with faster computation
- **Agreement**: 87% concordance between methods
- **Silhouette Score**: 0.48 (reasonable cluster structure)
- **Davies-Bouldin Index**: 0.82 (good separation and compactness)

### Business Insights

1. **Segment Size Distribution**: Largest segment (Window Shoppers) represents untapped potential
2. **Value Concentration**: Top 37% of customers (Power Shoppers + Impulse Buyers) drive 68% of revenue
3. **Engagement Opportunities**: Window Shoppers need conversion optimization
4. **Retention Focus**: Power Shoppers require white-glove treatment

## Files in This Directory

- `fetch_customer_data.py`: Data generation script
- `customer_clustering_analysis.ipynb`: Complete cluster analysis notebook
- `customer_data.csv`: Generated customer dataset (2,000 × 7)
- `CUSTOMER_DATA_DICTIONARY.md`: Detailed variable descriptions
- `dendrogram.png`: Hierarchical clustering visualization
- `elbow_plot.png`: K-means elbow method
- `silhouette_plot.png`: Silhouette analysis for multiple k
- `cluster_profiles.png`: Radar charts of segment characteristics
- `pca_clusters.png`: 2D PCA projection with cluster assignments

## Usage

```bash
# Generate the dataset
python fetch_customer_data.py

# Run the cluster analysis (Jupyter notebook)
jupyter notebook customer_clustering_analysis.ipynb
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
