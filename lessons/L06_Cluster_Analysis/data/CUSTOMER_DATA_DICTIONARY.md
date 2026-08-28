# E-Commerce Customer Data Dictionary

## Overview

This data dictionary describes the synthetic e-commerce customer behavioral dataset used for cluster analysis. The dataset contains 2,000 customer records generated to reflect realistic online shopping patterns across diverse customer types.

## Dataset Structure

- **File**: `customer_data.csv`
- **Rows**: 2,000 (customers)
- **Columns**: 7 (all behavioral features - no predefined target)
- **Data Type**: CSV with header row

## Variables

### Feature Variables (All Continuous)

| Variable | Type | Description | Range | Units | Precision |
|----------|------|-------------|-------|-------|-----------|
| `monthly_purchases` | Numeric | Average number of purchases per month | 0.2 - 15.0 | purchases/month | 1 decimal |
| `avg_basket_size` | Numeric | Average number of items per transaction | 1.0 - 25.0 | items/transaction | 1 decimal |
| `total_spend` | Numeric | Total spending over 12-month observation period | $50.0 - $8,000.0 | USD | 2 decimals |
| `session_duration` | Numeric | Average time spent per website visit | 2.0 - 60.0 | minutes | 1 decimal |
| `email_clicks` | Numeric | Average email marketing clicks per month | 0.0 - 12.0 | clicks/month | 1 decimal |
| `product_views` | Numeric | Average product pages viewed per session | 3.0 - 50.0 | pages/session | 1 decimal |
| `return_rate` | Numeric | Proportion of purchased items returned | 0.0 - 0.50 | rate (0-1) | 3 decimals |

## Expected Customer Segments (Post-Clustering)

Note: These segments are discovered through unsupervised learning, not predefined.

### Power Shoppers (~22% expected)

High-value customers who drive significant revenue.

**Behavioral Patterns**:
- High monthly purchase frequency (8-15 purchases/month)
- Large basket sizes (15-25 items)
- High total spending ($4,000-$8,000)
- Long sessions (40-60 minutes)
- High email engagement (8-12 clicks/month)
- High product exploration (30-50 views/session)
- Low return rates (0.05-0.15)

**Business Value**: Premium customers requiring retention focus

### Bargain Hunters (~28% expected)

Price-sensitive customers with frequent but low-value transactions.

**Behavioral Patterns**:
- Moderate purchase frequency (5-8 purchases/month)
- Small basket sizes (3-6 items)
- Moderate spending ($800-$1,500)
- Moderate sessions (20-30 minutes)
- Moderate email clicks (4-6 clicks/month)
- High product views (25-40 views/session) - comparison shopping
- Low return rates (0.08-0.18)

**Business Value**: Volume segment sensitive to promotions

### Window Shoppers (~35% expected)

Browsers with low conversion but high engagement potential.

**Behavioral Patterns**:
- Low purchase frequency (0.5-2 purchases/month)
- Small basket sizes (1-4 items)
- Low spending ($100-$600)
- Long sessions (30-50 minutes) - browsing without buying
- Very low email clicks (0-2 clicks/month)
- Moderate-high product views (20-35 views/session)
- Moderate return rates (0.15-0.25)

**Business Value**: Conversion optimization opportunity

### Impulse Buyers (~15% expected)

Sporadic high-value purchasers driven by promotions.

**Behavioral Patterns**:
- Low-moderate frequency (2-4 purchases/month)
- Large basket sizes (10-20 items) when they buy
- High-moderate spending ($2,000-$4,000)
- Short sessions (5-15 minutes) - quick decisions
- High email clicks (6-10 clicks/month) - promotion-driven
- Low product views (8-15 views/session) - minimal browsing
- High return rates (0.20-0.35) - impulse regret

**Business Value**: Flash sale and limited-time offer segment

## Data Generation Methodology

### Statistical Approach

- **Multivariate Normal Distributions**: Four distinct customer types generated from separate distributions
- **Realistic Correlations**: Variables exhibit plausible relationships:
  - High spending correlates with purchase frequency
  - Session duration correlates with product views
  - Email clicks correlate with purchase frequency
  - Return rate inversely correlates with session duration (deliberate purchases)

### Key Parameters

#### Power Shoppers Cluster
- **Sample Size**: ~440 customers (22%)
- **Mean Vector**: [12.0, 20.0, 6000.0, 50.0, 10.0, 40.0, 0.10]
- **Covariance**: Moderate variance, strong positive correlations between engagement metrics

#### Bargain Hunters Cluster
- **Sample Size**: ~560 customers (28%)
- **Mean Vector**: [6.5, 4.5, 1200.0, 25.0, 5.0, 32.0, 0.13]
- **Covariance**: Low variance in basket size, moderate in frequency

#### Window Shoppers Cluster
- **Sample Size**: ~700 customers (35%)
- **Mean Vector**: [1.2, 2.5, 350.0, 40.0, 1.0, 27.0, 0.20]
- **Covariance**: High variance in session duration, low in purchases

#### Impulse Buyers Cluster
- **Sample Size**: ~300 customers (15%)
- **Mean Vector**: [3.0, 15.0, 3000.0, 10.0, 8.0, 12.0, 0.28]
- **Covariance**: High variance in spending, moderate in frequency

## Data Quality Notes

### Bounds and Constraints

- All frequency and count variables constrained to realistic positive values
- Return rate constrained to [0, 0.50] (50% maximum return rate)
- Session duration bounded by realistic browsing times
- Spending reflects typical e-commerce ranges across segments

### Precision and Realism

- **Currency values**: Rounded to 2 decimals (e.g., $1,249.99)
- **Rates**: Rounded to 3 decimals (e.g., 0.145 = 14.5% return rate)
- **Frequencies/Counts**: Rounded to 1 decimal (e.g., 6.3 purchases/month)
- **Time measurements**: Rounded to 1 decimal (e.g., 28.5 minutes/session)

### Random Seed

- **Seed**: 42 (for reproducibility)
- Ensures consistent cluster structure across runs

## Clustering Analysis Setup

### Preprocessing Requirements

1. **Standardization**: All features MUST be standardized (z-scores)
   - Variables have vastly different scales (dollars vs. counts)
   - Unstandardized data would be dominated by `total_spend`

2. **Outlier Check**: Review boxplots before clustering
   - Some genuine outliers may exist (VIP customers)
   - Decision: Keep or remove based on business context

3. **Missing Values**: None in this synthetic dataset
   - Real data may require imputation

### Expected Clustering Results

#### Hierarchical Clustering
- **Ward's Method**: Clear dendrogram structure with 4 major branches
- **Cophenetic Correlation**: ~0.75-0.85 (good preservation of distances)
- **Optimal k**: 4 clusters (large fusion distance jump)

#### K-Means Clustering
- **Optimal k (Elbow)**: k = 4 (clear elbow point)
- **Optimal k (Silhouette)**: k = 4 (silhouette score ~0.45-0.50)
- **Convergence**: Typically 15-25 iterations
- **Stability**: High agreement (>85%) across multiple initializations

#### Validation Metrics
- **Average Silhouette Width**: 0.45-0.52 (reasonable structure)
- **Davies-Bouldin Index**: 0.75-0.90 (lower is better)
- **Calinski-Harabasz Score**: High (compact, well-separated clusters)

## Variable Importance for Clustering

### Primary Discriminators

1. **total_spend** - Strongest separator between high/low value customers
2. **monthly_purchases** - Distinguishes frequency patterns
3. **session_duration** - Separates browsers from quick buyers
4. **avg_basket_size** - Identifies bulk vs. single-item purchasers

### Secondary Discriminators

5. **email_clicks** - Indicates engagement level
6. **product_views** - Reveals exploration vs. directed shopping
7. **return_rate** - Reflects purchase deliberation vs. impulse

## Educational Applications

### Learning Objectives

1. **Standardization**: Understanding why and how to standardize
2. **Method Selection**: When to use hierarchical vs. k-means
3. **Optimal k Determination**: Using multiple validation criteria
4. **Cluster Interpretation**: Translating statistics to business insights
5. **Validation**: Assessing cluster quality and stability

### Common Analysis Questions

- How many natural customer segments exist?
- What behavioral patterns characterize each segment?
- Which clustering method performs better for this data?
- How stable are the cluster assignments?
- What marketing strategies fit each discovered segment?

## Business Applications

### Marketing Strategies by Segment

**Power Shoppers**:
- VIP loyalty program with exclusive perks
- Early access to new products
- Personalized product recommendations
- Dedicated customer service

**Bargain Hunters**:
- Promotional email campaigns
- Bundle deals and volume discounts
- Clearance sale notifications
- Price-drop alerts

**Window Shoppers**:
- Cart abandonment recovery emails
- Retargeting display ads
- Free shipping incentives
- Product comparison tools

**Impulse Buyers**:
- Flash sales and limited-time offers
- Urgency-based messaging ("Only 3 left!")
- Email campaigns during peak buying times
- Easy one-click checkout

### ROI Estimation

- **Power Shoppers**: High retention focus → 2x lifetime value
- **Bargain Hunters**: Promotion optimization → 15% revenue increase
- **Window Shoppers**: Conversion improvement → 5% conversion rate boost
- **Impulse Buyers**: Flash sale targeting → 25% higher response rate

## Extensions and Modifications

### Additional Features to Consider

- **Recency**: Days since last purchase (RFM analysis)
- **Device Type**: Mobile vs. desktop behavior
- **Category Preferences**: Fashion vs. electronics vs. home goods
- **Geographic Location**: Regional behavior patterns
- **Time of Day**: Shopping time preferences
- **Coupon Usage**: Price sensitivity indicators

### Alternative Clustering Approaches

- **DBSCAN**: Identify outliers and irregular-shaped clusters
- **Gaussian Mixture Models**: Soft clustering with probability distributions
- **Hierarchical + K-Means**: Use dendrogram to determine k, then k-means for final clusters
- **Fuzzy C-Means**: Allow customers to belong to multiple segments

### Temporal Analysis

- **Segment Evolution**: Track how customers move between segments
- **Seasonal Patterns**: Adjust segments for holiday shopping behavior
- **Cohort Analysis**: Compare behavior across customer acquisition periods

---

**Note**: This synthetic dataset is designed for educational purposes and reflects generalized patterns in e-commerce customer behavior. Actual patterns may vary significantly based on industry, product category, market maturity, and business model.
