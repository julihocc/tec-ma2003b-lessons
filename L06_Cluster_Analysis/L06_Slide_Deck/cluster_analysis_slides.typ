// Cluster Analysis Presentation using Touying
#import "@preview/touying:0.5.3": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Cluster Analysis],
    subtitle: [Discovering Natural Groupings in Data],
    author: [Juliho Castillo Colmenares],
    date: none,
    institution: [Tec de Monterrey],
    logo: none,
  ),
  config-colors(
    primary: rgb("#003366"),
    secondary: rgb("#1E88E5"),
    tertiary: rgb("#666666"),
  ),
  config-common(
    slide-fn: slide => {
      align(horizon, slide)
    },
  ),
)

#title-slide()

// Outline
#slide[
  = Today's Agenda

  + Introduction to Cluster Analysis
  + Distance and Similarity Measures
  + Hierarchical Clustering Methods
  + K-Means and Non-Hierarchical Methods
  + Determining Optimal Number of Clusters
  + Validation Techniques
  + Practical Considerations
  + Applications and Best Practices
]

// Section 1: Introduction
#slide[
  = What is Cluster Analysis?

  *Definition:* An exploratory technique to discover natural groupings in data *without predefined categories*
]

#slide[
  = What is Cluster Analysis?

  *Key Characteristics:*
  - Unsupervised learning method
  - No training labels required
  - Discovers hidden structure in data
  - Groups similar observations together

  #alert[
    *Goal:* Maximize within-cluster similarity and between-cluster dissimilarity
  ]
]

#slide[
  = Cluster Analysis vs. Discriminant Analysis

  #table(
    columns: (1fr, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 10pt,
    [*Cluster Analysis*], [*Discriminant Analysis*],
    [Unsupervised learning], [Supervised learning],
    [Discovers unknown groups], [Classifies into known groups],
    [No training labels], [Requires training labels],
    [Exploratory], [Predictive],
    [Groups observations], [Creates decision boundaries],
  )
]

#slide[
  = Applications: Marketing & Business

  *Marketing*
  - Customer segmentation for targeted campaigns
  - Market basket analysis

  *Business*
  - Fraud detection
  - Anomaly identification
]

#slide[
  = Applications: Science & Healthcare

  *Biology & Medicine*
  - Disease subtype identification
  - Gene expression analysis

  *Social Sciences*
  - Community detection in networks
  - Document clustering
]

// Section 2: Distance Measures
#slide[
  = Distance and Similarity Measures

  *Why Distance Matters*

  Clustering depends on measuring how "close" observations are to each other
]

#slide[
  = Common Distance Metrics

  + *Euclidean Distance* (L2 norm) - Most common
  + *Manhattan Distance* (L1 norm) - Robust to outliers
  + *Cosine Similarity* - For high-dimensional data
  + *Correlation Distance* - Pattern similarity
]

#slide[
  = Euclidean Distance

  *Formula:*

  $ d(x, y) = sqrt(sum_(i=1)^p (x_i - y_i)^2) $
]

#slide[
  = Euclidean Distance

  *Properties:*
  - Straight-line distance in n-dimensional space
  - Sensitive to scale differences
  - Assumes equal importance of all dimensions

  #alert[
    *Warning:* Always standardize variables with different scales!
  ]
]

#slide[
  = Manhattan Distance

  *Formula:*

  $ d(x, y) = sum_(i=1)^p |x_i - y_i| $
]

#slide[
  = Manhattan Distance

  *When to Use:*
  - Data contains outliers or extreme values
  - Variables represent counts
  - High-dimensional spaces

  *Advantage:* More robust than Euclidean distance
]

#slide[
  = Why Standardization is Critical

  *Problem:* Variables on different scales dominate distance calculations

  *Example:*
  - Age: 20-80 years
  - Income: 20,000-200,000 dollars

  Without standardization, income dominates!
]

#slide[
  = Z-score Standardization

  *Solution: Z-score Standardization*

  $ z_i = (x_i - mu) / sigma $

  Transform to mean = 0, standard deviation = 1
]

// Section 3: Hierarchical Clustering
#slide[
  = Hierarchical Clustering

  Builds a tree-like structure (dendrogram) showing nested clusters
]

#slide[
  = Two Approaches

  *Agglomerative (Bottom-Up):* Most common
  - Start: Each observation is its own cluster
  - Process: Merge closest clusters iteratively
  - End: All observations in one cluster
]

#slide[
  = Two Approaches

  *Divisive (Top-Down):* Less common
  - Start: All observations in one cluster
  - Process: Split most heterogeneous cluster
  - End: Each observation is its own cluster
]

#slide[
  = Linkage Methods

  How to Measure Distance Between Clusters?
]

#slide[
  = Single Linkage (Nearest Neighbor)

  $ d(C_1, C_2) = min_(x in C_1, y in C_2) d(x, y) $

  Distance between closest points in the two clusters
]

#slide[
  = Complete Linkage (Farthest Neighbor)

  $ d(C_1, C_2) = max_(x in C_1, y in C_2) d(x, y) $

  Distance between farthest points in the two clusters
]

#slide[
  = Average Linkage

  $ d(C_1, C_2) = 1/(n_1 n_2) sum_(x in C_1) sum_(y in C_2) d(x, y) $

  Average distance between all pairs of points
]

#slide[
  = Ward's Method

  *Ward's Method*

  Minimizes within-cluster sum of squares

  Tends to produce compact, equal-sized clusters
]

#slide[
  = Linkage Methods Comparison

  #table(
    columns: (1.2fr, 1fr, 1.5fr),
    align: (left, center, left),
    stroke: 0.5pt,
    inset: 8pt,
    [*Method*], [*Outlier Sensitivity*], [*Cluster Shape*],
    [Single Linkage], [High], [Elongated (chaining)],
    [Complete Linkage], [Low], [Compact, spherical],
    [Average Linkage], [Medium], [Balanced],
    [Ward's Method], [Medium], [Compact, equal-sized],
  )
]

#slide[
  = Linkage Methods Comparison

  #alert[
    *Recommendation:* Ward's method often works best in practice
  ]
]

#slide[
  = Dendrograms

  Visualizing Hierarchical Structure
]

#slide[
  = Reading a Dendrogram

  - Horizontal axis: Observations or clusters
  - Vertical axis: Distance at which clusters merge
  - Height of branches: Dissimilarity between merged clusters
]

#slide[
  = Determining Number of Clusters

  - Look for large vertical gaps (jumps in fusion distance)
  - Cut dendrogram where there's substantial increase
  - Draw horizontal line: number of vertical lines crossed = k clusters
]

#slide[
  = The Chaining Effect

  *Problem with Single Linkage:*

  Clusters form long, elongated chains rather than compact groups
]

#slide[
  = The Chaining Effect

  *Why it Happens:*
  - Observations connect via intermediate points
  - A-B-C-D form chain where each is close to neighbor
  - But A and D are far apart
]

#slide[
  = The Chaining Effect

  *Solution:*
  - Use complete or average linkage instead
  - Or Ward's method for compact clusters
]

// Section 4: K-Means
#slide[
  = K-Means Clustering

  Most popular non-hierarchical method
]

#slide[
  = K-Means Algorithm

  + *Initialize:* Select k random observations as centroids
  + *Assignment:* Assign each point to nearest centroid
  + *Update:* Recalculate centroids as cluster means
  + *Repeat:* Steps 2-3 until convergence

  *Convergence:* When assignments no longer change between iterations
]

#slide[
  = K-Means Objective Function

  *Goal:* Minimize within-cluster sum of squares (WCSS)

  $ min sum_(i=1)^k sum_(x in C_i) ||x - mu_i||^2 $

  where $mu_i$ is the centroid of cluster $C_i$
]

#slide[
  = K-Means Properties

  - Always converges (finite partitions, monotonically decreasing WCSS)
  - Typically converges in 10-30 iterations
  - Fast: O(n times k times p times iterations)
]

#slide[
  = K-Means: Advantages

  - Fast and scalable to large datasets
  - Simple to understand and implement
  - Efficient for exploratory analysis
]

#slide[
  = K-Means: Limitations

  - Requires specifying k in advance
  - Sensitive to initialization (different starts → different results)
  - Assumes spherical clusters
  - Sensitive to outliers
  - Tends to create equal-sized clusters
]

#slide[
  = K-Means++ Initialization

  *Problem:* Random initialization can lead to poor results
]

#slide[
  = K-Means++ Algorithm

  + Choose first centroid randomly
  + For each subsequent centroid:
     - Choose point with probability proportional to squared distance from nearest existing centroid
  + Repeat until k centroids selected

  *Benefit:* Spreads out initial centroids, significantly improves results
]

#slide[
  = K-Medoids (PAM)

  *Key Difference from K-Means:*
  - K-means: Centers are computed means (may not be actual points)
  - K-medoids: Centers are actual data points (medoids)
]

#slide[
  = K-Medoids (PAM)

  *Advantages:*
  - More robust to outliers
  - Works with any distance metric
  - Interpretable centers (actual observations)

  *Disadvantage:* Slower than k-means (higher computational cost)
]

// Section 5: Optimal k
#slide[
  = How Many Clusters?

  *The Fundamental Challenge:*

  No "ground truth" for correct number of clusters
]

#slide[
  = Multiple Approaches

  + *Elbow Method* - Look for bend in WCSS plot
  + *Silhouette Analysis* - Measure cluster quality
  + *Gap Statistic* - Compare to null reference
  + *Davies-Bouldin Index* - Ratio of compactness to separation
  + *Domain Knowledge* - Business requirements
]

#slide[
  = Elbow Method: Procedure

  + Run clustering for k = 1, 2, 3, ..., K_max
  + Calculate WCSS for each k
  + Plot WCSS vs. k
  + Look for "elbow" - diminishing returns point
]

#slide[
  = Elbow Method: Interpretation

  - WCSS always decreases as k increases
  - Elbow indicates where additional clusters don't help much
  - Choose k at the elbow point

  #alert[
    *Limitation:* Elbow not always clear - may need other methods
  ]
]

#slide[
  = Silhouette Analysis

  Measures how well each point fits within its cluster
]

#slide[
  = Silhouette Coefficient

  *Silhouette Coefficient for observation i:*

  $ s(i) = (b(i) - a(i)) / max(a(i), b(i)) $

  where:
  - $a(i)$ = avg distance to points in same cluster
  - $b(i)$ = avg distance to points in nearest neighboring cluster
]

#slide[
  = Silhouette Interpretation

  - $s(i) approx +1$: Well-matched to cluster
  - $s(i) approx 0$: On border between clusters
  - $s(i) approx -1$: Likely in wrong cluster
]

#slide[
  = Using Silhouette for Optimal k

  *Average Silhouette Width:*

  $ macron(s) = 1/n sum_(i=1)^n s(i) $
]

#slide[
  = Silhouette Procedure

  + Run clustering for different k values
  + Calculate average silhouette width for each k
  + Choose k that maximizes $macron(s)$

  *Advantage:* Provides both quality measure and optimal k
]

// Section 6: Validation
#slide[
  = Cluster Validation

  *Internal Validation* (using data only):
  - Within-Cluster Sum of Squares (WCSS) - lower is better
  - Silhouette Coefficient - higher is better
  - Davies-Bouldin Index - lower is better
  - Dunn Index - higher is better
]

#slide[
  = Cluster Validation

  *External Validation* (when true labels available):
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
]

#slide[
  = Davies-Bouldin Index

  Measures ratio of within-cluster dispersion to between-cluster separation

  $ "DB" = 1/k sum_(i=1)^k max_(j != i) ((sigma_i + sigma_j) / d(c_i, c_j)) $
]

#slide[
  = Davies-Bouldin Index

  *Interpretation:*
  - Lower values indicate better clustering
  - Compact clusters that are far apart
  - Can compare different k values or methods
]

// Section 7: Practical Considerations
#slide[
  = Curse of Dimensionality

  As dimensions (p) increase, problems arise
]

#slide[
  = Curse of Dimensionality: Problems

  + Distance becomes less meaningful (all points appear equidistant)
  + Data becomes sparse (observations spread out)
  + Computational cost increases dramatically
]

#slide[
  = Curse of Dimensionality: Solutions

  - Use PCA or feature selection before clustering
  - Select only relevant variables
  - Use specialized high-dimensional algorithms

  #alert[
    *Rule:* If p is large relative to n, reduce dimensions first
  ]
]

#slide[
  = Handling Outliers

  *Impact by Method:*

  #table(
    columns: (1.5fr, 1fr),
    align: (left, center),
    stroke: 0.5pt,
    inset: 10pt,
    [*Method*], [*Sensitivity*],
    [K-means], [High],
    [Ward's Method], [High],
    [Single Linkage], [Medium],
    [K-medoids], [Low (Robust)],
  )
]

#slide[
  = Handling Outliers: Strategies

  - Pre-processing: Detect and remove outliers
  - Use robust methods (k-medoids)
  - Accept outlier clusters
]

#slide[
  = When to Use Hierarchical Clustering

  - Small to medium datasets (n < 5,000)
  - Want to explore different k values
  - Need hierarchical structure
  - Don't know k in advance
]

#slide[
  = When to Use K-Means

  - Large datasets (n > 5,000)
  - Approximately know k
  - Need speed and efficiency
  - Clusters roughly spherical
]

// Section 8: Workflow and Best Practices
#slide[
  = Cluster Analysis Workflow

  + *Define objective* - What questions to answer?
  + *Select variables* - Domain knowledge
  + *Preprocess data* - Handle missing values, outliers
  + *Standardize* - If variables on different scales
  + *Choose method* - Based on data characteristics
]

#slide[
  = Cluster Analysis Workflow

  6. *Determine k* - Multiple criteria
  7. *Run clustering* - Multiple times for k-means
  8. *Validate results* - Internal and stability checks
  9. *Interpret clusters* - Profile and name clusters
  10. *Refine and iterate* - Based on insights
]

#slide[
  = Common Pitfalls to Avoid

  + *Not standardizing* when variables have different scales
  + *Using k-means* with non-spherical clusters
  + *Ignoring outliers* - can severely distort results
  + *Over-interpreting* - clustering always finds structure, even in random data
]

#slide[
  = Common Pitfalls to Avoid

  5. *Using too many variables* - curse of dimensionality
  6. *Running k-means once* - try multiple initializations
  7. *Choosing k without validation* - use multiple methods
]

#slide[
  = Best Practices

  + *Try multiple methods* - Compare hierarchical, k-means, etc.
  + *Validate stability* - Bootstrap samples, different initializations
  + *Visualize extensively* - Scatter plots, dendrograms, parallel coordinates
]

#slide[
  = Best Practices

  4. *Use domain knowledge* - Statistical metrics + practical sense
  5. *Document decisions* - Why certain methods, parameters chosen
  6. *Check interpretability* - Can you explain and use clusters?
]

// Summary
#slide[
  = Key Takeaways: Fundamental Concepts

  - Cluster analysis discovers natural groupings (unsupervised)
  - Distance measures are crucial (Euclidean, Manhattan)
  - Standardization essential for different scales
]

#slide[
  = Key Takeaways: Methods

  - Hierarchical: Creates tree structure, multiple k values
  - K-means: Fast, scalable, requires specifying k
  - K-medoids: Robust alternative to k-means
]

#slide[
  = Key Takeaways: Validation

  - Elbow method and silhouette analysis for optimal k
  - Multiple validation measures for quality assessment
]

#slide[
  = Summary: Method Selection Guide

  #table(
    columns: (1fr, 2fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 8pt,
    [*Situation*], [*Recommended Method*],
    [Small dataset (n < 1,000)], [Hierarchical (Ward's or Average)],
    [Large dataset (n > 10,000)], [K-means with k-means++],
    [Outliers present], [K-medoids or preprocessing],
    [Non-spherical clusters], [DBSCAN or hierarchical],
  )
]

#slide[
  = Summary: Method Selection Guide

  #table(
    columns: (1fr, 2fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 8pt,
    [*Situation*], [*Recommended Method*],
    [Don't know k], [Hierarchical, then elbow/silhouette],
    [High dimensions], [PCA first, then k-means],
    [Mixed data types], [Gower distance with hierarchical],
  )
]

#slide[
  = Advanced Topics (Beyond This Course)

  *Density-Based Methods:*
  - DBSCAN - finds arbitrary shapes, identifies outliers

  *Model-Based:*
  - Gaussian Mixture Models (GMM) - probabilistic approach
]

#slide[
  = Advanced Topics (Beyond This Course)

  *Fuzzy Clustering:*
  - Soft assignment (membership degrees)

  *Subspace Clustering:*
  - For high-dimensional data, different subspaces
]

// Applications
#slide[
  = Real-World Applications: Business

  *Marketing & Business:*
  - Customer segmentation for targeted marketing
  - Product recommendation systems
  - Market basket analysis
]

#slide[
  = Real-World Applications: Healthcare

  *Healthcare:*
  - Patient stratification for personalized medicine
  - Disease subtype identification
  - Medical image segmentation
]

#slide[
  = Real-World Applications: Finance

  *Finance:*
  - Fraud detection and anomaly identification
  - Credit risk assessment
  - Portfolio diversification
]

#slide[
  = Example: Customer Segmentation

  *Scenario:* E-commerce company with 100,000 customers

  *Variables:*
  - Purchase frequency
  - Average order value
  - Product category preferences
  - Time since last purchase
  - Customer lifetime value
]

#slide[
  = Example: Customer Segmentation Process

  + Standardize variables (different scales)
  + Try k-means for k = 2 to 10
  + Use elbow method and silhouette analysis
  + Identify k = 5 optimal clusters
  + Profile each segment
  + Develop targeted marketing strategies
]

// Resources
#slide[
  = Recommended Resources: Books

  *Textbooks:*
  - Everitt et al. (2011) - Cluster Analysis (5th ed.)
  - James et al. (2021) - Introduction to Statistical Learning
]

#slide[
  = Recommended Resources: Software

  *Software:*
  - Python: scikit-learn (KMeans, AgglomerativeClustering)
  - R: stats package (kmeans, hclust)
]

#slide[
  = Recommended Resources: Online

  *Online:*
  - StatQuest YouTube channel
  - Scikit-learn documentation
  - Coursera/edX courses on unsupervised learning
]

// Final slide
#slide[
  = Questions?

  #align(center)[
    #text(size: 24pt, weight: "bold")[
      Thank you for your attention!
    ]

    Juliho Castillo Colmenares \
    julihocc\@tec

    Office: Tec de Monterrey CCM Office 1540 \
    Office Hours: Monday-Friday, 9:00 AM - 5:00 PM
  ]
]

#slide[
  = Next Steps: This Week

  *For This Week:*
  - Review lecture notes thoroughly
  - Practice with provided examples
  - Complete practice questions
  - Prepare for E06 quiz
]

#slide[
  = Next Steps: Preparation for Evaluation

  *Preparation for Evaluation:*
  - Understand distance measures and when to use each
  - Know linkage methods and their properties
  - Practice interpreting dendrograms
  - Understand k-means algorithm and convergence
  - Be able to explain validation methods
]
