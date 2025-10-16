// Discriminant Analysis - Companion Notes
// MA2003B - Application of Multivariate Methods in Data Science
// Dr. Juliho Castillo - Tecnologico de Monterrey

#set document(
  title: "Discriminant Analysis - Companion Notes",
  author: "Dr. Juliho Castillo"
)

#set page(
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1"
)

#set text(
  font: "Liberation Sans",
  size: 11pt
)

#set heading(numbering: "1.")

#set par(justify: true, leading: 0.65em)

#align(center)[
  #text(size: 22pt, weight: "bold")[Discriminant Analysis]

  #v(0.3cm)
  #text(size: 14pt)[Companion Notes for Chapter 5]

  #v(0.5cm)
  #text(size: 12pt)[MA2003B - Multivariate Methods in Data Science]

  #v(0.2cm)
  #text(size: 11pt)[Dr. Juliho Castillo #sym.dot Tecnologico de Monterrey]
]

#v(1.5cm)

#outline(depth: 2, indent: 1em)

#pagebreak()

#set par(first-line-indent: 0em)

#block(
  fill: rgb("#e8f4f8"),
  inset: 1em,
  radius: 5pt,
  width: 100%,
  [
    *Hands-On Learning Resource*

    This document is accompanied by an interactive Jupyter notebook that demonstrates all concepts with executable Python code and visualizations:

    `ch5_guiding_example/marketing_discriminant_analysis.ipynb`

    The notebook analyzes a customer segmentation case study with 1,200 e-commerce customers, comparing Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) for marketing campaign optimization.

    Each major concept in these notes references specific modules in the notebook where you can see the implementation and results.
  ]
)

#v(1cm)

= Introduction: The Classification Problem

== Why Discriminant Analysis?

Imagine you are a marketing analyst at an e-commerce company with thousands of customers. You want to answer a fundamental question: *Given a customer's behavior, which segment do they belong to?* Are they a high-value customer worth premium retention efforts? A loyal regular who might respond to upselling? Or an occasional buyer who needs re-engagement?

This is a *classification problem* - we have several known groups and want to assign new observations to the correct group based on measured characteristics. Discriminant Analysis provides a principled statistical approach to this problem.

== Real-World Applications

*Business & Marketing*
- Customer segmentation for targeted campaigns
- Credit risk assessment (approve/reject loan applications)
- Churn prediction (will this customer leave?)

*Healthcare & Medicine*
- Disease diagnosis from patient symptoms and test results
- Treatment response prediction
- Medical imaging classification

*Manufacturing*
- Quality control (acceptable/borderline/defective products)
- Defect type classification
- Process monitoring and fault detection

*Sports & Performance*
- Athlete classification for training programs
- Talent identification systems
- Performance level assessment

== The Core Idea

Discriminant Analysis finds *discriminant functions* --- linear (or quadratic) combinations of your predictor variables that best separate your groups. Think of it as finding the "best viewing angle" to distinguish between groups in multidimensional space.

*Key Question*: Which combination of behavioral metrics (purchase frequency, order value, engagement, etc.) best distinguishes high-value customers from loyal customers from occasional customers?

= Mathematical Foundations

== The Setup

Let's formalize the problem. We have:

- $g$ groups (populations) we want to classify into
- $p$ predictor variables measured on each observation
- A training dataset with known group memberships
- Goal: Classify new observations into one of the $g$ groups

*Notation*:
- $bold(x) = (x_1, x_2, ..., x_p)^top$: vector of predictor variables for an observation
- $pi_k$: Prior probability of group $k$ (proportion in population)
- $bold(mu)_k$: Mean vector for group $k$
- $bold(Sigma)_k$: Covariance matrix for group $k$
- $f_k (bold(x))$: Probability density function for group $k$

== Classification Rules: Bayes Theorem

The foundation of discriminant analysis is Bayes theorem. Given observation $bold(x)$, the posterior probability of belonging to group $k$ is:

$ P(G = k | bold(x)) = frac(f_k (bold(x)) pi_k, sum_(j=1)^g f_j (bold(x)) pi_j) $

*Bayes Classification Rule*: Assign observation $bold(x)$ to the group with the highest posterior probability.

This is optimal in the sense that it minimizes the total probability of misclassification (if priors and densities are known).

== Linear Discriminant Analysis (LDA)

=== Assumptions

LDA makes two critical assumptions:

1. *Multivariate Normality*: Each group follows a multivariate normal distribution
2. *Equal Covariances*: All groups share the same covariance matrix: $bold(Sigma)_1 = bold(Sigma)_2 = ... = bold(Sigma)_g = bold(Sigma)$

Under these assumptions, the discriminant functions become *linear* in $bold(x)$.

=== Fisher Linear Discriminant

An alternative (but equivalent) approach by R.A. Fisher: Find linear combinations of variables that maximize the ratio of between-group variance to within-group variance.

*For two groups*, find weights $bold(a)$ that maximize:

$ "maximize" quad frac((overline(y)_1 - overline(y)_2)^2, s_1^2 + s_2^2) $

where $y = bold(a)^top bold(x)$ is the discriminant score.

The solution is: $bold(a) prop bold(Sigma)^(-1) (bold(mu)_1 - bold(mu)_2)$

This gives us the *linear discriminant function*.

=== Discriminant Scores

For observation $bold(x)$, the discriminant score for group $k$ is:

$ delta_k (bold(x)) = bold(x)^top bold(Sigma)^(-1) bold(mu)_k - frac(1, 2) bold(mu)_k^top bold(Sigma)^(-1) bold(mu)_k + log(pi_k) $

*Classification Rule*: Assign $bold(x)$ to the group with the largest discriminant score $delta_k (bold(x))$.

=== Geometric Interpretation

- Each discriminant function defines a *hyperplane* in $p$-dimensional space
- These hyperplanes are the *decision boundaries* between groups
- Decision boundaries are *linear* (hence the name)
- Boundaries are perpendicular bisectors of lines connecting group centroids (when priors are equal)

== Quadratic Discriminant Analysis (QDA)

=== When LDA Is Not Enough

QDA relaxes the equal covariance assumption. Each group $k$ has its own covariance matrix $bold(Sigma)_k$.

*When to use QDA*:
- Groups have genuinely different variability patterns
- You have sufficient sample size (more parameters to estimate)
- Linear boundaries do not fit the data well

=== Quadratic Discriminant Functions

The discriminant score becomes:

$ delta_k (bold(x)) = -frac(1, 2) log |bold(Sigma)_k| - frac(1, 2) (bold(x) - bold(mu)_k)^top bold(Sigma)_k^(-1) (bold(x) - bold(mu)_k) + log(pi_k) $

This is *quadratic* in $bold(x)$, leading to curved (quadratic) decision boundaries.

=== Trade-offs: LDA vs QDA

*LDA Advantages*:
- Fewer parameters ($p(p+1)/2$ vs $g dot p(p+1)/2$ for covariances)
- More stable with smaller sample sizes
- Less prone to overfitting
- Simpler interpretation

*QDA Advantages*:
- More flexible (can fit complex boundaries)
- Better accuracy when covariances truly differ
- Does not assume equal variances

*Rule of Thumb*: Start with LDA. Move to QDA if:
1. You have large sample size ($n$ much larger than $p$)
2. Groups show clearly different spreads
3. LDA performance is poor

= Practical Implementation

*Interactive Tutorial*: The notebook `ch5_guiding_example/marketing_discriminant_analysis.ipynb` demonstrates each step of this workflow with executable Python code, detailed explanations, and visualizations.

== The Analysis Workflow

=== Step 1: Data Preparation

*Notebook Reference*: See Module 2 for complete data preparation code.

*Feature Selection*
- Choose predictors that discriminate between groups
- Remove highly correlated predictors (multicollinearity issues)
- Consider domain knowledge

*Standardization*
- Standardize variables if they're on different scales
- Example: Do not mix dollars (0-200) with rates (0-1)
- The notebook uses `StandardScaler` to transform all features to mean=0, std=1

*Train/Test Split*
- Use stratified sampling to preserve group proportions
- Typical split: 70% train, 30% test
- Ensures all groups represented in both sets
- Implementation: `train_test_split` with `stratify=y` parameter

=== Step 2: Assumption Checking

*Multivariate Normality*
- Q-Q plots for each variable by group
- Multivariate tests (Mardia test, Henze-Zirkler test)
- Not critical for large samples (robust to violations)

*Equal Covariances (for LDA)*
- Box's M test (very sensitive, often rejects)
- Visual inspection: covariance matrices by group
- If violated: consider QDA or transformation

*Multicollinearity*
- Correlation matrix of predictors
- Variance Inflation Factors (VIF)
- Remove redundant predictors if needed

=== Step 3: Model Fitting

*Notebook Reference*: Module 3 (LDA) and Module 6 (QDA) demonstrate model fitting with scikit-learn.

*LDA Estimation*
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
```

Key outputs:
- Discriminant function coefficients (`lda.scalings_`)
- Group means (centroids) (`lda.means_`)
- Pooled covariance matrix (internal)
- Prior probabilities (`lda.priors_`)
- Explained variance ratio (`lda.explained_variance_ratio_`)

*QDA Estimation*
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
```

Key outputs:
- Group means (`qda.means_`)
- Group-specific covariance matrices (internal)
- Prior probabilities (`qda.priors_`)
- Posterior probabilities via `predict_proba()`

=== Step 4: Interpretation

*Notebook Reference*: Module 4 provides detailed coefficient interpretation with group means analysis.

*Discriminant Functions*

For LDA with $g$ groups, we get up to $min(g-1, p)$ discriminant functions.

*First discriminant function* (LD1): Explains most between-group variation
*Second discriminant function* (LD2): Second-most variation (orthogonal to LD1)

*Marketing Example Interpretation* (from the notebook):
- *LD1* (95.8% of variance) separates High-Value from Occasional customers
  - Strong negative coefficients: purchase frequency, loyalty points
  - Represents overall customer engagement and activity level
- *LD2* (4.2% of variance) captures remaining separation
  - Strong positive coefficient: average order value
  - Strong negative coefficient: browsing time
  - Represents order size patterns versus browsing efficiency

The notebook's Module 4 displays both the discriminant coefficients (scalings) and the group means on standardized features, allowing interpretation of which behavioral patterns define each segment.

*Discriminant Loadings*

Similar to factor analysis loadings - correlation between original variable and discriminant function.

High absolute loading = variable is important for that discriminant function.

The notebook creates a DataFrame of coefficients to easily identify the most influential features for each discriminant function.

=== Step 5: Validation

*Notebook Reference*: Modules 3, 6, 9, and 10 provide comprehensive validation analyses.

*Classification Accuracy*
- Confusion matrix on test set (Module 9)
- Overall accuracy, precision, recall per class
- Classification report with F1-scores
- Don't just look at overall accuracy!

*Cross-Validation*
- K-fold cross-validation (typically k=5 or k=10)
- Provides more robust performance estimate
- Checks stability across different data splits
- The notebook uses `cross_val_score` with cv=5

*Visual Validation*
- Scatter plot of discriminant scores (LD1 vs LD2) - Module 5
- Decision boundary plots (for 2D projections) - Module 8
- ROC curves for each segment (One-vs-Rest) - Module 10
- Verify groups are well-separated

*Advanced Metrics*
- Module 10 demonstrates ROC curve analysis with AUC scores
- Module 7 analyzes posterior probabilities to assess prediction confidence
- Module 9 provides side-by-side confusion matrix comparison for LDA vs QDA

= Applied Example: Marketing Segmentation

*Complete Implementation*: See `ch5_guiding_example/marketing_discriminant_analysis.ipynb` for the full interactive analysis with visualizations and detailed interpretations.

== Business Problem

An e-commerce company has 1,200 customers and wants to classify them into three segments for targeted marketing:

1. *High-Value* (30%): Premium customers, high spending and engagement
2. *Loyal* (40%): Regular customers, moderate spending, consistent
3. *Occasional* (30%): Infrequent buyers, need re-engagement

The dataset is synthetically generated using `fetch_marketing.py` to ensure reproducibility and known statistical properties for educational purposes.

== Variables (p = 8)

*Behavioral Metrics*:
- Purchase frequency (purchases/month)
- Average order value (USD)
- Browsing time (minutes/session)
- Cart abandonment rate (0-1)
- Email open rate (0-1)
- Loyalty points (accumulated)
- Support tickets (per month)
- Social engagement (interactions/month)

See `MARKETING_DATA_DICTIONARY.md` for complete variable descriptions and data generation methodology.

== Analysis Strategy

=== Why Both LDA and QDA?

*LDA*: Assumes all customer segments have similar variability patterns
*QDA*: Allows different patterns (e.g., Occasional customers might be more variable)

We'll fit both and compare performance.

=== Feature Standardization

Since variables are on different scales (dollars, rates, counts), we standardize:

$ z_j = frac(x_j - overline(x)_j, s_j) $

This ensures no variable dominates due to scale.

== Results

*Note*: The following results are from the Jupyter notebook analysis. Run `marketing_discriminant_analysis.ipynb` to reproduce these findings and generate visualizations.

=== Discriminant Functions

The notebook's Module 4 (Interpreting Discriminant Functions) reveals:

*LD1 (95.8% of between-group variance)*:
- Separates High-Value from Occasional customers along the primary axis
- Key drivers: Purchase frequency (strong negative), loyalty points (strong negative), average order value (negative)
- Interpretation: Overall customer value and activity level

*LD2 (4.2% of between-group variance)*:
- Distinguishes remaining group differences
- Key drivers: Average order value (strong positive), browsing time (negative)
- Interpretation: Order size versus browsing efficiency

*Insight*: Two independent dimensions describe customer segments:
1. Overall engagement and frequency (LD1 - dominant factor)
2. Purchase value patterns (LD2 - secondary factor)

=== Classification Performance

*LDA Results* (Module 3):
- Test accuracy: Nearly perfect classification
- Cross-validation: 99.9% (#sym.plus.minus 0.3%)
- Strong performance across all segments

*QDA Results* (Module 6):
- Test accuracy: Perfect classification
- Cross-validation: 100.0% (#sym.plus.minus 0.0%)
- Slightly outperforms LDA with more flexible boundaries

*Interpretation*: Both models achieve excellent performance. The synthetic data's well-separated structure allows for near-perfect classification, demonstrating the power of discriminant analysis when groups have distinct multivariate profiles.

=== Model Comparison and Visualization

The notebook includes comprehensive visualizations:

*Module 5*: Discriminant space scatter plot showing customer distribution in LD1-LD2 space with group centroids

*Module 8*: QDA decision boundaries visualization using purchase frequency and average order value

*Module 9*: Side-by-side confusion matrices comparing LDA and QDA performance

*Module 10*: ROC curves for each segment showing excellent discrimination (AUC near 1.0)

=== Model Selection Recommendation

Module 11 provides a comprehensive comparison and recommends *LDA* despite QDA's marginally better performance because:
- Similar accuracy (difference is negligible)
- Simpler model with fewer parameters
- More interpretable discriminant functions
- Lower overfitting risk
- Easier to explain to stakeholders

== Business Insights

=== Segment Characteristics

Based on the group means analysis in Module 4:

*High-Value Customers*:
- Positive standardized values for purchase frequency, order value, browsing time
- Very high email open rate and social engagement
- Low cart abandonment and support tickets
- Highest loyalty points accumulation
- Strategy: Premium services, personalized recommendations, retention focus

*Loyal Customers*:
- Moderate purchase frequency and order values
- Good email engagement and loyalty points
- Balanced across most metrics
- Strategy: Upselling, cross-selling, loyalty program enhancements

*Occasional Customers*:
- Low (negative) values for most engagement metrics
- High cart abandonment rate and support ticket volume
- Minimal loyalty points and social engagement
- Strategy: Re-engagement campaigns, cart recovery, customer education

=== Actionable Applications

*New Customer Classification*:
Once a customer accumulates enough behavioral data (typically after 2-3 months), the model automatically assigns them to a segment for tailored marketing. Module 7 shows how posterior probabilities provide confidence scores for each classification.

*Monitoring Segment Migration*:
Track customers over time - are Occasional customers moving to Loyal? Are Loyal customers at risk of becoming Occasional? The discriminant scores can detect early warning signs of segment transition.

*Marketing ROI Optimization*:
Focus expensive retention campaigns on High-Value customers (high classification confidence), use cheaper email campaigns for Loyal customers, and implement automated cart recovery for Occasional customers.

*Campaign Personalization*:
Module 7's posterior probability analysis identifies customers with ambiguous segment membership who might respond to hybrid marketing strategies.

= Advanced Topics

== Variable Selection

Not all variables may be necessary. Common approaches:

*Stepwise Discriminant Analysis*:
- Forward selection: Add variables one at a time (highest F-to-enter)
- Backward elimination: Remove variables (lowest F-to-remove)
- Uses Wilks' Lambda or F-statistics

*Shrinkage Methods*:
- Regularized Discriminant Analysis (RDA)
- Penalized LDA (e.g., L1 penalty for sparse solutions)

== Handling Imbalanced Classes

If groups have very different sizes (e.g., 95% acceptable, 5% defective):

*Adjust Prior Probabilities*:
- Use equal priors instead of sample proportions
- Or set priors based on business costs

*Sampling Techniques*:
- Oversample minority class
- Undersample majority class
- SMOTE (Synthetic Minority Over-sampling)

== Model Diagnostics

*Wilks' Lambda*:
Tests if group means are significantly different:
$ Lambda = frac(|bold(W)|, |bold(T)|) $

Small values (near 0) = strong group separation

*Canonical Correlation*:
Measures strength of relationship between discriminant functions and groups:
$ R_"can" = sqrt(frac("between-group SS", "total SS")) $

Values near 1 = excellent discrimination

== Comparison with Other Methods

*Logistic Regression*:
- More flexible (no normality assumption)
- Works better with binary outcomes
- Provides probability estimates directly
- Less interpretable for multiple groups

*Support Vector Machines (SVM)*:
- Can handle non-linear boundaries (kernel trick)
- No assumptions about distributions
- Harder to interpret
- Often better for high-dimensional data

*Random Forests*:
- Handles non-linear relationships naturally
- Robust to outliers
- Provides variable importance
- Black-box model (less interpretable)

*When to use Discriminant Analysis*:
- Moderate sample size, moderate dimensionality
- Interpretability is important
- Want to understand group differences
- Groups are reasonably normal and have similar spreads

= Common Pitfalls and Best Practices

== Common Mistakes

*1. Ignoring Assumptions*:
- Using LDA when groups have clearly different covariances
- Not checking for outliers (heavily influence results)

*2. Overfitting*:
- Too many predictors relative to sample size
- Rule: Need at least 20 observations per predictor *per group*

*3. Using Training Accuracy*:
- Always evaluate on held-out test set
- Training accuracy is overly optimistic

*4. Ignoring Class Imbalance*:
- Model may just predict majority class
- Check per-class metrics, not just overall accuracy

*5. Correlated Predictors*:
- Multicollinearity inflates standard errors
- Remove redundant variables

== Best Practices

*Data Quality*:
- Handle missing data appropriately
- Screen for outliers (Mahalanobis distance)
- Verify data entry errors

*Model Selection*:
- Start simple (LDA) before complex (QDA)
- Use cross-validation for honest performance estimates
- Consider multiple performance metrics

*Interpretation*:
- Don't just report accuracy - explain discriminant functions
- Visualize decision boundaries when possible
- Translate statistical results to domain insights

*Validation*:
- Test on truly independent data if possible
- Monitor performance over time in production
- Update model as patterns change

= Summary and Key Takeaways

*Discriminant Analysis is*:
- A classification method for assigning observations to groups
- Based on finding linear (LDA) or quadratic (QDA) combinations of predictors
- Optimal under normality and known covariances (Bayes rule)

*When to use it*:
- You have labeled training data
- Want interpretable group differences
- Moderate sample size and dimensionality
- Groups are reasonably normal

*Key decisions*:
- LDA vs QDA (equal covariances vs flexibility)
- Which predictors to include (domain knowledge + statistics)
- How to handle priors (equal vs proportional vs cost-based)

*Interpretation matters*:
- Discriminant functions show *how* groups differ
- Loadings reveal which variables matter most
- Classification accuracy tells you how well it works

*Always validate*:
- Hold-out test set or cross-validation
- Check per-group performance
- Monitor over time in real applications

The power of discriminant analysis lies not just in classification accuracy, but in *understanding* what makes groups different - turning multivariate data into actionable insights.

= Putting It Into Practice

To solidify your understanding of discriminant analysis, work through the complete marketing segmentation example:

*Step 1*: Generate the customer dataset
```bash
cd ch5_guiding_example
python fetch_marketing.py
```

*Step 2*: Open and run the Jupyter notebook
```bash
jupyter notebook marketing_discriminant_analysis.ipynb
```

*Step 3*: Execute each module sequentially, reading the explanations and examining the outputs

*Step 4*: Experiment with modifications:
- Try different train/test splits
- Exclude certain features to see impact on performance
- Adjust prior probabilities in the discriminant models
- Create additional visualizations

*Step 5*: Review the generated PNG files:
- `marketing_lda_scores.png`: Discriminant space visualization
- `marketing_qda_boundaries.png`: Decision boundaries in 2D
- `marketing_confusion_comparison.png`: LDA vs QDA performance
- `marketing_roc_curves.png`: Segment-specific classification quality

By working through the complete pipeline from data generation to model comparison, you will develop practical skills in applying discriminant analysis to real-world classification problems.

#align(center)[
  #v(2cm)
  #text(size: 10pt, style: "italic")[
    "The goal is to turn data into information, and information into insight." \ - Carly Fiorina
  ]
]
