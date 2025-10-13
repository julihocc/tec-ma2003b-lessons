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

== The Analysis Workflow

=== Step 1: Data Preparation

*Feature Selection*
- Choose predictors that discriminate between groups
- Remove highly correlated predictors (multicollinearity issues)
- Consider domain knowledge

*Standardization*
- Standardize variables if they're on different scales
- Example: Do not mix dollars (0-200) with rates (0-1)

*Train/Test Split*
- Use stratified sampling to preserve group proportions
- Typical split: 70% train, 30% test
- Ensures all groups represented in both sets

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

*LDA Estimation*
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
```

Key outputs:
- Discriminant function coefficients
- Group means (centroids)
- Pooled covariance matrix
- Prior probabilities

*QDA Estimation*
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
```

Key outputs:
- Group means
- Group-specific covariance matrices
- Prior probabilities

=== Step 4: Interpretation

*Discriminant Functions*

For LDA with $g$ groups, we get up to $min(g-1, p)$ discriminant functions.

*First discriminant function* (LD1): Explains most between-group variation
*Second discriminant function* (LD2): Second-most variation (orthogonal to LD1)

*Marketing Example Interpretation*:
- *LD1* separates High-Value from Occasional (overall customer value)
  - High positive: order value, loyalty points, social engagement
  - High negative: cart abandonment, support tickets
- *LD2* distinguishes Loyal from others (frequency vs browsing)
  - High positive: purchase frequency, email engagement
  - High negative: browsing time (efficient shoppers)

*Discriminant Loadings*

Similar to factor analysis loadings - correlation between original variable and discriminant function.

High absolute loading = variable is important for that discriminant function.

=== Step 5: Validation

*Classification Accuracy*
- Confusion matrix on test set
- Overall accuracy, precision, recall per class
- Don't just look at overall accuracy!

*Cross-Validation*
- K-fold cross-validation (typically k=5 or k=10)
- Provides more robust performance estimate
- Checks stability across different data splits

*Visual Validation*
- Scatter plot of discriminant scores (LD1 vs LD2)
- Decision boundary plots (for 2D projections)
- Verify groups are well-separated

= Applied Example: Marketing Segmentation

== Business Problem

An e-commerce company has 1,200 customers and wants to classify them into three segments for targeted marketing:

1. *High-Value* (30%): Premium customers, high spending and engagement
2. *Loyal* (40%): Regular customers, moderate spending, consistent
3. *Occasional* (30%): Infrequent buyers, need re-engagement

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

=== Discriminant Functions

*LD1 (67% of between-group variance)*:
- Separates High-Value from Occasional customers
- Key drivers: Order value (+), loyalty points (+), cart abandonment (-)

*LD2 (33% of between-group variance)*:
- Distinguishes Loyal customers
- Key drivers: Purchase frequency (+), browsing time (-)

*Insight*: Two independent dimensions describe customer segments:
1. Overall value/engagement level (LD1)
2. Efficiency of shopping behavior (LD2)

=== Classification Performance

*LDA Results*:
- Training accuracy: 100% (perfect separation in training)
- Test accuracy: 92%
- Cross-validation: 91% (#sym.plus.minus 2%)

*QDA Results*:
- Training accuracy: 100%
- Test accuracy: 94%
- Cross-validation: 93% (#sym.plus.minus 2%)

*Interpretation*: QDA slightly outperforms LDA, suggesting groups have somewhat different covariance structures. Both perform well.

=== Confusion Matrix (LDA on Test Set)

```
                 Predicted
              HV   Loyal  Occ
Actual  HV    103    5     0
        Loyal   4  140     4
        Occ     0    6   102
```

*Observations*:
- High-Value rarely misclassified (strong signal)
- Occasional sometimes confused with Loyal (moderate overlap)
- Overall strong performance

== Business Insights

=== Segment Characteristics

*High-Value Customers*:
- High discriminant score on LD1 (positive)
- Moderate score on LD2
- Strategy: Premium services, personalized recommendations, retention focus

*Loyal Customers*:
- Moderate LD1 score
- High LD2 score (efficient, frequent purchasers)
- Strategy: Upselling, cross-selling, loyalty program enhancements

*Occasional Customers*:
- Low LD1 score (negative)
- Low LD2 score
- Strategy: Re-engagement campaigns, cart recovery, education

=== Actionable Applications

*New Customer Classification*:
Once a customer accumulates enough behavioral data, the model automatically assigns them to a segment for tailored marketing.

*Monitoring Segment Migration*:
Track customers over time - are Occasional customers moving to Loyal? Are Loyal customers at risk of becoming Occasional?

*Marketing ROI Optimization*:
Focus expensive retention campaigns on High-Value customers (high accuracy), use cheaper email campaigns for Loyal customers.

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

#align(center)[
  #v(2cm)
  #text(size: 10pt, style: "italic")[
    "The goal is to turn data into information, and information into insight." \ - Carly Fiorina
  ]
]
