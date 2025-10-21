// Discriminant Analysis - Presentation Slides
// MA2003B - Application of Multivariate Methods in Data Science
// Dr. Juliho Castillo - Tecnologico de Monterrey

#import "@preview/polylux:0.4.0": *

#set page(paper: "presentation-16-9")
#set text(size: 20pt)

#slide[
  #align(center + horizon)[
    #text(size: 40pt, weight: "bold")[Discriminant Analysis]
    
    #v(1em)
    
    #text(size: 24pt)[Classification with Statistical Foundations]
    
    #v(2em)
    
    #text(size: 18pt)[
      MA2003B - Multivariate Methods in Data Science
      
      #v(0.5em)
      
      Dr. Juliho Castillo
      
      #v(0.3em)
      
      Tecnologico de Monterrey
      
      #v(0.5em)
      
      #datetime.today().display()
    ]
  ]
]

#slide[
  = The Classification Problem
  
  *Scenario:* E-commerce company with thousands of customers
  
  #v(1em)
  
  *Question:* Which segment does each customer belong to?
  
  #v(1em)
  
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    align: center + horizon,
    [
      *High-Value*
      
      Premium customers
      
      High spending
      
      Max engagement
    ],
    [
      *Loyal*
      
      Regular customers
      
      Moderate spending
      
      Consistent activity
    ],
    [
      *Occasional*
      
      Infrequent buyers
      
      Low engagement
      
      Need re-engagement
    ]
  )
]

#slide[
  = Real-World Applications
  
  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *Business & Marketing*
      - Customer segmentation
      - Credit risk assessment
      - Churn prediction
      
      *Healthcare*
      - Disease diagnosis
      - Treatment prediction
      - Medical imaging
    ],
    [
      *Manufacturing*
      - Quality control
      - Defect classification
      - Fault detection
      
      *Sports Analytics*
      - Athlete classification
      - Talent identification
      - Performance assessment
    ]
  )
]

#slide[
  = The Core Idea
  
  #v(1em)
  
  *Discriminant Analysis finds discriminant functions*
  
  Linear or quadratic combinations of predictors that *best separate groups*
  
  #v(2em)
  
  #align(center)[
    #box(fill: blue.lighten(90%), inset: 1.5em, radius: 8pt)[
      #text(size: 22pt)[
        Think of it as finding the *"best viewing angle"*
        
        to distinguish groups in multidimensional space
      ]
    ]
  ]
]

#slide[
  = Mathematical Framework
  
  *Setup:*
  - $g$ distinct groups or populations
  - $p$ predictor variables per observation
  - Training data with known group memberships
  
  #v(1em)
  
  *Key Notation:*
  - $bold(x) = (x_1, ..., x_p)^top$ predictor vector
  - $pi_k$ prior probability of group $k$
  - $bold(mu)_k$ mean vector for group $k$
  - $bold(Sigma)_k$ covariance matrix for group $k$
  - $f_k (bold(x))$ probability density for group $k$
]

#slide[
  = Bayes Theorem Foundation

  *Goal:* Classify observation with features $bold(x)$ into one of $g$ groups

  #v(1em)

  *Bayes Theorem gives posterior probability:*

  $ P(G = k | bold(x)) = frac(f_k (bold(x)) pi_k, sum_(j=1)^g f_j (bold(x)) pi_j) $

  #v(0.5em)

  where:
  - $pi_k$ = prior probability of group $k$
  - $f_k (bold(x))$ = probability density of $bold(x)$ in group $k$

  #v(1em)

  *Bayes Optimal Classification:*

  $ k^* = arg max_k P(G = k | bold(x)) = arg max_k f_k (bold(x)) pi_k $

  Denominator is same for all groups, so we can ignore it
]

#slide[
  = Example: Credit Risk - Setup
  
  *Business Context:*
  
  Bank evaluating loan application. Two possible outcomes:
  - Group 0: Customer will *not default* (repay loan)
  - Group 1: Customer will *default* (fail to repay)
  
  #v(1em)
  
  *Applicant Profile:*
  - Annual income: 50,000 USD
  - Debt-to-income ratio: 0.4 (40%)
  - Credit score: 650
  
  #v(1em)
  
  *Historical Data (Prior Probabilities):*
  - $pi_0 = 0.95$ (95% of past customers did not default)
  - $pi_1 = 0.05$ (5% of past customers defaulted)
]

#slide[
  = Example: Credit Risk - Likelihood
  
  *Probability Densities:*
  
  How likely is this profile in each group?
  
  #v(1em)
  
  *No Default Group ($k=0$):*
  
  $f_0 (bold(x)) = 0.0008$
  
  This profile is *uncommon* among non-defaulters (lower income, higher debt)
  
  #v(1em)
  
  *Default Group ($k=1$):*
  
  $f_1 (bold(x)) = 0.0030$
  
  This profile is *more typical* among defaulters (3.75 times more likely)
]

#slide[
  = Example: Credit Risk - Calculation
  
  *Step 1: Calculate numerators (prior times likelihood)*
  
  - No default: $f_0 (bold(x)) times pi_0 = 0.0008 times 0.95 = 0.00076$
  - Default: $f_1 (bold(x)) times pi_1 = 0.0030 times 0.05 = 0.00015$
  
  #v(0.5em)
  
  *Step 2: Calculate denominator (sum of numerators)*
  
  $"Total" = 0.00076 + 0.00015 = 0.00091$
  
  #v(0.5em)
  
  *Step 3: Calculate posterior probabilities*
  
  - $P($no default$|bold(x)) = 0.00076 / 0.00091 = 0.835$ (83.5%)
  - $P($default$|bold(x)) = 0.00015 / 0.00091 = 0.165$ (16.5%)
]

#slide[
  = Example: Credit Risk - Interpretation

  *Key Insight:*

  Even though this profile is *3.75x more common* among defaulters...

  The *prior probability* (95% vs 5%) is so strong that we still classify as *no default*

  #v(1em)

  *Decision Rule:*

  Classify as *no default* (83.5% > 16.5%)

  #v(1em)

  *Business Implications:*
  - Approve loan, but consider higher interest rate
  - Monitor account more closely
  - May require additional collateral
  - 16.5% risk is still significant for portfolio management
]

#slide[
  = From Bayes to Discriminant Analysis

  *The Challenge:*

  We need to specify $f_k (bold(x))$ for each group

  #v(1em)

  *The Assumption:*

  Assume each group follows *multivariate normal distribution*:

  $ f_k (bold(x)) = frac(1, (2 pi)^(p\/2) |bold(Sigma)_k|^(1\/2)) exp(-frac(1, 2) (bold(x) - bold(mu)_k)^top bold(Sigma)_k^(-1) (bold(x) - bold(mu)_k)) $

  #v(1em)

  *Key Parameters:*
  - $bold(mu)_k$ = mean vector for group $k$
  - $bold(Sigma)_k$ = covariance matrix for group $k$
]

#slide[
  = Simplifying the Math

  *Recall:* We want to maximize $f_k (bold(x)) pi_k$

  #v(1em)

  *Trick:* Maximize $log(f_k (bold(x)) pi_k)$ instead (same result, easier math)

  #v(1em)

  *Taking the logarithm:*

  $ log(f_k (bold(x)) pi_k) = -frac(p, 2) log(2 pi) - frac(1, 2) log|bold(Sigma)_k| $
  $ - frac(1, 2) (bold(x) - bold(mu)_k)^top bold(Sigma)_k^(-1) (bold(x) - bold(mu)_k) + log(pi_k) $

  #v(1em)

  Drop constant terms (same for all groups), define discriminant score $delta_k (bold(x))$
]

#slide[
  = The Discriminant Score

  *What is it?*

  A simplified scoring function for each class $k$: assign $bold(x)$ to class with highest score

  #v(1em)

  *General formula (after dropping constants):*

  $ delta_k (bold(x)) = -frac(1, 2) log|bold(Sigma)_k| - frac(1, 2) (bold(x) - bold(mu)_k)^top bold(Sigma)_k^(-1) (bold(x) - bold(mu)_k) + log(pi_k) $

  #v(1em)

  *Three components:*

  *1.* $-frac(1, 2) log|bold(Sigma)_k|$ = penalty for group spread (larger covariance = lower score)

  *2.* $-frac(1, 2) (bold(x) - bold(mu)_k)^top bold(Sigma)_k^(-1) (bold(x) - bold(mu)_k)$ = Mahalanobis distance (closer to center = higher score)

  *3.* $log(pi_k)$ = prior probability boost (more common classes = higher score)
]

#slide[
  = Classification with Discriminant Scores

  *Decision Rule:*

  $ "Predicted class" = arg max_k delta_k (bold(x)) $

  Assign observation to the class with the highest discriminant score

  #v(1em)

  *Why this works:*

  - Maximizing $delta_k (bold(x))$ is equivalent to maximizing $P(G = k | bold(x))$
  - We dropped only constant terms (same for all groups)
  - Classification decision remains optimal (Bayes rule)

  #v(1em)

  *Computational benefit:*

  - Avoid computing actual probabilities (no denominator needed)
  - Work with simpler expressions (log-scale, no exponentials)
  - Still get optimal Bayesian classification
]

#slide[
  = Two Scenarios: LDA vs QDA

  *Scenario 1: Equal Covariances* (LDA assumption)

  If $bold(Sigma)_1 = bold(Sigma)_2 = ... = bold(Sigma)_g = bold(Sigma)$

  Then $log|bold(Sigma)_k|$ is constant across groups

  The quadratic term $(bold(x) - bold(mu)_k)^top bold(Sigma)^(-1) (bold(x) - bold(mu)_k)$ expands to terms linear in $bold(x)$

  Result: *Linear discriminant function*

  #v(1em)

  *Scenario 2: Different Covariances* (QDA assumption)

  Each group has $bold(Sigma)_k$

  Keep all terms including $log|bold(Sigma)_k|$

  Result: *Quadratic discriminant function*
]

#slide[
  = Summary: Bayes to LDA/QDA

  #align(center)[
    #box(fill: blue.lighten(90%), inset: 1em, radius: 8pt)[
      *The Complete Connection*
    ]
  ]

  #v(1em)

  *Step 1:* Bayes optimal rule requires maximizing $f_k (bold(x)) pi_k$

  #v(0.5em)

  *Step 2:* Assume multivariate normal: $f_k (bold(x)) tilde N(bold(mu)_k, bold(Sigma)_k)$

  #v(0.5em)

  *Step 3:* Take logarithm for computational convenience

  #v(0.5em)

  *Step 4:* Simplify based on covariance assumption:
  - *Equal covariances* $arrow.r$ LDA (linear boundaries)
  - *Different covariances* $arrow.r$ QDA (quadratic boundaries)

  #v(1em)

  #align(center)[
    Both methods are *Bayesian classifiers* under normality assumption
  ]
]

#slide[
  = Linear Discriminant Analysis (LDA)
  
  *Two Critical Assumptions:*
  
  #v(1em)
  
  *1. Multivariate Normality*
  
  Each group follows multivariate normal distribution
  
  #v(1em)
  
  *2. Equal Covariances*
  
  $bold(Sigma)_1 = bold(Sigma)_2 = ... = bold(Sigma)_g = bold(Sigma)$
  
  #v(2em)
  
  #align(center)[
    Result: *Linear* decision boundaries
  ]
]

#slide[
  = LDA: Deriving the Discriminant Scores

  *Start with log-likelihood, assume $bold(Sigma)_k = bold(Sigma)$ for all $k$:*

  $ log(f_k (bold(x)) pi_k) = -frac(1, 2) log|bold(Sigma)| - frac(1, 2) (bold(x) - bold(mu)_k)^top bold(Sigma)^(-1) (bold(x) - bold(mu)_k) + log(pi_k) $

  #v(0.5em)

  *Expand the quadratic term:*

  $ (bold(x) - bold(mu)_k)^top bold(Sigma)^(-1) (bold(x) - bold(mu)_k) = bold(x)^top bold(Sigma)^(-1) bold(x) - 2 bold(x)^top bold(Sigma)^(-1) bold(mu)_k + bold(mu)_k^top bold(Sigma)^(-1) bold(mu)_k $

  #v(0.5em)

  *Drop terms constant across groups:*

  Drop $-frac(1, 2) log|bold(Sigma)|$ and $bold(x)^top bold(Sigma)^(-1) bold(x)$

  #v(0.5em)

  *Define LDA discriminant score:*

  $ delta_k (bold(x)) = bold(x)^top bold(Sigma)^(-1) bold(mu)_k - frac(1, 2) bold(mu)_k^top bold(Sigma)^(-1) bold(mu)_k + log(pi_k) $

  This is *linear* in $bold(x)$
]

#slide[
  = Fisher's Approach
  
  *Alternative (equivalent) formulation:*
  
  Maximize ratio of between-group to within-group variance
  
  #v(1em)
  
  *For two groups:*
  
  $ "maximize" quad frac((overline(y)_1 - overline(y)_2)^2, s_1^2 + s_2^2) $
  
  where $y = bold(a)^top bold(x)$
  
  #v(1em)
  
  *Solution:* $bold(a) prop bold(Sigma)^(-1) (bold(mu)_1 - bold(mu)_2)$
]

#slide[
  = Quadratic Discriminant Analysis (QDA)
  
  *Relaxes equal covariance assumption*
  
  Each group $k$ has own covariance $bold(Sigma)_k$
  
  #v(1em)
  
  *When to use QDA:*
  - Groups have different variability patterns
  - Sufficient sample size
  - Linear boundaries inadequate
  
  #v(1em)
  
  *Result:* Quadratic (curved) decision boundaries
]

#slide[
  = QDA: Deriving the Discriminant Scores

  *Now allow different $bold(Sigma)_k$ for each group:*

  $ log(f_k (bold(x)) pi_k) = -frac(1, 2) log|bold(Sigma)_k| - frac(1, 2) (bold(x) - bold(mu)_k)^top bold(Sigma)_k^(-1) (bold(x) - bold(mu)_k) + log(pi_k) $

  #v(0.5em)

  *Key difference from LDA:*

  Cannot drop $log|bold(Sigma)_k|$ (varies by group)

  Cannot drop $bold(x)^top bold(Sigma)_k^(-1) bold(x)$ (different $bold(Sigma)_k$ for each group)

  #v(0.5em)

  *Define QDA discriminant score:*

  $ delta_k (bold(x)) = -frac(1, 2) log |bold(Sigma)_k| - frac(1, 2) (bold(x) - bold(mu)_k)^top bold(Sigma)_k^(-1) (bold(x) - bold(mu)_k) + log(pi_k) $

  #v(0.5em)

  This is *quadratic* in $bold(x)$, producing curved decision boundaries
]

#slide[
  = LDA vs QDA Trade-offs
  
  #table(
    columns: (1.5fr, 1fr, 1fr),
    align: (left, center, center),
    [*Criterion*], [*LDA*], [*QDA*],
    [Parameters], [Fewer], [More],
    [Sample size need], [Smaller], [Larger],
    [Decision boundaries], [Linear], [Curved],
    [Interpretability], [Simpler], [Complex],
    [Overfitting risk], [Lower], [Higher]
  )
  
  #v(1em)
  
  *Rule of thumb:* Start with LDA, move to QDA if needed
]

#slide[
  = Analysis Workflow
  
  *Step 1: Data Preparation*
  - Feature selection (avoid multicollinearity)
  - Standardization (equal scales)
  - Stratified train-test split
  
  *Step 2: Assumption Checking*
  - Multivariate normality (Q-Q plots, tests)
  - Equal covariances (Box's M test)
  - Multicollinearity (VIF)
  
  *Step 3: Model Fitting*
  - Fit LDA and/or QDA
  - Extract discriminant functions
]

#slide[
  = Analysis Workflow (cont.)
  
  *Step 4: Interpretation*
  - Examine discriminant coefficients
  - Identify key separating variables
  - Calculate group means on functions
  
  *Step 5: Validation*
  - Test set accuracy
  - Confusion matrix
  - Cross-validation
  - ROC curves and AUC
  - Visualize decision boundaries
]

#slide[
  = Python Implementation
  
  ```python
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  
  # Prepare data
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, stratify=y, random_state=42
  )
  
  # Standardize
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # Fit LDA
  lda = LinearDiscriminantAnalysis()
  lda.fit(X_train_scaled, y_train)
  
  # Predict
  y_pred = lda.predict(X_test_scaled)
  accuracy = lda.score(X_test_scaled, y_test)
  ```
]

#slide[
  = Marketing Example: Setup
  
  *Business Problem:*
  
  E-commerce with 1,200 customers, 3 segments for targeting
  
  #v(1em)
  
  *Three Segments:*
  - High-Value (30%): Premium customers
  - Loyal (40%): Regular customers
  - Occasional (30%): Infrequent buyers
  
  #v(1em)
  
  *Eight Behavioral Metrics:*
  
  Purchase frequency, order value, browsing time, cart abandonment, email open rate, loyalty points, support tickets, social engagement
]

#slide[
  = Marketing Example: Results
  
  *Discriminant Functions:*
  
  - *LD1 (95.8%):* Overall customer value
    - Drivers: frequency, loyalty points, order value
    - Separates High-Value from Occasional
  
  - *LD2 (4.2%):* Order size patterns
    - Drivers: order value, browsing time
  
  #v(1em)
  
  *Performance:*
  - LDA: 99.9% accuracy
  - QDA: 100.0% accuracy
  
  *Recommendation:* Use LDA (simpler, equally effective)
]

#slide[
  = Business Insights
  
  *High-Value:* High frequency, strong engagement, premium retention strategy
  
  *Loyal:* Moderate metrics, upselling and cross-selling focus
  
  *Occasional:* Low frequency, high abandonment, re-engagement campaigns
  
  #v(1em)
  
  *Applications:*
  - Auto-classify new customers (2-3 months)
  - Monitor segment migration
  - Optimize marketing ROI
  - Personalize campaigns
]

#slide[
  = Advanced Topics
  
  *Variable Selection:*
  - Stepwise methods (forward/backward)
  - Regularized DA (RDA)
  - Penalized LDA
  
  *Imbalanced Classes:*
  - Adjust prior probabilities
  - Oversampling (SMOTE)
  - Undersampling
  
  *Diagnostics:*
  - Wilks' Lambda
  - Canonical correlation
]

#slide[
  = Comparison with Other Methods
  
  #table(
    columns: (1.5fr, 2fr),
    align: (left, left),
    [*Method*], [*Best For*],
    [Logistic Regression], [Binary outcomes, no normality assumption],
    [SVM], [Non-linear boundaries, no assumptions],
    [Random Forest], [Non-linear, robust to outliers],
    [Discriminant Analysis], [Interpretability, understanding differences]
  )
]

#slide[
  = Common Pitfalls
  
  *Mistakes to Avoid:*
  - Ignoring assumptions (normality, equal covariances)
  - Not checking for outliers
  - Overfitting (too many predictors)
  - Evaluating only on training data
  - Ignoring class imbalance
  - Using correlated predictors
]

#slide[
  = Best Practices
  
  *Data Quality:*
  - Handle missing data
  - Screen for outliers
  - Verify data integrity
  
  *Model Selection:*
  - Start with LDA baseline
  - Use cross-validation
  - Report multiple metrics
  
  *Validation:*
  - Independent test data
  - Monitor over time
  - Update as needed
]

#slide[
  = Key Takeaways
  
  *Core Value:*
  - Not just prediction, but *understanding* group differences
  - Interpretable discriminant functions
  - Probabilistic classification confidence
  
  #v(1em)
  
  *When to Use:*
  - Labeled training data
  - Need interpretability
  - Moderate dimensionality
  - Approximate multivariate normality
  
  #v(1em)
  
  *Decision: LDA vs QDA*
  
  Start simple (LDA), add complexity (QDA) only if justified
]

#slide[
  = Hands-On Learning
  
  *Interactive Notebook:*
  
  `ch5_guiding_example/marketing_discriminant_analysis.ipynb`
  
  #v(1em)
  
  *Complete workflow:*
  1. Data generation (reproducible)
  2. Exploratory analysis
  3. LDA implementation
  4. QDA comparison
  5. Decision boundaries
  6. Performance evaluation
  
  #v(1em)
  
  *Experiment with different splits, features, priors!*
]

#slide[
  #v(3em)
  
  #align(center)[
    #text(size: 36pt, weight: "bold")[Questions?]
    
    #v(2em)
    
    #text(size: 18pt, style: "italic")[
      "The goal is to turn data into information,
      
      and information into insight."
      
      - Carly Fiorina
    ]
    
    #v(2em)
    
    #text(size: 16pt)[
      MA2003B - Multivariate Methods
      
      Dr. Juliho Castillo
      
      Tecnologico de Monterrey
    ]
  ]
]


