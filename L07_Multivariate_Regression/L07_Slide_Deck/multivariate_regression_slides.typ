// Multivariate Regression Presentation using Touying
#import "@preview/touying:0.5.3": *
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Multivariate Regression],
    subtitle: [Advanced Methods for Multivariate Analysis],
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
)

#title-slide()

// Outline
#slide[
  = Today's Agenda

  + Logistic Regression Model
  + Inferences for Variances and Covariance Matrices
  + Inferences for a Vector of Means
  + MANOVA (Multivariate Analysis of Variance)
  + Canonical Correlation Analysis
  + Factor Analysis with Regression
  + Programming and Commercial Systems
]

// Section 1: Logistic Regression
#slide[
  = Logistic Regression

  Moving Beyond Linear Regression
]

#slide[
  = When Linear Regression Fails

  *Problem:* Binary outcomes (Yes/No, Success/Failure, 0/1)

  Linear regression assumptions violated:
  - Response not continuous
  - Errors not normal
  - Predictions can exceed [0,1]
]

#slide[
  = Logistic Regression Solution

  *Key Idea:* Model the probability of success

  $ P(Y = 1 | X) = p(X) $

  where $0 <= p(X) <= 1$
]

#slide[
  = The Logit Transformation

  *Logit (Log-Odds):*

  $ "logit"(p) = log(p / (1-p)) = beta_0 + beta_1 X_1 + dots + beta_p X_p $
]

#slide[
  = The Logit Transformation

  *Properties:*
  - Maps [0,1] to $(-infinity, +infinity)$
  - Linear in parameters
  - Interpretable as log-odds ratio
]

#slide[
  = The Logistic Function

  *Inverse Logit:*

  $ p(X) = (e^(beta_0 + beta_1 X_1 + dots + beta_p X_p)) / (1 + e^(beta_0 + beta_1 X_1 + dots + beta_p X_p)) $

  Also written as:

  $ p(X) = 1 / (1 + e^(-(beta_0 + beta_1 X_1 + dots + beta_p X_p))) $
]

#slide[
  = Why Not Ordinary Least Squares?

  *Problems with OLS for Binary Response:*
  - Predicted probabilities can be negative or exceed 1
  - Errors follow Bernoulli distribution, not Normal
  - Heteroscedastic errors
  - Violates fundamental assumptions
]

#slide[
  = Maximum Likelihood Estimation

  *Bernoulli Distribution:*

  $ P(Y_i = y_i | X_i) = p(X_i)^(y_i) (1 - p(X_i))^(1-y_i) $
]

#slide[
  = Maximum Likelihood Estimation

  *Log-Likelihood Function:*

  $ ell(beta) = sum_(i=1)^n [y_i log(p(X_i)) + (1-y_i) log(1-p(X_i))] $

  *Goal:* Find $beta$ that maximizes $ell(beta)$
]

#slide[
  = Interpreting Coefficients

  *Coefficient $beta_j$:*
  - One unit increase in $X_j$ changes log-odds by $beta_j$
  - Odds ratio: $e^(beta_j)$

  #alert[
    *Example:* If $beta_1 = 0.5$, then $e^(0.5) = 1.65$ means 65% increase in odds
  ]
]

#slide[
  = Model Fit and Diagnostics

  *Deviance:* Measures goodness of fit

  $ D = -2 log(cal(L)) $

  *Pseudo R-squared:* McFadden's $R^2$, Nagelkerke $R^2$
]

#slide[
  = Classification Performance

  *Confusion Matrix:*

  #table(
    columns: (1fr, 1fr, 1fr),
    align: center,
    stroke: 0.5pt,
    inset: 10pt,
    [], [*Predicted 0*], [*Predicted 1*],
    [*Actual 0*], [True Negative (TN)], [False Positive (FP)],
    [*Actual 1*], [False Negative (FN)], [True Positive (TP)],
  )
]

#slide[
  = Classification Metrics

  *Accuracy:* $("TP" + "TN") / n$

  *Sensitivity (Recall):* $"TP" / ("TP" + "FN")$

  *Specificity:* $"TN" / ("TN" + "FP")$

  *Precision:* $"TP" / ("TP" + "FP")$
]

// Section 2: Covariance Matrix Inference
#slide[
  = Inferences for Covariance Matrices

  Testing Variability Structure
]

#slide[
  = Why Test Covariance Matrices?

  *Applications:*
  - Homogeneity assumptions in MANOVA
  - Comparing variability between groups
  - Validating models
  - Quality control
]

#slide[
  = The Wishart Distribution

  *Multivariate Generalization of Chi-Square*

  If $X_1, dots, X_n tilde N_p(mu, Sigma)$, then:

  $ S tilde W_p(n-1, Sigma) $

  where $S = sum_(i=1)^n (X_i - macron(X))(X_i - macron(X))^T$
]

#slide[
  = Testing Single Covariance Matrix

  *Null Hypothesis:*

  $ H_0: Sigma = Sigma_0 $

  *Test Statistic:* Based on likelihood ratio

  $ Lambda = |S| / |Sigma_0| $
]

#slide[
  = Box's M Test

  *Testing Equality of Covariance Matrices*

  $ H_0: Sigma_1 = Sigma_2 = dots = Sigma_g $
]

#slide[
  = Box's M Test Statistic

  $ M = (n - g) log|S_"pooled"| - sum_(i=1)^g (n_i - 1) log|S_i| $

  where:
  - $S_i$ = covariance matrix for group $i$
  - $S_"pooled"$ = pooled covariance matrix
]

#slide[
  = Box's M Test Properties

  *Asymptotic Distribution:* Chi-square for large samples

  #alert[
    *Limitation:* Very sensitive to normality violations
  ]

  *Alternatives:* Permutation tests, robust methods
]

#slide[
  = Bartlett's Test for Univariate Data

  *Special Case:* Testing equality of variances (p=1)

  $ H_0: sigma_1^2 = sigma_2^2 = dots = sigma_g^2 $

  *Test Statistic:* Chi-square distributed
]

// Section 3: Inferences for Mean Vectors
#slide[
  = Inferences for a Vector of Means

  Multivariate Hypothesis Testing
]

#slide[
  = From t-test to Hotelling's T-squared

  *Univariate:* t-test for single mean

  $ t = (macron(x) - mu_0) / (s / sqrt(n)) $

  *Multivariate:* Hotelling's $T^2$ for mean vector
]

#slide[
  = Hotelling's T-squared Test

  *One-Sample Test:*

  $ H_0: bold(mu) = bold(mu)_0 $

  *Test Statistic:*

  $ T^2 = n(macron(bold(X)) - bold(mu)_0)^T S^(-1) (macron(bold(X)) - bold(mu)_0) $
]

#slide[
  = Distribution of T-squared

  *Transform to F Distribution:*

  $ F = ((n-p) T^2) / ((n-1) p) tilde F_(p, n-p) $

  where:
  - $p$ = number of variables
  - $n$ = sample size
]

#slide[
  = Two-Sample Hotelling's T-squared

  *Testing Difference Between Groups:*

  $ H_0: bold(mu)_1 = bold(mu)_2 $
]

#slide[
  = Two-Sample T-squared Statistic

  $ T^2 = ((n_1 n_2) / (n_1 + n_2)) (macron(bold(X))_1 - macron(bold(X))_2)^T S_"pooled"^(-1) (macron(bold(X))_1 - macron(bold(X))_2) $

  *F Transformation:*

  $ F = ((n_1 + n_2 - p - 1) T^2) / ((n_1 + n_2 - 2) p) tilde F_(p, n_1+n_2-p-1) $
]

#slide[
  = Confidence Region for Mean Vector

  *Multivariate Confidence Region:*

  Ellipsoid centered at $macron(bold(X))$

  $ n(bold(mu) - macron(bold(X)))^T S^(-1) (bold(mu) - macron(bold(X))) <= ((n-1)p) / (n-p) F_(alpha; p, n-p) $
]

#slide[
  = Simultaneous Confidence Intervals

  *Bonferroni Correction:*

  For $p$ variables, use $alpha / p$ for each interval

  *T-squared Intervals:* More efficient but wider than individual intervals
]

// Section 4: MANOVA
#slide[
  = MANOVA

  Multivariate Analysis of Variance
]

#slide[
  = What is MANOVA?

  *Extension of ANOVA to Multiple Dependent Variables*

  - ANOVA: One response variable
  - MANOVA: Multiple response variables simultaneously
]

#slide[
  = Why Use MANOVA?

  *Instead of Multiple ANOVAs:*

  + Controls Type I error rate
  + Accounts for correlations among responses
  + More powerful when responses related
  + Tests overall group effect
]

#slide[
  = MANOVA Model

  *One-Way MANOVA:*

  $ bold(Y)_(i j) = bold(mu) + bold(alpha)_i + bold(epsilon)_(i j) $

  where:
  - $bold(Y)_(i j)$ = response vector for observation $j$ in group $i$
  - $bold(mu)$ = overall mean vector
  - $bold(alpha)_i$ = group effect vector
  - $bold(epsilon)_(i j)$ = error vector
]

#slide[
  = MANOVA Assumptions

  + *Multivariate Normality:* Errors follow multivariate normal
  + *Independence:* Observations independent
  + *Homogeneity of Covariance:* Equal covariance matrices across groups
]

#slide[
  = Testing Assumptions

  *Multivariate Normality:*
  - Mardia's test
  - Q-Q plots for each variable

  *Homogeneity:* Box's M test
]

#slide[
  = MANOVA Matrices

  *Between-Groups Matrix (H):*

  $ bold(H) = sum_(i=1)^g n_i (macron(bold(Y))_i - macron(bold(Y)))(macron(bold(Y))_i - macron(bold(Y)))^T $

  *Within-Groups Matrix (E):*

  $ bold(E) = sum_(i=1)^g sum_(j=1)^(n_i) (bold(Y)_(i j) - macron(bold(Y))_i)(bold(Y)_(i j) - macron(bold(Y))_i)^T $
]

#slide[
  = Wilks' Lambda

  *Most Common Test Statistic:*

  $ Lambda = |bold(E)| / |bold(E) + bold(H)| $
]

#slide[
  = Wilks' Lambda Properties

  *Interpretation:*
  - Range: [0, 1]
  - Small values: Strong group differences
  - Lambda = 1: No group differences

  *Represents:* Proportion of total variance not explained by groups
]

#slide[
  = Other MANOVA Test Statistics

  *Pillai's Trace:*

  $ V = "tr"(bold(H)(bold(H) + bold(E))^(-1)) $

  *Hotelling-Lawley Trace:*

  $ U = "tr"(bold(H) bold(E)^(-1)) $

  *Roy's Largest Root:* Largest eigenvalue of $bold(H) bold(E)^(-1)$
]

#slide[
  = Choosing Test Statistic

  #table(
    columns: (1.5fr, 2fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 10pt,
    [*Statistic*], [*Best When*],
    [Wilks' Lambda], [General use (most common)],
    [Pillai's Trace], [Robust to violations],
    [Hotelling-Lawley], [Equal group sizes],
    [Roy's Root], [Group difference on one dimension],
  )
]

#slide[
  = Post-Hoc Tests in MANOVA

  *After Significant MANOVA:*

  + Univariate ANOVAs (with correction)
  + Discriminant analysis
  + Contrast tests for specific hypotheses
]

// Section 5: Canonical Correlation
#slide[
  = Canonical Correlation Analysis

  Relating Two Sets of Variables
]

#slide[
  = What is Canonical Correlation?

  *Purpose:* Find maximum correlation between linear combinations of two sets of variables

  - Set 1: $X_1, X_2, dots, X_p$
  - Set 2: $Y_1, Y_2, dots, Y_q$
]

#slide[
  = Canonical Correlation vs. Other Methods

  #table(
    columns: (1.5fr, 1fr, 1fr),
    align: (left, center, center),
    stroke: 0.5pt,
    inset: 8pt,
    [*Method*], [*Set 1*], [*Set 2*],
    [Correlation], [1 variable], [1 variable],
    [Multiple Regression], [Multiple], [1 variable],
    [Canonical Correlation], [Multiple], [Multiple],
  )
]

#slide[
  = Canonical Variates

  *First Canonical Variate Pair:*

  $ U_1 = a_(11) X_1 + a_(12) X_2 + dots + a_(1 p) X_p $
  $ V_1 = b_(11) Y_1 + b_(12) Y_2 + dots + b_(1 q) Y_q $

  such that $"cor"(U_1, V_1)$ is maximized
]

#slide[
  = Number of Canonical Correlations

  *How Many Pairs?*

  $ k = min(p, q) $

  Each subsequent pair:
  - Uncorrelated with previous pairs
  - Maximizes remaining correlation
]

#slide[
  = Canonical Correlation Coefficients

  *Ordering:*

  $ rho_1 >= rho_2 >= dots >= rho_k >= 0 $

  where $rho_i$ is the $i$-th canonical correlation
]

#slide[
  = Testing Significance

  *Test All Correlations:*

  $ H_0: rho_1 = rho_2 = dots = rho_k = 0 $

  *Test Remaining Correlations:*

  $ H_0: rho_(m+1) = dots = rho_k = 0 $
]

#slide[
  = Wilks' Lambda for Canonical Correlation

  $ Lambda = product_(i=1)^k (1 - rho_i^2) $

  Approximate chi-square distribution for testing
]

#slide[
  = Canonical Loadings

  *Structure Coefficients:*

  Correlation between original variables and canonical variates

  - Help interpret meaning of canonical variates
  - More stable than canonical weights
]

#slide[
  = Redundancy Analysis

  *Proportion of Variance Explained:*

  How much variance in one set is explained by the other set through canonical variates

  $ "Redundancy" = (1/p) sum_(j=1)^p R^2_(X_j, V_1) $
]

#slide[
  = Interpreting Canonical Correlations

  + *Examine significance:* Are correlations statistically significant?
  + *Check magnitude:* Are correlations practically meaningful?
  + *Interpret loadings:* What do canonical variates represent?
  + *Assess redundancy:* How much variance explained?
]

// Section 6: Factor Analysis with Regression
#slide[
  = Factor Analysis with Regression

  Combining Dimension Reduction and Prediction
]

#slide[
  = The Multicollinearity Problem

  *Issue:* Highly correlated predictors in regression

  *Consequences:*
  - Unstable coefficient estimates
  - Large standard errors
  - Difficult interpretation
  - Poor prediction in new samples
]

#slide[
  = Factor-Based Regression Solution

  *Strategy:*

  + Extract factors from correlated predictors
  + Use factor scores as predictors
  + Fit regression with orthogonal factors
]

#slide[
  = Factor-Based Regression Workflow

  + *Factor Analysis:* Extract factors from $X$ variables
  + *Compute Factor Scores:* For each observation
  + *Regression:* Predict $Y$ using factor scores
  + *Interpretation:* Results in terms of factors
]

#slide[
  = Benefits of Factor-Based Regression

  *Advantages:*
  - Reduces multicollinearity (orthogonal factors)
  - Dimensionality reduction (fewer predictors)
  - Conceptual interpretation (latent constructs)
  - More stable estimates
]

#slide[
  = Comparing Approaches

  #table(
    columns: (1.5fr, 1fr, 1fr),
    align: (left, center, center),
    stroke: 0.5pt,
    inset: 8pt,
    [*Aspect*], [*Direct Regression*], [*Factor Regression*],
    [Multicollinearity], [Problem], [Eliminated],
    [Interpretation], [Original variables], [Latent factors],
    [Predictors], [Many], [Few],
    [Variance explained], [Higher], [May be lower],
  )
]

#slide[
  = Principal Components Regression

  *Alternative Approach:*

  Use PCA instead of factor analysis

  *Difference:*
  - PCA: Explains total variance
  - FA: Explains common variance (removes unique variance)
]

#slide[
  = Other Methods for Multicollinearity

  *Ridge Regression:* Shrinks coefficients toward zero

  *Lasso:* Variable selection via L1 penalty

  *Partial Least Squares:* Finds components that predict Y well
]

#slide[
  = Cautions and Limitations

  *Factor-Based Regression Limitations:*
  - Factor extraction somewhat subjective
  - Results depend on specific sample
  - Prediction requires computing factor scores with same loadings
  - May lose some predictive information
]

// Section 7: Software
#slide[
  = Programming and Commercial Systems

  Implementing Multivariate Methods
]

#slide[
  = Python for Multivariate Analysis

  *Key Libraries:*
  - `statsmodels`: Statistical models and tests
  - `scikit-learn`: Machine learning algorithms
  - `numpy` / `scipy`: Numerical computations
  - `pandas`: Data manipulation
]

#slide[
  = Python: Logistic Regression

  ```python
  from sklearn.linear_model import LogisticRegression

  model = LogisticRegression()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  probabilities = model.predict_proba(X_test)
  ```
]

#slide[
  = Python: Hotelling's T-squared

  ```python
  from scipy.stats import chi2
  import numpy as np

  # Compute T-squared statistic
  diff = mean1 - mean2
  S_pooled_inv = np.linalg.inv(S_pooled)
  T2 = (n1 * n2) / (n1 + n2) * diff.T @ S_pooled_inv @ diff

  # Transform to F
  p = len(mean1)
  F_stat = ((n1 + n2 - p - 1) * T2) / ((n1 + n2 - 2) * p)
  ```
]

#slide[
  = Python: MANOVA

  ```python
  from statsmodels.multivariate.manova import MANOVA

  # Fit MANOVA model
  manova = MANOVA.from_formula(
      'Y1 + Y2 + Y3 ~ Group',
      data=df
  )

  # Test results
  print(manova.mv_test())
  ```
]

#slide[
  = Python: Canonical Correlation

  ```python
  from sklearn.cross_decomposition import CCA

  # Canonical correlation analysis
  cca = CCA(n_components=2)
  cca.fit(X_set, Y_set)

  # Transform to canonical variates
  X_c, Y_c = cca.transform(X_set, Y_set)

  # Canonical correlations
  correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                  for i in range(2)]
  ```
]

#slide[
  = R for Multivariate Analysis

  *Key Packages:*
  - `stats`: Base statistical functions
  - `MASS`: Advanced statistical methods
  - `car`: Companion to Applied Regression
  - `vegan`: Multivariate analysis
]

#slide[
  = R: MANOVA Example

  ```r
  # Fit MANOVA
  model <- manova(cbind(Y1, Y2, Y3) ~ Group, data = df)

  # Test results
  summary(model, test = "Wilks")
  summary(model, test = "Pillai")

  # Follow-up univariate tests
  summary.aov(model)
  ```
]

#slide[
  = Commercial Software: SPSS

  *GUI-Based Analysis:*
  - Analyze > General Linear Model > Multivariate
  - Analyze > Regression > Binary Logistic
  - Analyze > Correlate > Canonical Correlation

  *Syntax:* Also supports command syntax for reproducibility
]

#slide[
  = Commercial Software: SAS

  *Key Procedures:*
  - `PROC LOGISTIC`: Logistic regression
  - `PROC GLM`: General linear models (MANOVA)
  - `PROC CANCORR`: Canonical correlation
  - `PROC FACTOR`: Factor analysis
]

#slide[
  = Software Comparison

  #table(
    columns: (1fr, 1.5fr, 1.5fr),
    align: (left, left, left),
    stroke: 0.5pt,
    inset: 8pt,
    [*Software*], [*Strengths*], [*Limitations*],
    [Python], [Free, flexible, ML integration], [Statistical testing less developed],
    [R], [Free, comprehensive stats], [Steeper learning curve],
    [SPSS], [GUI, easy to learn], [Expensive, less flexible],
    [SAS], [Enterprise, comprehensive], [Very expensive, complex],
  )
]

#slide[
  = Choosing Software

  *Considerations:*
  - Cost (free vs. commercial)
  - Learning curve
  - Specific methods needed
  - Integration with workflow
  - Reproducibility requirements
  - Team expertise
]

#slide[
  = Best Practices: Code Documentation

  *Essential Elements:*
  - Comment your code clearly
  - Document data preprocessing steps
  - Record package versions
  - Save random seeds for reproducibility
  - Version control (Git)
]

#slide[
  = Best Practices: Workflow

  + *Data Cleaning:* Handle missing values, outliers
  + *Exploratory Analysis:* Visualize distributions
  + *Check Assumptions:* Test before analysis
  + *Run Analysis:* Use appropriate methods
  + *Validate Results:* Cross-validation, diagnostics
  + *Document:* Clear reporting
]

// Summary and Resources
#slide[
  = Key Takeaways: Models

  *Logistic Regression:*
  - Use for binary outcomes
  - Maximum likelihood estimation
  - Interpret via odds ratios
]

#slide[
  = Key Takeaways: Inference

  *Covariance Matrix Tests:*
  - Box's M test for equality
  - Wishart distribution foundation

  *Mean Vector Tests:*
  - Hotelling's T-squared generalizes t-test
  - Confidence regions are ellipsoids
]

#slide[
  = Key Takeaways: Advanced Methods

  *MANOVA:*
  - Multiple response variables simultaneously
  - Wilks' Lambda most common test
  - Controls Type I error

  *Canonical Correlation:*
  - Relates two variable sets
  - Multiple correlation pairs
]

#slide[
  = Key Takeaways: Applications

  *Factor-Based Regression:*
  - Addresses multicollinearity
  - Dimension reduction
  - Interpretable factors

  *Software:*
  - Python: scikit-learn, statsmodels
  - R: stats, MASS
  - Commercial: SPSS, SAS
]

#slide[
  = Common Pitfalls to Avoid

  + Using logistic regression without checking convergence
  + Ignoring multicollinearity in regression
  + Not checking MANOVA assumptions (Box's M)
  + Over-interpreting weak canonical correlations
  + Using too many factors in factor-based regression
]

#slide[
  = Method Selection Guide

  #table(
    columns: (1.5fr, 2fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 8pt,
    [*Situation*], [*Method*],
    [Binary outcome], [Logistic regression],
    [Multiple groups, multiple responses], [MANOVA],
    [Relate two variable sets], [Canonical correlation],
    [Multicollinear predictors], [Factor/PCA regression],
  )
]

#slide[
  = Recommended Resources: Books

  *Textbooks:*
  - Agresti (2018) - Introduction to Categorical Data Analysis
  - Johnson & Wichern (2007) - Applied Multivariate Statistical Analysis
  - Rencher & Christensen (2012) - Methods of Multivariate Analysis
]

#slide[
  = Recommended Resources: Online

  *StatQuest YouTube Channel:*

  + *Logistic Regression:* \
    #link("https://www.youtube.com/watch?v=yIYKR4sgzI8")

  + *MANOVA Concepts:* Search "MANOVA StatQuest"

  + *PCA (for PCR):* \
    #link("https://www.youtube.com/watch?v=FgakZw6K1QQ")
]

#slide[
  = Recommended Resources: Software

  *Documentation:*
  - Scikit-learn: #link("https://scikit-learn.org")
  - Statsmodels: #link("https://www.statsmodels.org")
  - R Documentation: #link("https://www.rdocumentation.org")
]

// Final slides
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
  - Prepare for E07 quiz
]

#slide[
  = Next Steps: Preparation for Evaluation

  *Key Topics to Master:*
  - Logistic regression: logit transformation, MLE, interpretation
  - Hotelling's T-squared: computation and F transformation
  - MANOVA: assumptions, Wilks' Lambda, interpretation
  - Canonical correlation: number of pairs, loadings, significance
  - Factor-based regression: workflow, benefits, limitations
  - Software implementation: Python and R basics
]

#slide[
  = Integration with Previous Topics

  *Building on Earlier Concepts:*
  - Factor Analysis (L04) → Factor-based regression
  - Discriminant Analysis (L05) → MANOVA post-hoc
  - PCA principles → Principal components regression

  *Comprehensive Framework:*
  All methods part of the multivariate toolkit
]
