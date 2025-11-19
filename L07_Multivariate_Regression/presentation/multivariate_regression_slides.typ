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

// Case Study Introduction
#slide[
  = Case Study: Healthcare Risk Assessment

  *Companion Example Throughout This Presentation*

  A hospital study analyzing cardiovascular disease (CVD) risk using multivariate methods to:
  - Predict high-risk patients from lifestyle and physiological factors
  - Compare health profiles between risk groups
  - Evaluate the effectiveness of lifestyle interventions
  - Understand complex relationships between multiple health variables

  Real dataset with 1,000 patients demonstrating practical applications of each technique
]

#slide[
  = The Research Question

  *Scenario:* Hospital evaluating cardiovascular disease (CVD) risk

  *Dataset:*
  - 1,000 patients
  - 13 predictor variables
  - Multiple health outcomes
  - Treatment intervention study
]

#slide[
  = Variables in Our Study

  *Lifestyle Factors:*
  - *Age:* Years lived, affects baseline health risk
  - *BMI:* Body Mass Index, ratio of weight to height indicating body composition
  - *Exercise hours/week:* Physical activity level, linked to cardiovascular health
  - *Smoking years:* Duration of tobacco use, major risk factor for multiple diseases
  - *Alcohol consumption:* Frequency/amount of alcohol intake, impacts liver and heart health
  - *Stress score:* Quantified psychological stress level, affects blood pressure and hormones
  - *Sleep hours:* Nightly sleep duration, influences metabolic and cardiovascular function

  *Physiological Measurements:*
  - *Blood pressure (systolic/diastolic):* Force of blood against artery walls (peak/resting)
  - *Cholesterol:* Blood lipid levels, excess increases arterial plaque buildup
  - *Glucose:* Blood sugar level, indicator of diabetes risk and metabolic function
  - *Triglycerides:* Fat molecules in blood, high levels increase heart disease risk
  - *HDL:* "Good cholesterol" that removes harmful lipids from bloodstream
]

#slide[
  = Research Objectives

  + *Predict* CVD risk from patient characteristics
  + *Compare* health profiles between risk groups
  + *Evaluate* lifestyle intervention effectiveness
  + *Understand* relationships between lifestyle and physiology
  + *Validate* assumptions for multivariate tests
]

// Section 1: Logistic Regression
#slide[
  = Logistic Regression

  Moving Beyond Linear Regression

  *What is Logistic Regression?*

  A statistical method for modeling binary outcomes (Yes/No, Success/Failure, 0/1) using predictor variables

  *Key Features:*
  - Predicts probabilities (0 to 1) rather than continuous values
  - Uses the logistic (sigmoid) function to model the probability of an event
  - Estimates coefficients via maximum likelihood (not least squares)
  - Widely used in classification problems: medical diagnosis, credit scoring, marketing

  *When to Use:* When your outcome variable is categorical (especially binary)
]

#slide[
  = When Linear Regression Fails

  *Problem:* Binary outcomes (Yes/No, Success/Failure, 0/1)

  Using ordinary least squares (OLS) for binary responses creates fundamental problems:

  *Prediction Issues:*
  - Predicted probabilities can be negative or exceed 1 (nonsensical values)
  - No mechanism to constrain predictions to [0, 1]

  *Assumption Violations:*
  - Response is not continuous (violates normality assumption)
  - Errors follow Bernoulli distribution, not Normal distribution
  - Heteroscedastic errors (variance depends on X: $"Var"(Y|X) = p(X)(1-p(X))$)
  - Non-constant variance violates homoscedasticity assumption

  Linear regression is mathematically possible but statistically inappropriate for binary data.
]

#slide[
  = Logistic Regression Solution

  Our goal is to find a function that maps predictor values to probabilities while staying within [0, 1]. Linear regression fails here because predictions can fall outside this valid probability range.
]

#slide[
  Given a binary outcome variable $Y$ (taking values 0 or 1 for failure or success) and a vector of predictor variables $X = (X_1, X_2, dots, X_p)$, we model the probability of success as:

  $ P(Y = 1 | X) = p(X) $

  where $0 <= p(X) <= 1$ represents the probability that $Y = 1$ given the values of $X$. This approach ensures our predictions always remain valid probabilities.
]

#slide[
  = The Logit Transformation

  To connect probabilities (bounded between 0 and 1) to a linear combination of predictors (unbounded), we use the *logit transformation*, also known as the *log-odds*.

  Recall that $p(X) = P(Y = 1 | X)$ is our probability of success. The odds of success are $p(X) / (1-p(X))$, and taking the natural logarithm gives us:

  $ "logit"(p(X)) = log(p(X) / (1-p(X))) = beta_0 + beta_1 X_1 + dots + beta_p X_p $

  where:
  - $p(X)$ is the same probability function defined in the previous slide
  - $beta_0, beta_1, dots, beta_p$ are coefficients to be estimated
  - The logit maps probabilities from [0, 1] to $(-infinity, +infinity)$
  - This transformation makes the model linear in the parameters
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

  *Inverse Logit:* Solving for $p(X)$ from the logit equation gives the logistic function.

  *Notation:* $X = (X_1, X_2, dots, X_p)$ is the vector of $p$ predictor variables. We write $p(X)$ to mean the probability depends on all predictors.

  Define the linear combination: $eta = beta_0 + beta_1 X_1 + beta_2 X_2 + dots + beta_p X_p$

  Then the logistic function is:

  $ p(X) = (e^eta) / (1 + e^eta) = 1 / (1 + e^(-eta)) $

  This is the *logistic* or *sigmoid* function, guaranteeing $p(X) in [0, 1]$ for any predictor values.
]

#slide[
  = Maximum Likelihood Estimation

  Since we cannot use ordinary least squares, we estimate coefficients using *Maximum Likelihood Estimation (MLE)*.

  *Our Data:* We have collected $n$ observations. Each observation consists of:
  - An outcome: $y_i in {0, 1}$ (e.g., $y_i = 1$ means "has disease", $y_i = 0$ means "no disease")
  - Predictor values: $x_(i 1), x_(i 2), dots, x_(i p)$ (the values of $p$ predictors for person $i$)

  *Unknown Parameters:* We need to estimate $(p + 1)$ coefficients
  - $beta_0$ = intercept coefficient
  - $beta_1, beta_2, dots, beta_p$ = coefficients for the $p$ predictors

  We write $bold(beta) = (beta_0, beta_1, dots, beta_p)^T$ for the entire vector of unknown parameters.
]

#slide[
  = Maximum Likelihood Estimation

  *The Probability Model:* For each observation $i$, the probability of success depends on:
  1. The predictor values for that observation: $x_(i 1), x_(i 2), dots, x_(i p)$
  2. The unknown parameters: $beta_0, beta_1, dots, beta_p$

  Define the linear combination: $eta_i = beta_0 + beta_1 x_(i 1) + beta_2 x_(i 2) + dots + beta_p x_(i p)$

  The probability is computed using the logistic function:

  $ P(Y_i = 1 | x_(i 1), dots, x_(i p); beta_0, dots, beta_p) = 1 / (1 + e^(-eta_i)) $

  We write this as $pi_i (bold(beta))$ for brevity, emphasizing it depends on the parameters $bold(beta)$.

  *Important:* Each $pi_i (bold(beta))$ depends on BOTH the observed predictors for observation $i$ AND the unknown parameters $bold(beta)$.
]

#slide[
  = Maximum Likelihood Estimation

  *Probability of Observing Outcome $y_i$:* Given parameters $bold(beta)$, the probability of observing the actual outcome $y_i$ for observation $i$ is:

  $ P(y_i | bold(x)_i, bold(beta)) = [pi_i (bold(beta))]^(y_i) [1 - pi_i (bold(beta))]^(1-y_i) $

  This formula evaluates to:
  - $pi_i (bold(beta))$ when the observed outcome is $y_i = 1$ (success)
  - $1 - pi_i (bold(beta))$ when the observed outcome is $y_i = 0$ (failure)

  *Key Idea of MLE:* Find the parameter values $bold(beta)$ that make the observed data $(y_1, y_2, dots, y_n)$ most probable.
]

#slide[
  = Maximum Likelihood Estimation

  *Likelihood Function:* Assuming observations are independent, the probability of observing ALL our data is:

  $ L(bold(beta)) = product_(i=1)^n P(y_i | bold(x)_i, bold(beta)) = product_(i=1)^n [pi_i (bold(beta))]^(y_i) [1 - pi_i (bold(beta))]^(1-y_i) $

  *Log-Likelihood Function:* Taking natural logarithm (easier to maximize):

  $ ell(bold(beta)) = sum_(i=1)^n [y_i log(pi_i (bold(beta))) + (1-y_i) log(1-pi_i (bold(beta)))] $

  *Goal:* Find $bold(beta)^*$ that maximizes $ell(bold(beta))$. This requires numerical optimization (Newton-Raphson, gradient descent, etc.) since no closed-form solution exists.
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

  *Deviance:*
  - An analogue to _Sum of Squared Errors_ for generalized linear models.
  - Measures discrepancy via log-likelihood ($cal(L)$) compared to a perfect ("saturated") model.
  - $ D = 2(log(cal(L)_"saturated") - log(cal(L)_"model")) $
  - *Lower values indicate a better fit.*

  *Pseudo R-squared:*
  - An analogue to $R^2$ that measures improvement over a null (intercept-only) model.
  - *Higher values indicate a better fit.*
  - McFadden's $R^2$: $ R^2_"McF" = 1 - (log(cal(L)_"model")) / (log(cal(L)_"null")) $
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

// Case Study Application: Logistic Regression
#slide[
  = Case Study: Predicting CVD Risk

  *Application of Logistic Regression*

  Objective: Predict high CVD risk (0/1) from 13 patient characteristics
]

#slide[
  = CVD Prediction: Model Setup

  *Predictors (13 variables):*
  - Demographics: age, BMI
  - Lifestyle: exercise, smoking, alcohol, stress, sleep
  - Physiology: BP, cholesterol, glucose, triglycerides, HDL

  *Outcome:* CVD risk high (binary: 0 = low risk, 1 = high risk)

  *Data split:* 70% training (n=700), 30% testing (n=300)
]

#slide[
  = Top Risk Factors: Odds Ratios

  #table(
    columns: (2fr, 1fr, 1.5fr),
    align: (left, center, left),
    stroke: 0.5pt,
    inset: 10pt,
    [*Predictor*], [*Odds Ratio*], [*Interpretation*],
    [Exercise hours/week], [0.72], [28% lower odds per hour],
    [Stress score], [1.25], [25% higher odds per point],
    [Sleep hours], [0.80], [20% lower odds per hour],
    [BMI], [1.19], [19% higher odds per unit],
  )

  *All significant predictors contribute to risk assessment*
]

#slide[
  = CVD Prediction: Model Performance

  *Confusion Matrix (Test Set):*
  #table(
    columns: (1.5fr, 1fr, 1fr),
    align: center,
    stroke: 0.5pt,
    inset: 10pt,
    [], [*Pred Low*], [*Pred High*],
    [*Actual Low*], [106], [44],
    [*Actual High*], [42], [108],
  )

  *Metrics:*
  - Accuracy: 71%
  - AUC-ROC: 0.77
  - Balanced precision/recall
]

#slide[
  = Key Insights: CVD Prediction

  + Exercise is the strongest protective factor (OR = 0.72)
  + Stress significantly increases risk (OR = 1.25)
  + Model achieves good discrimination (AUC = 0.77)
  + Can identify high-risk patients for intervention

  #alert[
    *Clinical Value:* Early identification enables preventive care
  ]
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

// Case Study Application: Box's M Test
#slide[
  = Case Study: Validating MANOVA Assumptions

  *Application of Box's M Test*

  Question: Are covariance matrices equal between treatment groups (MANOVA assumption)?
]

#slide[
  = Box's M Test: Setup

  *Testing Homogeneity of Covariances:*

  $ H_0: bold(Sigma)_"Control" = bold(Sigma)_"Intervention" $

  *Variables (p = 4):*
  - Systolic BP, Diastolic BP
  - Cholesterol, Glucose

  *Groups:*
  - Control: n = 479
  - Intervention: n = 521

  *Why test?* MANOVA assumes equal covariance matrices across groups
]

#slide[
  = Box's M Test: Results

  *Test Statistic:*
  $ M = 8.49 $

  *Degrees of Freedom:* 10

  *Interpretation:* $M < 30$ (rule of thumb)

  #alert[
    *Conclusion:* Covariance matrices are approximately equal. MANOVA assumption satisfied.
  ]

  *Implication:* Our MANOVA results are valid and trustworthy
]

#slide[
  = Key Insights: Assumption Testing

  + Box's M test validates MANOVA assumptions
  + Equal covariances ensure valid inference
  + Small M statistic (8.49) indicates homogeneity
  + Treatment groups have similar variability patterns

  *Methodological importance:* Always check assumptions before interpreting results
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

// Case Study Application: Hotelling's T-squared
#slide[
  = Case Study: Comparing Risk Groups

  *Application of Hotelling's T-squared*

  Question: Do high-risk and low-risk CVD patients differ in their multivariate health profile?
]

#slide[
  = Health Profile Comparison: Setup

  *Two Groups:*
  - Low risk: n = 500
  - High risk: n = 500

  *Variables (p = 6):*
  - Systolic BP, Diastolic BP
  - Cholesterol, Glucose
  - Triglycerides, HDL

  *Goal:* Single omnibus test for all 6 variables simultaneously
]

#slide[
  = Mean Differences by Risk Group

  #table(
    columns: (2fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    stroke: 0.5pt,
    inset: 8pt,
    [*Variable*], [*Low Risk*], [*High Risk*], [*Difference*],
    [Systolic BP], [123.8], [131.1], [+7.3],
    [Diastolic BP], [78.1], [82.9], [+4.7],
    [Cholesterol], [184.1], [196.4], [+12.3],
    [Glucose], [110.0], [116.8], [+6.9],
    [Triglycerides], [133.6], [145.5], [+11.9],
    [HDL], [44.5], [41.2], [-3.3],
  )
]

#slide[
  = Hotelling's T-squared Results

  *Test Statistic:*
  $ T^2 = 228.65 $

  *F Transformation:*
  $ F = 37.92, quad "df" = (6, 993) $

  *P-value:* < 0.0001

  #alert[
    *Conclusion:* Strong evidence that high-risk and low-risk patients have significantly different health profiles
  ]
]

#slide[
  = Key Insights: Risk Group Differences

  + High-risk patients show higher values across all adverse markers
  + Largest differences: cholesterol (+12.3 mg/dL) and triglycerides (+11.9 mg/dL)
  + HDL (protective) is lower in high-risk group (-3.3 mg/dL)
  + Multivariate test accounts for correlations among measurements

  *Clinical significance:* Pattern of differences suggests metabolic syndrome
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

// Case Study Application: MANOVA
#slide[
  = Case Study: Treatment Intervention

  *Application of MANOVA*

  Question: Does a lifestyle intervention improve multiple health outcomes simultaneously?
]

#slide[
  = Treatment Intervention Study: Setup

  *Groups:*
  - Control: n = 479 (standard care)
  - Intervention: n = 521 (lifestyle program)

  *Outcomes (p = 4):*
  - Systolic BP
  - Diastolic BP
  - Cholesterol
  - Glucose

  *Why MANOVA?* Controls Type I error while testing all outcomes together
]

#slide[
  = MANOVA Results: Treatment Effect

  *Wilks' Lambda:* $Lambda = 0.889$

  *F Statistic:* $F = 31.05$, df = (4, 995)

  *P-value:* < 0.0001

  #alert[
    *Conclusion:* The intervention significantly improves health outcomes across the multivariate profile
  ]

  *Also significant:* Pillai's trace, Hotelling-Lawley, Roy's root (all p < 0.0001)
]

#slide[
  = Mean Improvements by Treatment Group

  #table(
    columns: (2fr, 1fr, 1fr, 1.2fr),
    align: (left, center, center, center),
    stroke: 0.5pt,
    inset: 8pt,
    [*Outcome*], [*Control*], [*Intervention*], [*Difference*],
    [Systolic BP], [130.8], [124.3], [-6.5#super[\*\*\*]],
    [Diastolic BP], [82.7], [78.5], [-4.2#super[\*\*\*]],
    [Cholesterol], [194.4], [186.5], [-7.9#super[\*\*\*]],
    [Glucose], [116.3], [110.7], [-5.6#super[\*\*\*]],
  )

  #super[\*\*\*] All differences significant at p < 0.001 in follow-up ANOVAs
]

#slide[
  = Key Insights: Intervention Effects

  + Intervention reduces all cardiovascular risk markers
  + Largest effect on blood pressure (-6.5 / -4.2 mmHg)
  + Clinically meaningful reductions in cholesterol and glucose
  + MANOVA provides single omnibus test (no Type I error inflation)

  *Clinical significance:* Comprehensive lifestyle changes yield broad health benefits
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

// Case Study Application: Canonical Correlation
#slide[
  = Case Study: Lifestyle vs. Physiology

  *Application of Canonical Correlation*

  Question: How do lifestyle factors relate to physiological health markers?
]

#slide[
  = Lifestyle-Physiology Relationship: Setup

  *Set 1 - Lifestyle Factors (p = 5):*
  - Exercise hours/week
  - Smoking years
  - Alcohol units/week
  - Stress score
  - Sleep hours

  *Set 2 - Physiological Markers (q = 6):*
  - Systolic BP, Diastolic BP
  - Cholesterol, Glucose
  - Triglycerides, HDL

  *Maximum pairs:* min(5, 6) = 5
]

#slide[
  = Canonical Correlations: Results

  #table(
    columns: (1.5fr, 1.5fr, 2fr),
    align: (center, center, left),
    stroke: 0.5pt,
    inset: 10pt,
    [*Pair*], [*Correlation*], [*Interpretation*],
    [1], [0.639], [Strong relationship],
    [2], [0.244], [Moderate relationship],
    [3-5], [< 0.12], [Weak relationships],
  )

  *Focus on first canonical correlation (r = 0.639)*
]

#slide[
  = First Canonical Variate: Lifestyle

  *Canonical Loadings (Structure Coefficients):*

  #table(
    columns: (2fr, 1fr),
    align: (left, center),
    stroke: 0.5pt,
    inset: 10pt,
    [*Variable*], [*Loading*],
    [Exercise hours], [+0.65],
    [Stress score], [-0.53],
    [Alcohol units], [-0.37],
    [Sleep hours], [+0.33],
    [Smoking years], [-0.27],
  )

  *Interpretation:* Healthy lifestyle pattern (more exercise, less stress)
]

#slide[
  = First Canonical Variate: Physiology

  *Canonical Loadings:*

  #table(
    columns: (2fr, 1fr),
    align: (left, center),
    stroke: 0.5pt,
    inset: 10pt,
    [*Variable*], [*Loading*],
    [Diastolic BP], [-0.70],
    [Systolic BP], [-0.68],
    [Cholesterol], [-0.65],
    [HDL], [+0.65],
    [Glucose], [-0.61],
    [Triglycerides], [-0.59],
  )

  *Interpretation:* Favorable health profile (lower BP, higher HDL)
]

#slide[
  = Key Insights: Lifestyle-Physiology Link

  + Strong canonical correlation (r = 0.639) between lifestyle and health
  + Healthy lifestyle pattern → Favorable physiological profile
  + Exercise and low stress most important lifestyle factors
  + Blood pressure and cholesterol most related physiological markers

  *Clinical significance:* Lifestyle interventions can meaningfully improve multiple health markers
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
  = Case Study Summary

  *Healthcare Risk Assessment: What We Learned*
]

#slide[
  = Key Findings: Prediction and Classification

  *Logistic Regression Results:*
  - 71% accuracy predicting CVD risk (AUC = 0.77)
  - Exercise strongest protective factor (OR = 0.72)
  - Stress increases risk 25% per point (OR = 1.25)
  - Model identifies high-risk patients for early intervention

  *Clinical Value:* Enables targeted prevention strategies
]

#slide[
  = Key Findings: Group Comparisons

  *Hotelling's T-squared:*
  - High-risk patients differ significantly across 6 health markers (T² = 228.65, p < 0.0001)
  - Largest differences: cholesterol (+12.3) and triglycerides (+11.9)
  - Pattern suggests metabolic syndrome

  *Box's M Test:*
  - Covariance matrices equal between groups (M = 8.49)
  - MANOVA assumptions validated
]

#slide[
  = Key Findings: Treatment Effectiveness

  *MANOVA Results:*
  - Intervention improves all health outcomes (Λ = 0.889, p < 0.0001)
  - Blood pressure: -6.5 / -4.2 mmHg
  - Cholesterol: -7.9 mg/dL
  - Glucose: -5.6 mg/dL

  *Clinical Impact:* Comprehensive lifestyle changes yield broad benefits
]

#slide[
  = Key Findings: Lifestyle-Health Relationships

  *Canonical Correlation:*
  - Strong link between lifestyle and physiology (r = 0.639)
  - Healthy lifestyle pattern: ↑ exercise, ↓ stress
  - Favorable health profile: ↓ BP, ↑ HDL
  - 40.8% shared variance between domains

  *Clinical Insight:* Lifestyle interventions affect multiple health markers simultaneously
]

#slide[
  = Methodological Insights

  + *Multivariate methods reveal patterns* missed by univariate tests
  + *Type I error control* critical with multiple outcomes
  + *Assumption testing* (Box's M) validates results
  + *Effect sizes matter* beyond statistical significance
  + *Clinical context* guides interpretation

  All methods demonstrated with real healthcare data
]

#slide[
  = Key Takeaways: Models

  *Logistic Regression:*
  - Use for binary outcomes
  - Maximum likelihood estimation
  - Interpret via odds ratios
  - *Case Study:* 71% accuracy predicting CVD risk
]

#slide[
  = Key Takeaways: Inference

  *Covariance Matrix Tests:*
  - Box's M test for equality
  - Wishart distribution foundation
  - *Case Study:* M = 8.49 (assumption satisfied)

  *Mean Vector Tests:*
  - Hotelling's T-squared generalizes t-test
  - Confidence regions are ellipsoids
  - *Case Study:* T² = 228.65 (strong group differences)
]

#slide[
  = Key Takeaways: Advanced Methods

  *MANOVA:*
  - Multiple response variables simultaneously
  - Wilks' Lambda most common test
  - Controls Type I error
  - *Case Study:* Λ = 0.889 (intervention effective)

  *Canonical Correlation:*
  - Relates two variable sets
  - Multiple correlation pairs
  - *Case Study:* r = 0.639 (lifestyle-health link)
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
  - *Case Study:* All analyses implemented in Python
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
