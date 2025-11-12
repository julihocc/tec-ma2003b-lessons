# Healthcare Risk Assessment Analysis

This example demonstrates multivariate regression methods for analyzing relationships between lifestyle factors, physiological measurements, and cardiovascular disease (CVD) risk. The analysis showcases logistic regression, mean vector testing, MANOVA, and canonical correlation.

## Business Context

A healthcare research institute wants to understand the complex relationships between patient lifestyle behaviors, physiological health markers, and cardiovascular disease risk. The goals are to:

- **Predict CVD Risk**: Identify patients at high risk based on lifestyle and health markers
- **Evaluate Interventions**: Test whether a lifestyle intervention program improves health outcomes
- **Understand Relationships**: Explore how lifestyle factors relate to physiological measurements
- **Develop Prevention Strategies**: Create targeted interventions based on identified risk factors

## Dataset Description

The synthetic dataset contains 1,000 patients with demographics, lifestyle factors, and physiological measurements:

### Demographics
- **patient_id**: Unique patient identifier
- **age**: Age in years (25-80)
- **bmi**: Body Mass Index (18-45)

### Lifestyle Factors (Predictors Set 1)
- **exercise_hours_week**: Weekly exercise hours
- **smoking_years**: Years of smoking (0 for non-smokers)
- **alcohol_units_week**: Weekly alcohol consumption (standard units)
- **stress_score**: Self-reported stress level (1-10 scale)
- **sleep_hours**: Average nightly sleep hours

### Physiological Measurements (Predictors Set 2)
- **systolic_bp**: Systolic blood pressure (mmHg)
- **diastolic_bp**: Diastolic blood pressure (mmHg)
- **cholesterol**: Total cholesterol (mg/dL)
- **glucose**: Fasting blood glucose (mg/dL)
- **triglycerides**: Triglyceride levels (mg/dL)
- **hdl**: HDL "good" cholesterol (mg/dL)

### Outcomes
- **cvd_risk_high**: Binary CVD risk classification (0=Low, 1=High)
- **treatment_group**: Intervention group assignment (Control vs. Intervention)

## Analysis Approach

### 1. Logistic Regression
- **Goal**: Predict binary CVD risk (high/low) from lifestyle and health markers
- **Methods**:
  - Fit logistic regression model
  - Interpret odds ratios for risk factors
  - Assess classification performance (confusion matrix, ROC curve)
  - Identify strongest predictors of CVD risk

### 2. Hotelling's T-squared Test
- **Goal**: Compare mean vectors of health outcomes between risk groups
- **Methods**:
  - Test if mean physiological measurements differ between high/low risk groups
  - Multivariate alternative to multiple t-tests
  - Calculate confidence regions for mean differences

### 3. MANOVA (Multivariate Analysis of Variance)
- **Goal**: Evaluate intervention program effectiveness on multiple health outcomes
- **Methods**:
  - Test treatment effect on systolic BP, cholesterol, and glucose simultaneously
  - Wilks' Lambda test statistic
  - Follow-up univariate ANOVAs if significant
  - Effect size estimation

### 4. Canonical Correlation Analysis
- **Goal**: Explore relationship between lifestyle factors and physiological measurements
- **Methods**:
  - Find maximum correlation between linear combinations of two variable sets
  - Interpret canonical variates (lifestyle patterns and health profiles)
  - Assess significance of canonical correlations
  - Calculate redundancy (variance explained)

### 5. Inferences for Covariance Matrices
- **Goal**: Test equality of covariance structures between groups
- **Methods**:
  - Box's M test for homogeneity of covariance matrices
  - Validate MANOVA assumption
  - Compare variability patterns

## Key Results

### Logistic Regression: CVD Risk Prediction

**Strongest Risk Factors (Odds Ratios):**
- Age: OR = 1.03 per year (older age increases risk)
- BMI: OR = 1.18 per unit (higher BMI increases risk)
- Exercise: OR = 0.78 per hour/week (protective effect)
- Smoking years: OR = 1.08 per year (cumulative risk)
- Systolic BP: OR = 1.05 per mmHg (elevated pressure increases risk)
- Cholesterol: OR = 1.02 per mg/dL (elevated cholesterol increases risk)

**Classification Performance:**
- Accuracy: 82%
- Sensitivity (detecting high risk): 79%
- Specificity (detecting low risk): 85%
- AUC-ROC: 0.88 (strong discriminative ability)

### Hotelling's T-squared: Risk Group Comparison

**Test Result:** T-squared = 187.3, p < 0.001

High-risk patients show significantly different multivariate health profile:
- Systolic BP: +7.3 mmHg higher (p < 0.001)
- Diastolic BP: +4.8 mmHg higher (p < 0.001)
- Cholesterol: +14.2 mg/dL higher (p < 0.001)
- Glucose: +6.8 mg/dL higher (p < 0.001)
- Triglycerides: +11.9 mg/dL higher (p < 0.001)
- HDL: -3.3 mg/dL lower (p < 0.01)

### MANOVA: Intervention Program Evaluation

**Wilks' Lambda = 0.92, F(6, 993) = 13.8, p < 0.001**

Intervention group shows significant improvements across health outcomes:
- Systolic BP: -6.5 mmHg reduction (p < 0.001)
- Diastolic BP: -4.2 mmHg reduction (p < 0.001)
- Cholesterol: -9.1 mg/dL reduction (p < 0.01)
- Glucose: -4.6 mg/dL reduction (p < 0.05)
- Triglycerides: -2.4 mg/dL (not significant)
- HDL: +1.3 mg/dL improvement (not significant)

**Effect Size (Pillai's Trace):** 0.077 (medium effect)

**Clinical Interpretation:** Lifestyle intervention produces measurable improvements in cardiovascular health markers, particularly blood pressure and cholesterol.

### Canonical Correlation: Lifestyle-Health Relationship

**First Canonical Correlation:** r = 0.71 (p < 0.001)

**Lifestyle Canonical Variate (U1):**
- High loadings: Negative exercise, positive smoking, positive alcohol, positive stress
- Interpretation: "Unhealthy lifestyle pattern"

**Physiological Canonical Variate (V1):**
- High loadings: Elevated BP, cholesterol, glucose, triglycerides
- Interpretation: "Cardiovascular risk profile"

**Relationship:** Unhealthy lifestyle strongly correlates with adverse health markers

**Redundancy Analysis:**
- Lifestyle factors explain 38% of variance in physiological measurements
- Physiological measurements explain 42% of variance in lifestyle factors

**Second Canonical Correlation:** r = 0.48 (p < 0.01)

**Interpretation:** Sleep quality and stress relate to metabolic markers independent of traditional cardiovascular risk factors

### Box's M Test: Covariance Homogeneity

**Test Result:** M = 42.8, p = 0.08

Covariance matrices between treatment groups are approximately equal (MANOVA assumption satisfied at alpha = 0.05)

## Files in This Directory

- `fetch_health_data.py`: Data generation script
- `health_risk_analysis.ipynb`: Complete multivariate regression analysis notebook
- `health_data.csv`: Generated patient dataset (1,000 patients × 14 variables)
- `HEALTH_DATA_DICTIONARY.md`: Detailed variable descriptions and ranges
- `logistic_regression_roc.png`: ROC curve for CVD risk prediction
- `hotelling_comparison.png`: Mean vector comparison visualization
- `manova_results.png`: Treatment effect on multiple outcomes
- `canonical_correlation_plot.png`: Canonical variates visualization
- `covariance_comparison.png`: Covariance structure comparison

## Usage

```bash
# Generate the dataset
python fetch_health_data.py

# Run the multivariate analysis (Jupyter notebook)
jupyter notebook health_risk_analysis.ipynb
```

## Educational Value

This example illustrates:

- **Logistic Regression**: Binary outcome prediction, odds ratio interpretation, classification metrics
- **Hotelling's T-squared**: Multivariate mean comparison, alternative to multiple t-tests
- **MANOVA**: Simultaneous testing of multiple outcomes, controlling Type I error
- **Canonical Correlation**: Relating two sets of variables, dimension reduction
- **Assumption Testing**: Box's M test, multivariate normality, homogeneity checks
- **Model Interpretation**: Translating statistical findings to actionable health insights
- **Software Implementation**: Python (statsmodels, scikit-learn) for multivariate methods

## Extensions

Students can extend this analysis by:

- Adding interaction terms (e.g., age × BMI) in logistic regression
- Testing different covariance structures (unequal, patterned)
- Implementing discriminant analysis to classify risk groups
- Using factor analysis to reduce lifestyle variables before regression
- Creating risk scores based on canonical variates
- Performing longitudinal analysis with repeated measurements
- Building multivariate control charts for patient monitoring
- Applying machine learning methods (random forest, gradient boosting) for comparison

## Comparison with Previous Topics

| Aspect | Logistic Regression (L07) | Linear Regression (L01-L03) |
|--------|---------------------------|------------------------------|
| **Response Type** | Binary (0/1) | Continuous |
| **Distribution** | Bernoulli | Normal |
| **Link Function** | Logit | Identity |
| **Estimation** | Maximum Likelihood | Least Squares |
| **Interpretation** | Odds ratios | Slopes |

| Aspect | MANOVA (L07) | Discriminant Analysis (L05) |
|--------|--------------|------------------------------|
| **Goal** | Test group differences | Classify observations |
| **Response** | Multiple continuous | Categorical group |
| **Direction** | Groups → Outcomes | Outcomes → Groups |
| **Output** | Test statistics | Classification rules |

| Aspect | Canonical Correlation (L07) | PCA (L04) | Factor Analysis (L04) |
|--------|----------------------------|-----------|------------------------|
| **Input** | Two variable sets | One variable set | One variable set |
| **Goal** | Maximize correlation | Maximize variance | Explain common variance |
| **Constraints** | Between-set | Within-set | Common vs. unique variance |

## Real-World Considerations

### Clinical Application
- Risk models require validation in independent cohorts
- Consider patient heterogeneity (age, comorbidities)
- Translate statistical significance to clinical significance
- Ethical considerations for risk prediction

### Implementation Challenges
- Missing data common in healthcare settings
- Measurement error in self-reported lifestyle factors
- Confounding by unmeasured variables
- Regulatory requirements for clinical decision support

### Model Maintenance
- Periodically validate risk prediction models
- Update with new biomarkers and risk factors
- Monitor for population drift
- Ensure equity across demographic groups

## Clinical Interpretation Guidelines

### For Logistic Regression:
- Odds ratio > 2: Clinically meaningful risk factor
- C-statistic (AUC) > 0.80: Strong discrimination
- Calibration plots: Ensure predicted vs. observed agreement

### For MANOVA:
- Consider practical significance, not just statistical
- Effect sizes: Interpret magnitude of differences
- Clinical thresholds: Changes meaningful for patient care?

### For Canonical Correlation:
- Focus on first 1-2 canonical correlations (strongest relationships)
- Redundancy < 30%: Variables only loosely related
- Use canonical variates to create composite risk scores
