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

- Exercise hours/week: OR = 0.72 (28% lower odds per hour - strongest protective factor)
- Stress score: OR = 1.25 (25% higher odds per point)
- Sleep hours: OR = 0.80 (20% lower odds per hour)
- BMI: OR = 1.19 (19% higher odds per unit)
- Age: OR = 1.03 per year
- Smoking years: OR = 1.12 per year
- Alcohol units: OR = 1.12 per unit

**Classification Performance:**

- Accuracy: 71%
- Sensitivity (detecting high risk): 72%
- Specificity (detecting low risk): 71%
- AUC-ROC: 0.77 (good discriminative ability)
- Balanced precision/recall

**Clinical Value:** Model enables early identification of high-risk patients for targeted preventive care.

### Hotelling's T-squared: Risk Group Comparison

**Test Result:** T² = 228.65, F(6, 993) = 37.92, p < 0.0001

High-risk patients (n=500) vs. low-risk patients (n=500) show significantly different multivariate health profile:

- Systolic BP: 131.1 vs. 123.8 mmHg (+7.3 mmHg)
- Diastolic BP: 82.9 vs. 78.1 mmHg (+4.7 mmHg)
- Cholesterol: 196.4 vs. 184.1 mg/dL (+12.3 mg/dL)
- Glucose: 116.8 vs. 109.9 mg/dL (+6.9 mg/dL)
- Triglycerides: 145.5 vs. 133.6 mg/dL (+11.9 mg/dL)
- HDL: 41.2 vs. 44.5 mg/dL (-3.3 mg/dL)

**Clinical Interpretation:** Pattern of differences suggests metabolic syndrome in high-risk group.

### MANOVA: Intervention Program Evaluation

#### Test Results

**Wilks' Lambda = 0.889, F(4, 995) = 31.05, p < 0.0001**

Intervention group (n=521) vs. Control (n=479) shows significant improvements:

| Outcome | Control | Intervention | Difference | p-value |
|---------|---------|-------------|------------|---------|
| Systolic BP | 130.8 mmHg | 124.3 mmHg | -6.5 mmHg | <0.001 |
| Diastolic BP | 82.7 mmHg | 78.5 mmHg | -4.2 mmHg | <0.001 |
| Cholesterol | 194.4 mg/dL | 186.5 mg/dL | -7.9 mg/dL | <0.001 |
| Glucose | 116.3 mg/dL | 110.7 mg/dL | -5.6 mg/dL | <0.001 |

#### Clinical Interpretation

All differences significant (p < 0.001) in follow-up univariate ANOVAs. Comprehensive lifestyle intervention produces measurable improvements across all cardiovascular risk markers, with largest effects on blood pressure.

### Canonical Correlation: Lifestyle-Health Relationship

**First Canonical Correlation:** r = 0.639 (p < 0.001)

**Lifestyle Canonical Variate (U1) - Loadings:**
- Exercise hours: +0.65 (healthy lifestyle indicator)
- Stress score: -0.53
- Alcohol units: -0.37
- Sleep hours: +0.33
- Smoking years: -0.27
- **Interpretation:** "Healthy lifestyle pattern" (more exercise, less stress)

**Physiological Canonical Variate (V1) - Loadings:**
- Diastolic BP: -0.70
- Systolic BP: -0.68
- Cholesterol: -0.65
- HDL: +0.65 (protective)
- Glucose: -0.61
- Triglycerides: -0.59
- **Interpretation:** "Favorable health profile" (lower BP, higher HDL)

**Relationship:** Strong canonical correlation (r = 0.639) indicates healthy lifestyle pattern is strongly associated with favorable physiological profile.

**Variance Explained:**
- First canonical correlation explains 40.8% shared variance
- Second canonical correlation: r = 0.244 (moderate relationship)
- Remaining correlations: < 0.12 (weak)

**Clinical Insight:** Lifestyle interventions can meaningfully improve multiple health markers simultaneously.

### Box's M Test: Covariance Homogeneity

**Test Result:** M = 8.49, df = 10

**Interpretation:** M < 30 (rule of thumb indicates homogeneity)

**Conclusion:** Covariance matrices are approximately equal between treatment groups. MANOVA assumption satisfied, validating our MANOVA results.

**Implication:** Treatment groups have similar variability patterns; differences are in means, not covariance structure.

## Presentation Structure

The presentation (`multivariate_regression_slides.typ`) integrates theoretical concepts with the practical health risk analysis case study:

**Part 1: Theoretical Foundations** - Core multivariate regression concepts with case study references:

- Logistic regression fundamentals (with CVD risk prediction example)
- Odds ratio interpretation (with exercise, stress, sleep effects)
- Hotelling's T-squared test (with risk group comparison showing metabolic syndrome pattern)
- MANOVA principles (with intervention program evaluation showing comprehensive improvements)
- Canonical correlation analysis (with lifestyle-physiology relationship, r=0.639)
- Box's M test (with covariance homogeneity validation, M=8.49)
- Model validation and assumptions (with ROC curves, classification metrics)

**Part 2: Practical Application** - Complete healthcare risk assessment workflow:

- Business context and research objectives
- Dataset overview (1,000 patients, 14 variables)
- Logistic regression implementation (CVD risk prediction, AUC=0.77)
- Model interpretation (odds ratios, protective vs. risk factors)
- Classification performance evaluation (71% accuracy, balanced sensitivity/specificity)
- Hotelling's T² comparison (high-risk vs. low-risk health profiles)
- MANOVA intervention evaluation (Wilks' Λ=0.889, significant improvements)
- Canonical correlation exploration (lifestyle patterns and health profiles)
- Clinical interpretation and actionable insights

The presentation serves as a companion to the Jupyter notebook, demonstrating how theoretical multivariate regression methods apply to real healthcare decision-making throughout.

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
