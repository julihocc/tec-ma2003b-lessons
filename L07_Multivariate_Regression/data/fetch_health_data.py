# %%
# Healthcare Risk Assessment Data Generation
# Chapter 7 - Multivariate Regression Example
# Generates synthetic patient data for multivariate regression analysis

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Simple logger
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

script_dir = Path(__file__).resolve().parent
data_file = script_dir / "health_data.csv"

logger.info("Starting healthcare risk assessment data generation")

# %%
# Set random seed for reproducibility
np.random.seed(42)

n_samples = 1000
logger.info(f"Generating {n_samples} patient records with multiple health indicators")

# %%
# Generate patient demographics and lifestyle factors
age = np.random.normal(52, 12, n_samples).clip(25, 80).round(0)
bmi = np.random.normal(26.5, 4.2, n_samples).clip(18, 45).round(1)
exercise_hours = np.random.exponential(2.5, n_samples).clip(0, 12).round(1)
smoking_years = np.random.exponential(5, n_samples).clip(0, 40).round(0)
alcohol_units = np.random.exponential(4, n_samples).clip(0, 25).round(1)
stress_score = np.random.normal(5.5, 2.0, n_samples).clip(1, 10).round(1)
sleep_hours = np.random.normal(7.0, 1.2, n_samples).clip(4, 10).round(1)

# %%
# Generate physiological measurements (correlated with lifestyle)
# Systolic blood pressure (mmHg)
systolic_bp = (
    95
    + 0.35 * age
    + 0.8 * bmi
    - 1.2 * exercise_hours
    + 0.4 * smoking_years
    + 0.6 * alcohol_units
    + 1.5 * stress_score
    - 2.0 * sleep_hours
    + np.random.normal(0, 8, n_samples)
).clip(90, 180).round(0)

# Diastolic blood pressure (mmHg)
diastolic_bp = (
    60
    + 0.20 * age
    + 0.5 * bmi
    - 0.8 * exercise_hours
    + 0.3 * smoking_years
    + 0.4 * alcohol_units
    + 1.0 * stress_score
    - 1.2 * sleep_hours
    + np.random.normal(0, 6, n_samples)
).clip(60, 110).round(0)

# Total cholesterol (mg/dL)
cholesterol = (
    140
    + 0.4 * age
    + 1.2 * bmi
    - 2.0 * exercise_hours
    + 0.5 * smoking_years
    + 0.8 * alcohol_units
    + 2.0 * stress_score
    - 1.5 * sleep_hours
    + np.random.normal(0, 15, n_samples)
).clip(120, 300).round(0)

# Fasting glucose (mg/dL)
glucose = (
    75
    + 0.25 * age
    + 1.0 * bmi
    - 1.5 * exercise_hours
    + 0.3 * smoking_years
    + 0.5 * alcohol_units
    + 1.2 * stress_score
    - 0.8 * sleep_hours
    + np.random.normal(0, 10, n_samples)
).clip(70, 200).round(0)

# Triglycerides (mg/dL)
triglycerides = (
    80
    + 0.30 * age
    + 1.5 * bmi
    - 2.5 * exercise_hours
    + 0.6 * smoking_years
    + 1.2 * alcohol_units
    + 1.8 * stress_score
    - 1.0 * sleep_hours
    + np.random.normal(0, 20, n_samples)
).clip(50, 300).round(0)

# HDL cholesterol (mg/dL) - higher is better
hdl = (
    50
    - 0.08 * age
    - 0.3 * bmi
    + 1.5 * exercise_hours
    - 0.2 * smoking_years
    - 0.3 * alcohol_units
    - 0.4 * stress_score
    + 0.8 * sleep_hours
    + np.random.normal(0, 8, n_samples)
).clip(25, 80).round(0)

# %%
# Generate cardiovascular disease risk (binary outcome for logistic regression)
# Risk based on composite of health factors
risk_score = (
    0.02 * age
    + 0.15 * bmi
    - 0.25 * exercise_hours
    + 0.12 * smoking_years
    + 0.10 * alcohol_units
    + 0.20 * stress_score
    - 0.15 * sleep_hours
    + 0.015 * systolic_bp
    + 0.012 * cholesterol
    + 0.010 * glucose
    + np.random.normal(0, 2, n_samples)
)

# Convert to binary: high risk if score above median
cvd_risk_high = (risk_score > np.median(risk_score)).astype(int)

# %%
# Generate treatment group (for MANOVA comparison)
# Randomly assign to control vs. intervention group
treatment_group = np.random.choice(['Control', 'Intervention'], size=n_samples, p=[0.5, 0.5])

# Intervention group shows slight improvements in health outcomes
# (add small improvements to physiological measurements for intervention group)
intervention_mask = treatment_group == 'Intervention'
systolic_bp[intervention_mask] -= np.random.normal(5, 3, intervention_mask.sum()).clip(0, 15)
diastolic_bp[intervention_mask] -= np.random.normal(3, 2, intervention_mask.sum()).clip(0, 10)
cholesterol[intervention_mask] -= np.random.normal(8, 5, intervention_mask.sum()).clip(0, 25)
glucose[intervention_mask] -= np.random.normal(4, 3, intervention_mask.sum()).clip(0, 15)

# Ensure bounds after intervention adjustments
systolic_bp = systolic_bp.clip(90, 180).round(0)
diastolic_bp = diastolic_bp.clip(60, 110).round(0)
cholesterol = cholesterol.clip(120, 300).round(0)
glucose = glucose.clip(70, 200).round(0)

# %%
# Create DataFrame
df = pd.DataFrame({
    # Demographics
    'patient_id': range(1, n_samples + 1),
    'age': age,
    'bmi': bmi,

    # Lifestyle factors (Set 1 for canonical correlation)
    'exercise_hours_week': exercise_hours,
    'smoking_years': smoking_years,
    'alcohol_units_week': alcohol_units,
    'stress_score': stress_score,
    'sleep_hours': sleep_hours,

    # Physiological measurements (Set 2 for canonical correlation)
    'systolic_bp': systolic_bp,
    'diastolic_bp': diastolic_bp,
    'cholesterol': cholesterol,
    'glucose': glucose,
    'triglycerides': triglycerides,
    'hdl': hdl,

    # Outcomes
    'cvd_risk_high': cvd_risk_high,
    'treatment_group': treatment_group
})

# %%
# Save dataset
df.to_csv(data_file, index=False)
logger.info(f"Saved health data to {data_file}")

# %%
# Data summary
print("=== Healthcare Risk Assessment Dataset Generated ===")
print(f"Total patients: {len(df)}")
print(f"Output file: {data_file}")

print("\nCVD Risk distribution:")
print(df['cvd_risk_high'].value_counts())
print(f"High risk prevalence: {df['cvd_risk_high'].mean():.1%}")

print("\nTreatment group distribution:")
print(df['treatment_group'].value_counts())

print("\nLifestyle factors summary:")
lifestyle_vars = ['exercise_hours_week', 'smoking_years', 'alcohol_units_week', 'stress_score', 'sleep_hours']
print(df[lifestyle_vars].describe().round(2))

print("\nPhysiological measurements summary:")
physio_vars = ['systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose', 'triglycerides', 'hdl']
print(df[physio_vars].describe().round(2))

print("\nHealth outcomes by CVD risk level:")
print(df.groupby('cvd_risk_high')[physio_vars].mean().round(1))

print("\nHealth outcomes by treatment group:")
print(df.groupby('treatment_group')[physio_vars].mean().round(1))

print("\n" + "=" * 60)
print("Dataset suitable for:")
print("1. Logistic Regression: Predicting cvd_risk_high")
print("2. Hotelling's T-squared: Comparing mean vectors by group")
print("3. MANOVA: Testing treatment effect on multiple outcomes")
print("4. Canonical Correlation: Relating lifestyle to health outcomes")
print("5. Factor-based Regression: Reducing multicollinearity")
print("=" * 60)

logger.info("Healthcare risk assessment data generation completed")
