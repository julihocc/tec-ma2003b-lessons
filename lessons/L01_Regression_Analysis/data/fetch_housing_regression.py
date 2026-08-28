#!/usr/bin/env python3
"""
fetch_housing_regression.py
Generates realistic synthetic residential property data for Chapter 1: Regression Analysis.
Demonstrates:
- Simple and multiple linear regression
- ANOVA and confidence/prediction intervals
- Residual normality and heteroskedasticity diagnostics
- Stepwise variable selection
- Polynomial and non-linear transformations
- Influential points and outlier diagnostics (Cook's distance)
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def generate_housing_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    logger.info(f"Generating {n_samples} residential property records...")

    # 1. Independent Predictors
    sqft_living = np.random.gamma(shape=9.0, scale=220.0, size=n_samples)
    sqft_living = np.clip(sqft_living, 600, 4800)

    bedrooms = np.round(1.0 + 0.0008 * sqft_living + np.random.normal(0, 0.6, n_samples))
    bedrooms = np.clip(bedrooms, 1, 6).astype(int)

    bathrooms = np.round((0.5 + 0.0007 * sqft_living + 0.3 * bedrooms + np.random.normal(0, 0.4, n_samples)) * 2) / 2
    bathrooms = np.clip(bathrooms, 1.0, 4.5)

    house_age_years = np.random.uniform(0, 65, size=n_samples)
    dist_city_center_km = np.random.exponential(scale=8.5, size=n_samples) + 1.2
    dist_city_center_km = np.clip(dist_city_center_km, 1.0, 40.0)

    building_grade = np.random.choice(
        [4, 5, 6, 7, 8, 9, 10],
        size=n_samples,
        p=[0.05, 0.15, 0.30, 0.25, 0.15, 0.07, 0.03]
    )

    energy_efficiency_rating = 220 - 1.2 * building_grade - 0.4 * (65 - house_age_years) + np.random.normal(0, 15, n_samples)
    energy_efficiency_rating = np.clip(energy_efficiency_rating, 45, 260)

    # 2. Target Variable: Sale Price (with non-linear age effect and heteroskedastic noise)
    # True structural equation:
    # Price = Base + 145*sqft + 12000*bath - 8500*dist + 28000*grade - 1200*age + 15*(age-30)^2 + Heteroskedastic Error
    base_price = 85000.0
    linear_signal = (
        base_price
        + 148.5 * sqft_living
        + 4500.0 * bedrooms
        + 14500.0 * bathrooms
        - 4200.0 * dist_city_center_km
        + 32000.0 * (building_grade - 6)
        - 850.0 * house_age_years
        + 18.5 * np.power(house_age_years - 30.0, 2)  # Non-linear historic/renovation premium
        - 180.0 * energy_efficiency_rating
    )

    # Heteroskedastic error variance proportional to square footage
    error_scale = 18000.0 + 8.5 * sqft_living
    errors = np.random.normal(loc=0.0, scale=error_scale, size=n_samples)

    # Introduce ~1.5% high-leverage / outlier anomalies for diagnostic learning
    n_outliers = int(0.015 * n_samples)
    outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
    errors[outlier_indices] += np.random.choice([-1, 1], size=n_outliers) * np.random.uniform(150000, 280000, size=n_outliers)

    sale_price = np.maximum(linear_signal + errors, 75000.0)

    df = pd.DataFrame({
        "property_id": np.arange(1001, 1001 + n_samples),
        "sqft_living": np.round(sqft_living, 1),
        "bedrooms": bedrooms,
        "bathrooms": np.round(bathrooms, 1),
        "house_age_years": np.round(house_age_years, 1),
        "dist_city_center_km": np.round(dist_city_center_km, 2),
        "building_grade": building_grade,
        "energy_efficiency_rating": np.round(energy_efficiency_rating, 1),
        "sale_price": np.round(sale_price, 2)
    })

    return df

def main():
    script_dir = Path(__file__).resolve().parent
    output_csv = script_dir / "housing_regression.csv"

    df = generate_housing_data(n_samples=1000, random_state=42)
    df.to_csv(output_csv, index=False)
    logger.info(f"Dataset successfully saved to: {output_csv}")

    print("\n=== Housing Regression Dataset Overview ===")
    print(f"Shape: {df.shape} (rows, columns)")
    print("\nFirst 5 Records:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])

if __name__ == "__main__":
    main()
