#!/usr/bin/env python3
"""
fetch_environmental_data.py
Generates realistic synthetic environmental air quality monitoring data for Chapter 2: Multivariate Analysis.
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def generate_environmental_data(n_samples: int = 800, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    logger.info(f"Generating {n_samples} environmental monitoring records...")

    # 1. Define Population Mean Vector (mu) and Covariance Matrix (Sigma)
    var_names = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity"]
    means = np.array([35.0, 68.0, 28.0, 12.0, 1.4, 45.0, 22.0, 58.0])
    stds = np.array([12.0, 22.0, 9.5, 4.5, 0.45, 14.0, 6.5, 15.0])

    corr_matrix = np.array([
        [1.00,  0.84, 0.62, 0.48, 0.55, -0.32, 0.15, -0.25],
        [0.84,  1.00, 0.58, 0.52, 0.50, -0.28, 0.18, -0.20],
        [0.62,  0.58, 1.00, 0.65, 0.72, -0.45, 0.10, -0.15],
        [0.48,  0.52, 0.65, 1.00, 0.58, -0.22, 0.05, -0.08],
        [0.55,  0.50, 0.72, 0.58, 1.00, -0.38, 0.08, -0.12],
        [-0.32,-0.28,-0.45,-0.22,-0.38,  1.00, 0.55, -0.42],
        [0.15,  0.18, 0.10, 0.05, 0.08,  0.55, 1.00, -0.50],
        [-0.25,-0.20,-0.15,-0.08,-0.12, -0.42,-0.50,  1.00]
    ])

    D = np.diag(stds)
    cov_matrix = D @ corr_matrix @ D

    # 2. Draw Multivariate Normal Samples
    data = np.random.multivariate_normal(mean=means, cov=cov_matrix, size=n_samples)

    # 3. Add Subtle Multivariate Outliers (~2.5% of sample)
    n_outliers = int(0.025 * n_samples)
    outlier_idx = np.random.choice(n_samples, size=n_outliers, replace=False)
    for idx in outlier_idx:
        data[idx, 2] += np.random.uniform(18, 30)   # High NO2
        data[idx, 5] += np.random.uniform(25, 40)   # High O3
        data[idx, 7] += np.random.uniform(20, 30)   # High humidity

    # Ensure physical positivity
    data[:, 0] = np.maximum(data[:, 0], 2.0)
    data[:, 1] = np.maximum(data[:, 1], 5.0)
    data[:, 2] = np.maximum(data[:, 2], 1.0)
    data[:, 3] = np.maximum(data[:, 3], 0.5)
    data[:, 4] = np.maximum(data[:, 4], 0.1)
    data[:, 5] = np.maximum(data[:, 5], 1.0)
    data[:, 7] = np.clip(data[:, 7], 10.0, 100.0)

    df = pd.DataFrame(data, columns=var_names)
    df.insert(0, "station_id", np.arange(101, 101 + n_samples))

    df["pm25"] = df["pm25"].round(1)
    df["pm10"] = df["pm10"].round(1)
    df["no2"] = df["no2"].round(1)
    df["so2"] = df["so2"].round(2)
    df["co"] = df["co"].round(2)
    df["o3"] = df["o3"].round(1)
    df["temperature"] = df["temperature"].round(1)
    df["humidity"] = df["humidity"].round(1)

    # 4. Introduce Missing Values (~3% MCAR/MAR)
    mask = pd.DataFrame(np.random.rand(n_samples, len(var_names)) < 0.03, columns=var_names)
    df_with_nan = df.copy()
    df_with_nan[var_names] = df_with_nan[var_names].mask(mask)

    return df_with_nan

def main():
    script_dir = Path(__file__).resolve().parent
    output_csv = script_dir / "environmental_data.csv"

    df = generate_environmental_data(n_samples=800, random_state=42)
    df.to_csv(output_csv, index=False)
    logger.info(f"Dataset successfully saved to: {output_csv}")

    print("\n=== Environmental Air Quality Dataset Overview ===")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Total Missing Values: {df.isna().sum().sum()}")
    print("\nMissing Values by Feature:")
    print(df.isna().sum())
    print("\nFirst 5 Records:")
    print(df.head())

if __name__ == "__main__":
    main()
