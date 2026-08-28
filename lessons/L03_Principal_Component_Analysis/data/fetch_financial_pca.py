#!/usr/bin/env python3
"""
fetch_financial_pca.py
Generates realistic synthetic daily returns data for Chapter 3: Principal Component Analysis (PCA).
Demonstrates:
- Cases where PCA is used (Multicollinearity, asset allocation risk factor modeling)
- Geometrical description of principal axes and variance maximization
- Spectral decomposition of covariance (S) and correlation (R) matrices
- Component retention rules (Kaiser criterion, Scree plot, cumulative variance, Parallel Analysis)
- 2D/3D PCA Biplots and score interpretation in Python (scikit-learn)
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def generate_financial_pca_data(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    logger.info(f"Generating {n_samples} financial market asset class observations...")

    asset_names = [
        "us_equities",
        "global_equities",
        "emerging_markets",
        "tech_sector",
        "energy_sector",
        "financials_sector",
        "treasury_bonds",
        "corporate_bonds",
        "commodities_gold",
        "real_estate_reits"
    ]

    # 1. Simulate 3 underlying macroeconomic latent market drivers (Orthogonal Factors)
    # F1: Global Growth / Market Equity Factor (Mean=0, Var=1.0)
    # F2: Interest Rate / Monetary Tightening Factor (Mean=0, Var=1.0)
    # F3: Energy / Commodity Inflation Shock Factor (Mean=0, Var=1.0)
    F1 = np.random.normal(0, 1.0, size=n_samples)
    F2 = np.random.normal(0, 1.0, size=n_samples)
    F3 = np.random.normal(0, 1.0, size=n_samples)

    # 2. Factor Loadings onto Asset Returns (percentage daily returns)
    # Asset = mu + L1*F1 + L2*F2 + L3*F3 + Idiosyncratic Noise
    us_eq = 0.04 + 0.92 * F1 - 0.20 * F2 + 0.05 * F3 + np.random.normal(0, 0.25, n_samples)
    gl_eq = 0.03 + 0.88 * F1 - 0.25 * F2 + 0.15 * F3 + np.random.normal(0, 0.30, n_samples)
    em_eq = 0.02 + 0.82 * F1 - 0.35 * F2 + 0.30 * F3 + np.random.normal(0, 0.45, n_samples)
    tech  = 0.06 + 1.15 * F1 - 0.45 * F2 - 0.10 * F3 + np.random.normal(0, 0.35, n_samples)
    energy= 0.02 + 0.65 * F1 + 0.15 * F2 + 0.85 * F3 + np.random.normal(0, 0.50, n_samples)
    fin   = 0.03 + 0.85 * F1 + 0.35 * F2 - 0.05 * F3 + np.random.normal(0, 0.35, n_samples)
    bonds = 0.01 - 0.25 * F1 - 0.85 * F2 - 0.15 * F3 + np.random.normal(0, 0.20, n_samples)
    corp  = 0.02 + 0.20 * F1 - 0.70 * F2 - 0.10 * F3 + np.random.normal(0, 0.22, n_samples)
    gold  = 0.02 - 0.15 * F1 - 0.30 * F2 + 0.75 * F3 + np.random.normal(0, 0.40, n_samples)
    reits = 0.02 + 0.60 * F1 - 0.65 * F2 - 0.05 * F3 + np.random.normal(0, 0.35, n_samples)

    data = np.column_stack([
        us_eq, gl_eq, em_eq, tech, energy, fin, bonds, corp, gold, reits
    ])

    df = pd.DataFrame(np.round(data, 3), columns=asset_names)
    df.insert(0, "trading_day", np.arange(1, n_samples + 1))

    return df

def main():
    script_dir = Path(__file__).resolve().parent
    output_csv = script_dir / "financial_market_data.csv"

    df = generate_financial_pca_data(n_samples=600, random_state=42)
    df.to_csv(output_csv, index=False)
    logger.info(f"Dataset successfully saved to: {output_csv}")

    print("\n=== Financial Market PCA Dataset Overview ===")
    print(f"Shape: {df.shape} (rows, columns)")
    print("\nFirst 5 Records (Daily Asset Returns %):")
    print(df.head())
    print("\nCorrelation Matrix Summary:")
    print(df.drop(columns=["trading_day"]).corr().round(2))

if __name__ == "__main__":
    main()
