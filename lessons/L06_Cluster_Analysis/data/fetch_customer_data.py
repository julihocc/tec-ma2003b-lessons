# %%
# E-Commerce Customer Data Generation
# Chapter 6 - Cluster Analysis Example
# Generates synthetic customer behavioral data for unsupervised segmentation

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
data_file = script_dir / "customer_data.csv"

logger.info("Starting e-commerce customer data generation")

# %%
# Define customer behavioral patterns (4 natural clusters)
np.random.seed(42)  # For reproducibility

n_samples = 2000
# Note: In cluster analysis, we don't have predefined labels
# These are the TRUE underlying patterns we want to discover

# Cluster proportions (discovered, not predefined)
cluster_sizes = {
    "PowerShoppers": int(0.22 * n_samples),  # 22% - High-value customers
    "BargainHunters": int(0.28 * n_samples),  # 28% - Price-sensitive buyers
    "WindowShoppers": int(0.35 * n_samples),  # 35% - Browsers with low conversion
    "ImpulseBuyers": int(0.15 * n_samples),  # 15% - Sporadic big spenders
}

logger.info(f"Generating {n_samples} customers with {len(cluster_sizes)} natural patterns")

# %%
# Power Shoppers: High spending, frequent purchases, highly engaged
power_mean = np.array(
    [
        12.0,  # monthly_purchases
        20.0,  # avg_basket_size
        6000.0,  # total_spend (12 months)
        50.0,  # session_duration (minutes)
        10.0,  # email_clicks per month
        40.0,  # product_views per session
        0.10,  # return_rate (low - deliberate purchases)
    ]
)

power_cov = np.array(
    [
        [2.0, 5.0, 500.0, 8.0, 1.5, 5.0, -0.01],  # monthly_purchases
        [5.0, 6.25, 800.0, 10.0, 2.0, 6.0, -0.015],  # avg_basket_size
        [500.0, 800.0, 400000.0, 600.0, 120.0, 400.0, -2.0],  # total_spend
        [8.0, 10.0, 600.0, 64.0, 12.0, 50.0, -1.0],  # session_duration
        [1.5, 2.0, 120.0, 12.0, 2.25, 8.0, -0.15],  # email_clicks
        [5.0, 6.0, 400.0, 50.0, 8.0, 49.0, -0.8],  # product_views
        [-0.01, -0.015, -2.0, -1.0, -0.15, -0.8, 0.004],  # return_rate
    ]
)

# %%
# Bargain Hunters: Moderate frequency, small baskets, price-sensitive
bargain_mean = np.array(
    [
        6.5,  # monthly_purchases
        4.5,  # avg_basket_size (small orders)
        1200.0,  # total_spend
        25.0,  # session_duration
        5.0,  # email_clicks
        32.0,  # product_views (comparison shopping)
        0.13,  # return_rate
    ]
)

bargain_cov = np.array(
    [
        [1.5, 1.2, 200.0, 4.0, 0.8, 3.0, -0.008],  # monthly_purchases
        [1.2, 1.0, 80.0, 2.5, 0.5, 2.0, -0.005],  # avg_basket_size
        [200.0, 80.0, 40000.0, 150.0, 40.0, 120.0, -0.5],  # total_spend
        [4.0, 2.5, 150.0, 25.0, 3.0, 15.0, -0.2],  # session_duration
        [0.8, 0.5, 40.0, 3.0, 1.0, 4.0, -0.08],  # email_clicks
        [3.0, 2.0, 120.0, 15.0, 4.0, 36.0, -0.3],  # product_views
        [-0.008, -0.005, -0.5, -0.2, -0.08, -0.3, 0.0025],  # return_rate
    ]
)

# %%
# Window Shoppers: Low purchase frequency, high browsing, low engagement
window_mean = np.array(
    [
        1.2,  # monthly_purchases (very low)
        2.5,  # avg_basket_size
        350.0,  # total_spend (low)
        40.0,  # session_duration (high - browsing)
        1.0,  # email_clicks (very low)
        27.0,  # product_views
        0.20,  # return_rate
    ]
)

window_cov = np.array(
    [
        [0.5, 0.6, 50.0, 5.0, 0.2, 2.0, 0.01],  # monthly_purchases
        [0.6, 1.0, 60.0, 3.0, 0.15, 1.5, 0.008],  # avg_basket_size
        [50.0, 60.0, 10000.0, 200.0, 15.0, 80.0, 0.3],  # total_spend
        [5.0, 3.0, 200.0, 81.0, 8.0, 40.0, 0.5],  # session_duration (high variance)
        [0.2, 0.15, 15.0, 8.0, 0.64, 3.0, 0.05],  # email_clicks
        [2.0, 1.5, 80.0, 40.0, 3.0, 36.0, 0.4],  # product_views
        [0.01, 0.008, 0.3, 0.5, 0.05, 0.4, 0.01],  # return_rate
    ]
)

# %%
# Impulse Buyers: Sporadic high-value purchases, quick decisions, promotion-driven
impulse_mean = np.array(
    [
        3.0,  # monthly_purchases (sporadic)
        15.0,  # avg_basket_size (large when buying)
        3000.0,  # total_spend (high)
        10.0,  # session_duration (quick decisions)
        8.0,  # email_clicks (promotion-driven)
        12.0,  # product_views (low - minimal browsing)
        0.28,  # return_rate (high - impulse regret)
    ]
)

impulse_cov = np.array(
    [
        [0.81, 2.5, 300.0, 1.5, 1.0, 1.5, 0.02],  # monthly_purchases
        [2.5, 9.0, 900.0, 3.0, 2.0, 3.0, 0.03],  # avg_basket_size
        [300.0, 900.0, 250000.0, 200.0, 150.0, 180.0, 2.5],  # total_spend
        [1.5, 3.0, 200.0, 16.0, 5.0, 12.0, 0.4],  # session_duration
        [1.0, 2.0, 150.0, 5.0, 2.25, 6.0, 0.25],  # email_clicks
        [1.5, 3.0, 180.0, 12.0, 6.0, 25.0, 0.5],  # product_views
        [0.02, 0.03, 2.5, 0.4, 0.25, 0.5, 0.01],  # return_rate
    ]
)

# %%
# Generate multivariate normal data for each natural pattern
data_frames = []
cluster_labels = []  # True labels (for validation only - NOT used in clustering)

for cluster_name, size in cluster_sizes.items():
    if cluster_name == "PowerShoppers":
        mean = power_mean
        cov = power_cov
    elif cluster_name == "BargainHunters":
        mean = bargain_mean
        cov = bargain_cov
    elif cluster_name == "WindowShoppers":
        mean = window_mean
        cov = window_cov
    else:  # ImpulseBuyers
        mean = impulse_mean
        cov = impulse_cov

    # Generate data
    cluster_data = np.random.multivariate_normal(mean, cov, size)

    # Create DataFrame
    df_cluster = pd.DataFrame(
        cluster_data,
        columns=[
            "monthly_purchases",
            "avg_basket_size",
            "total_spend",
            "session_duration",
            "email_clicks",
            "product_views",
            "return_rate",
        ],
    )

    # Store true label for validation (not included in output CSV)
    cluster_labels.extend([cluster_name] * size)

    data_frames.append(df_cluster)
    logger.info(f"Generated {size} customers with {cluster_name} pattern")

# %%
# Combine all clusters
df = pd.concat(data_frames, ignore_index=True)

# Add true labels as a hidden column (for educational validation only)
df["true_cluster"] = cluster_labels

# Ensure realistic bounds and precision
df["monthly_purchases"] = df["monthly_purchases"].clip(lower=0.2).round(1)
df["avg_basket_size"] = df["avg_basket_size"].clip(lower=1.0).round(1)
df["total_spend"] = df["total_spend"].clip(lower=50.0).round(2)
df["session_duration"] = df["session_duration"].clip(lower=2.0).round(1)
df["email_clicks"] = df["email_clicks"].clip(lower=0).round(1)
df["product_views"] = df["product_views"].clip(lower=3.0).round(1)
df["return_rate"] = df["return_rate"].clip(0, 0.50).round(3)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
# Save dataset WITHOUT true labels (unsupervised learning)
df_unsupervised = df.drop(columns=["true_cluster"])
df_unsupervised.to_csv(data_file, index=False)
logger.info(f"Saved customer data to {data_file} (WITHOUT true labels - unsupervised)")

# Save dataset WITH true labels to separate file for validation
validation_file = script_dir / "customer_data_with_labels.csv"
df.to_csv(validation_file, index=False)
logger.info(f"Saved labeled data to {validation_file} (FOR VALIDATION ONLY)")

# %%
# Data summary
print("=== E-Commerce Customer Dataset Generated ===")
print(f"Total customers: {len(df)}")
print(f"Unsupervised file (main): {data_file}")
print(f"Validation file (with true labels): {validation_file}")

print("\nTrue cluster distribution (hidden from students initially):")
print(df["true_cluster"].value_counts())

print("\nFeature summary:")
print(df_unsupervised.describe().round(2))

print("\nTrue cluster means by feature (for instructor reference):")
print(df.groupby("true_cluster").mean().round(2))

print("\n" + "=" * 60)
print("NOTE: The main CSV file does NOT contain cluster labels.")
print("Students must discover these patterns using cluster analysis.")
print("=" * 60)

logger.info("E-commerce customer data generation completed")
