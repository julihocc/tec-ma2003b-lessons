#!/usr/bin/env python3
# %%
"""
fetch_educational.py

Generates realistic synthetic educational assessment data for Factor Analysis demonstration.
Saves data as `educational.csv` in the same folder.

Data structure:
- 200 students assessed on 9 educational measures
- Three underlying latent factors with realistic correlations:
  * Quantitative reasoning (Math, Algebra, Geometry)
  * Verbal ability (Reading Comprehension, Vocabulary, Writing)
  * Interpersonal skills (Collaboration, Leadership, Communication)
- Realistic score distributions (0-100 scale, mean ~70, SD ~12)
- Moderate cross-loadings between related abilities
- Individual measurement error for each assessment

Usage:
    python fetch_educational.py

This creates pedagogically realistic data suitable for demonstrating:
- Factor retention decisions
- Factor rotation benefits
- Communality interpretation
- Construct validation
"""

# %%
import os

# %%
import numpy as np
import pandas as pd


# %%
def main():
    """Generate synthetic educational assessment data with realistic structure"""
    dst = os.path.join(os.path.dirname(__file__), "educational.csv")

    # Set random seed for reproducible data
    np.random.seed(42)

    # Number of students
    n_students = 200

    # Generate student IDs
    student_ids = [f"STUD_{i:03d}" for i in range(1, n_students + 1)]

    # Generate three correlated latent factors (not orthogonal - realistic!)
    # Cognitive abilities tend to correlate moderately
    mean = [0, 0, 0]
    cov = [
        [1.0, 0.3, 0.2],  # Quantitative reasoning
        [0.3, 1.0, 0.25],  # Verbal reasoning (moderate correlation with quant)
        [0.2, 0.25, 1.0],  # Interpersonal skills (weak correlation with cognitive)
    ]
    latent_factors = np.random.multivariate_normal(mean, cov, n_students)

    quantitative_ability = latent_factors[:, 0:1]
    verbal_ability = latent_factors[:, 1:2]
    interpersonal_ability = latent_factors[:, 2:3]

    # Generate unique measurement errors for each variable (different variances)
    errors = [np.random.normal(0, 0.4, (n_students, 1)) for _ in range(9)]

    # Create observed educational assessments with realistic loading patterns
    # Strong loadings (0.7-0.85) for primary factor, weak cross-loadings

    # Quantitative cluster (strong on quantitative, weak on verbal)
    math_score = 0.80 * quantitative_ability + 0.10 * verbal_ability + errors[0]
    algebra_score = 0.85 * quantitative_ability + 0.05 * verbal_ability + errors[1]
    geometry_score = 0.75 * quantitative_ability + 0.15 * verbal_ability + errors[2]

    # Verbal cluster (strong on verbal, weak on quantitative)
    reading_comp = 0.15 * quantitative_ability + 0.82 * verbal_ability + errors[3]
    vocabulary = 0.05 * quantitative_ability + 0.78 * verbal_ability + errors[4]
    writing = 0.10 * quantitative_ability + 0.80 * verbal_ability + errors[5]

    # Interpersonal cluster (primarily interpersonal, slight verbal component)
    collaboration = 0.10 * verbal_ability + 0.83 * interpersonal_ability + errors[6]
    leadership = 0.05 * verbal_ability + 0.77 * interpersonal_ability + errors[7]
    communication = 0.20 * verbal_ability + 0.75 * interpersonal_ability + errors[8]

    # Convert to realistic scale (0-100) with reasonable mean and SD
    def scale_to_100(arr, target_mean=70, target_sd=12):
        """Convert standardized scores to 0-100 scale"""
        scaled = arr * target_sd + target_mean
        return np.clip(scaled, 0, 100)  # Ensure within 0-100 range

    # Create DataFrame with realistic educational score ranges
    data = {
        "Student": student_ids,
        "MathScore": np.round(scale_to_100(math_score.flatten(), 72, 13), 1),
        "AlgebraScore": np.round(scale_to_100(algebra_score.flatten(), 68, 14), 1),
        "GeometryScore": np.round(scale_to_100(geometry_score.flatten(), 70, 12), 1),
        "ReadingComp": np.round(scale_to_100(reading_comp.flatten(), 73, 11), 1),
        "Vocabulary": np.round(scale_to_100(vocabulary.flatten(), 75, 13), 1),
        "Writing": np.round(scale_to_100(writing.flatten(), 71, 12), 1),
        "Collaboration": np.round(scale_to_100(collaboration.flatten(), 76, 10), 1),
        "Leadership": np.round(scale_to_100(leadership.flatten(), 69, 13), 1),
        "Communication": np.round(scale_to_100(communication.flatten(), 74, 11), 1),
    }

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(dst, index=False)
    print(f"Generated {len(df)} student records")
    print(f"Saved to {dst}")

    # Print summary statistics
    print("\nSummary statistics:")
    print(df.describe().round(2))

    return 0


# %%
if __name__ == "__main__":
    exit(main())
