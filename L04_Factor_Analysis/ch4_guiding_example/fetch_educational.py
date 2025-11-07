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
import csv
import os
import warnings

# Suppress numpy MINGW-W64 warnings on Windows
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*MINGW-W64.*')

# %%
import numpy as np


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

    # Create data arrays with realistic educational score ranges
    math_scores = np.round(scale_to_100(math_score.flatten(), 72, 13), 1)
    algebra_scores = np.round(scale_to_100(algebra_score.flatten(), 68, 14), 1)
    geometry_scores = np.round(scale_to_100(geometry_score.flatten(), 70, 12), 1)
    reading_comp_scores = np.round(scale_to_100(reading_comp.flatten(), 73, 11), 1)
    vocabulary_scores = np.round(scale_to_100(vocabulary.flatten(), 75, 13), 1)
    writing_scores = np.round(scale_to_100(writing.flatten(), 71, 12), 1)
    collaboration_scores = np.round(scale_to_100(collaboration.flatten(), 76, 10), 1)
    leadership_scores = np.round(scale_to_100(leadership.flatten(), 69, 13), 1)
    communication_scores = np.round(scale_to_100(communication.flatten(), 74, 11), 1)

    # Save to CSV
    with open(dst, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Student', 'MathScore', 'AlgebraScore', 'GeometryScore',
                        'ReadingComp', 'Vocabulary', 'Writing',
                        'Collaboration', 'Leadership', 'Communication'])
        # Write data rows
        for i in range(n_students):
            writer.writerow([
                student_ids[i],
                math_scores[i],
                algebra_scores[i],
                geometry_scores[i],
                reading_comp_scores[i],
                vocabulary_scores[i],
                writing_scores[i],
                collaboration_scores[i],
                leadership_scores[i],
                communication_scores[i]
            ])

    print(f"Generated {n_students} student records")
    print(f"Saved to {dst}")

    # Print summary statistics
    print("\nSummary statistics:")
    all_scores = np.column_stack([
        math_scores, algebra_scores, geometry_scores,
        reading_comp_scores, vocabulary_scores, writing_scores,
        collaboration_scores, leadership_scores, communication_scores
    ])
    print(f"  Mean: {np.mean(all_scores, axis=0).round(2)}")
    print(f"  Std:  {np.std(all_scores, axis=0).round(2)}")
    print(f"  Min:  {np.min(all_scores, axis=0).round(2)}")
    print(f"  Max:  {np.max(all_scores, axis=0).round(2)}")

    return 0


# %%
if __name__ == "__main__":
    exit(main())
