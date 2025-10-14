# Educational Assessment Data Dictionary

## Overview

This synthetic dataset contains realistic educational assessment data for 200 students across 9 educational measures. The data demonstrates three underlying latent factors with pedagogically realistic correlations, designed for Factor Analysis instruction.

## Data Structure

- **Format**: CSV file (`educational.csv`)
- **Observations**: 200 students
- **Variables**: 9 assessment scores + 1 identifier
- **Source**: Synthetic data with realistic factor structure
- **KMO Measure**: 0.799 (Good - suitable for Factor Analysis)

## Variables

### Identifier

- **Student**: Unique student identifier (format: STUD_001, STUD_002, etc.)

### Assessment Variables

All assessment variables are scaled 0-100 (typical educational score range).

#### Quantitative Reasoning Domain

- **MathScore**: General mathematics assessment
  - **Scale**: 0-100 continuous
  - **Mean**: ~72, **SD**: ~13
  - **Interpretation**: Measures general mathematical reasoning and problem-solving
  - **Primary Factor**: Quantitative Reasoning (loading ~0.80)
  - **MSA**: 0.793

- **AlgebraScore**: Algebraic reasoning and equation solving
  - **Scale**: 0-100 continuous
  - **Mean**: ~68, **SD**: ~14
  - **Interpretation**: Measures abstract symbolic reasoning
  - **Primary Factor**: Quantitative Reasoning (loading ~0.85)
  - **MSA**: 0.728

- **GeometryScore**: Spatial reasoning and geometric concepts
  - **Scale**: 0-100 continuous
  - **Mean**: ~70, **SD**: ~12
  - **Interpretation**: Measures visual-spatial mathematical ability
  - **Primary Factor**: Quantitative Reasoning (loading ~0.75)
  - **MSA**: 0.850

#### Verbal Ability Domain

- **ReadingComp**: Reading comprehension assessment
  - **Scale**: 0-100 continuous
  - **Mean**: ~73, **SD**: ~11
  - **Interpretation**: Measures understanding of written text
  - **Primary Factor**: Verbal Ability (loading ~0.82)
  - **MSA**: 0.816

- **Vocabulary**: Vocabulary knowledge assessment
  - **Scale**: 0-100 continuous
  - **Mean**: ~75, **SD**: ~13
  - **Interpretation**: Measures breadth and depth of word knowledge
  - **Primary Factor**: Verbal Ability (loading ~0.78)
  - **MSA**: 0.818

- **Writing**: Written expression and composition
  - **Scale**: 0-100 continuous
  - **Mean**: ~71, **SD**: ~12
  - **Interpretation**: Measures ability to communicate in writing
  - **Primary Factor**: Verbal Ability (loading ~0.80)
  - **MSA**: 0.815

#### Interpersonal Skills Domain

- **Collaboration**: Teamwork and collaborative skills
  - **Scale**: 0-100 continuous
  - **Mean**: ~76, **SD**: ~10
  - **Interpretation**: Measures ability to work effectively with others
  - **Primary Factor**: Interpersonal Skills (loading ~0.83)
  - **MSA**: 0.784

- **Leadership**: Leadership and initiative
  - **Scale**: 0-100 continuous
  - **Mean**: ~69, **SD**: ~13
  - **Interpretation**: Measures ability to guide and motivate others
  - **Primary Factor**: Interpersonal Skills (loading ~0.77)
  - **MSA**: 0.790

- **Communication**: Oral communication and presentation
  - **Scale**: 0-100 continuous
  - **Mean**: ~74, **SD**: ~10
  - **Interpretation**: Measures verbal expression and listening skills
  - **Primary Factor**: Interpersonal Skills (loading ~0.75)
  - **MSA**: 0.797

## Known Factor Structure

This synthetic dataset was generated with three correlated latent factors, reflecting realistic educational psychology patterns.

### Factor 1: Quantitative Reasoning

- **Theoretical Construct**: Mathematical and quantitative problem-solving ability
- **Manifest Variables**: MathScore (0.80), AlgebraScore (0.85), GeometryScore (0.75)
- **Expected Communality**: High (h² > 0.70)
- **Within-Cluster Correlation**: ~0.80 (strong)

### Factor 2: Verbal Ability

- **Theoretical Construct**: Language comprehension and expression
- **Manifest Variables**: ReadingComp (0.82), Vocabulary (0.78), Writing (0.80)
- **Expected Communality**: High (h² > 0.70)
- **Within-Cluster Correlation**: ~0.77 (strong)

### Factor 3: Interpersonal Skills

- **Theoretical Construct**: Social-emotional competence
- **Manifest Variables**: Collaboration (0.83), Leadership (0.77), Communication (0.75)
- **Expected Communality**: High (h² > 0.70)
- **Within-Cluster Correlation**: ~0.81 (strong)

### Inter-Factor Correlations

The latent factors are **correlated** (not orthogonal), reflecting realistic patterns:

- **Quantitative ↔ Verbal**: r = 0.30 (moderate positive)
- **Quantitative ↔ Interpersonal**: r = 0.20 (weak positive)
- **Verbal ↔ Interpersonal**: r = 0.25 (weak-moderate positive)

This correlation structure reflects research showing that cognitive and social-emotional abilities are related but distinct.

## Data Generation Parameters

- **Random Seed**: 42 (for reproducibility)
- **Sample Size**: 200 students
- **Latent Factors**: Three correlated factors (multivariate normal)
- **Factor Loadings**: Range 0.75-0.85 (strong primary loadings)
- **Cross-Loadings**: Range 0.05-0.20 (realistic weak secondary loadings)
- **Measurement Error**: σ = 0.4 per variable (unique error variance)
- **Score Scale**: Transformed to 0-100 range with realistic means (~70) and SDs (~12)

## Expected Analysis Results

### Factor Analysis Suitability

- **Bartlett's Test**: χ² = 1393.9, p < 0.001 (significant - suitable for FA)
- **Overall KMO**: 0.799 (Good)
- **Individual MSA**: All variables > 0.72 (all adequate)

### Expected Factor Solution

- **Number of Factors**: 3 (eigenvalues > 1.0, clear theoretical structure)
- **Variance Explained**: ~70-75% of common variance
- **Rotation Benefit**: Varimax or Promax rotation will reveal simple structure
- **Communalities**: High for all variables (h² > 0.65)
- **Simple Structure**: Each variable loads primarily on one factor

### PCA Comparison

- **Components vs Factors**: PCA will also suggest 3 components
- **Loadings**: Similar pattern but PCA includes unique variance
- **Variance Explained**: PCA will explain slightly more total variance
- **Interpretation**: FA focuses on shared variance (better for construct validation)

## Correlation Structure

### Within-Domain Correlations (Strong)

- Quantitative tests: r ≈ 0.78-0.84
- Verbal tests: r ≈ 0.74-0.78
- Interpersonal measures: r ≈ 0.80-0.81

### Cross-Domain Correlations (Weak to Moderate)

- Quantitative ↔ Verbal: r ≈ 0.20-0.38
- Quantitative ↔ Interpersonal: r ≈ 0.14-0.23
- Verbal ↔ Interpersonal: r ≈ 0.17-0.35

This pattern makes the data suitable for demonstrating:
- Factor retention decisions (clear 3-factor structure)
- Rotation benefits (moving from general to simple structure)
- Communality interpretation (distinguishing shared vs unique variance)
- Construct validation (testing theoretical models)

## Usage in Education

This dataset is designed for teaching:

- **Factor Analysis Fundamentals**: Assumptions testing (KMO, Bartlett's)
- **Factor Extraction**: Principal axis factoring, maximum likelihood
- **Factor Rotation**: Comparing orthogonal (Varimax) vs oblique (Promax) rotation
- **Interpretation**: Loading patterns, communalities, factor scores
- **Method Comparison**: PCA vs FA on same data
- **Validation**: Assessing simple structure and construct validity

## Pedagogical Advantages

1. **Realistic Structure**: No artificial "noise variables" - all measures are educationally meaningful
2. **Good Psychometric Properties**: KMO = 0.799, all MSA > 0.70
3. **Clear Factor Structure**: Three distinct but correlated domains
4. **Appropriate Complexity**: Not too simple (more than 2 factors), not overwhelming
5. **Educational Context**: Students can relate to the assessment types
6. **Realistic Correlations**: Factors correlate as they would in real educational data

## References

- Generated using multivariate normal distribution with realistic covariance structure
- Factor correlations based on educational psychology research (cognitive-social ability relationships)
- Loading patterns designed to demonstrate simple structure after rotation
- Score distributions reflect typical educational assessment ranges
