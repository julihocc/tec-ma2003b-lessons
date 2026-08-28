# Mathematical Content Audit

**Date:** 2026-08-27  
**Overall assessment:** **Needs revision**

## Scope and Validation

This audit reviews the mathematical and statistical content of Lessons L01–L07 across Typst notes, presentations, primary notebooks, data generators, and data dictionaries. It checks formulas, assumptions, interpretations, internal consistency, and whether conclusions are supported by the analyses. It is a source-level audit, not a proof-by-proof verification of every exercise.

Runtime validation succeeded:

- `uv run python scripts/run_notebooks.py`: **7/7 notebooks passed** from clean kernels.
- `uv run python lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py`: **5/5 snippets passed**.

Passing execution does not resolve the substantive issues below. Independent checks were also performed for Box's M, parallel analysis, and the LDA/QDA cross-validation workflow.

## Priority Findings

### 1. L07 uses an invalid Box's M decision rule — High

`lessons/L07_Multivariate_Regression/notebook/health_risk_analysis.ipynb` compares the raw Box's M statistic with an arbitrary cutoff of 30, then treats the result as confirmation that covariance matrices are equal. The presentation repeats this rule and describes MANOVA as validated.

Box's M requires a sampling-distribution approximation, degrees of freedom, and a p-value; a universal raw-statistic threshold is not valid. Recalculation gives raw \(M=8.494\), corrected \(\chi^2=8.457\), \(df=10\), and \(p=0.584\). The correct conclusion is that the test **does not reject** equal covariance matrices—not that equality has been proven or MANOVA made trustworthy.

**Action:** implement the corrected test, report its p-value and approximation, and replace “assumption satisfied/validated” with cautious fail-to-reject language.

### 2. L07's Hotelling test is circular — High

The generator creates `cvd_risk_high` from a score containing systolic blood pressure, cholesterol, and glucose (`data/fetch_health_data.py`, lines 126–141). The notebook then uses that derived label to test whether the same variables differ between its groups. The significant result is therefore partly built into the grouping rule; its p-value is not evidence of an independently discovered group difference.

**Action:** use an externally defined outcome or treatment group for inferential testing. If the current simulation is retained, describe the result only as recovery of the data-generating rule.

### 3. L07 presents synthetic results as clinical evidence — High

The lesson README and data dictionary identify the data as synthetic, but the presentation states that the methods use “real healthcare data” and makes clinical-impact claims such as identifying protective factors and prevention targets. Logistic coefficient magnitude also cannot establish the “strongest” factor across predictors measured in different units, and an association is not a causal protective effect.

**Action:** label the data as synthetic on every output, remove real-world clinical and causal claims, and compare standardized effects or meaningful unit contrasts when ranking associations.

### 4. L04 forces orthogonal factors despite a correlated-factor generator — High

The educational-data generator assigns latent-factor correlations of 0.20–0.30 (`data/fetch_educational.py`, lines 56–63), while the primary notebook and presentation use Varimax, which constrains rotated factors to be orthogonal. The resulting claims that the abilities are distinct or separate follow partly from the chosen rotation rather than the evidence.

**Action:** make an oblique rotation such as Promax or Oblimin the primary analysis, report the factor-correlation matrix \(\Phi\), and retain Varimax only as a sensitivity comparison.

### 5. L04 overstates exploratory recovery as construct validation — High

The same synthetic sample is generated with three factors and analyzed with EFA, yet the executive report and notebook call the result “definitive evidence” and “perfect construct validation.” High communalities are also said to show minimal measurement error, although uniqueness combines specific variance and error variance and cannot isolate measurement error.

**Action:** describe this as recovery of a simulated structure. Reserve validation claims for independent data and an explicit CFA/fit-comparison design; remove the measurement-error interpretation.

### 6. L07 contains an incorrect Wilks' Lambda identity — High

`notes/multivariate_regression_notes.typ`, line 402, states

\[
\Lambda=\frac{|W|}{|T|}=\frac{|W|}{|W|+|B|}.
\]

Although \(T=W+B\), determinants are not additive. The correct expression is

\[
\Lambda=\frac{|W|}{|W+B|}.
\]

The slides use the correct form. Wilks' Lambda is a ratio of generalized variances, not a literal scalar “proportion of total variance.”

**Action:** correct the notes and align their interpretation with the slides.

## Lesson-by-Lesson Findings

### L01 — Regression Analysis

- `notes/regression_analysis_notes.typ`, line 226, treats \(100\beta_1\%\) as the exact effect in a log-linear model. The exact one-unit effect is \(100(e^{\beta_1}-1)\%\); \(100\beta_1\%\) is a small-coefficient approximation.
- The linear-log and log-log “1% change” interpretations should likewise be labelled local approximations for finite changes.
- The main OLS, ANOVA, confidence/prediction interval, VIF, Cook's distance, and HC3 formulas are otherwise consistent.

**Severity:** Medium.

### L02 — Multivariate Foundations

- `notes/multivariate_analysis_notes.typ`, line 161, calls \(D=\operatorname{diag}(\sigma_{11},\ldots,\sigma_{pp})\) a standard-deviation matrix. Those diagonal elements are variances. The following formula \(R=D^{-1/2}\Sigma D^{-1/2}\) is correct only when \(D\) is the variance diagonal.
- The notebook applies single KNN imputation and then computes Fisher correlation confidence intervals using the full row count. This treats imputed values as observed and understates uncertainty.

**Action:** rename \(D\) as the diagonal variance matrix. For inference after missing-data handling, use pairwise complete observations under a stated missingness assumption or propagate imputation uncertainty through multiple imputation or an appropriate bootstrap.

**Severity:** Medium.

### L03 — Principal Component Analysis

The core eigendecomposition, explained-variance, score, loading, and biplot discussions are sound. The parallel-analysis simulation, however, runs PCA on unstandardized random-normal samples while the empirical analysis is correlation-based. Each null sample should be standardized, or its correlation matrix should be decomposed.

An independent 2,000-replicate check found that this correction does not change the retained dimension: both implementations retain three components. Representative 95th-percentile null eigenvalues changed from `[1.298, 1.214, 1.152]` to `[1.263, 1.187, 1.130]`.

**Severity:** Low.

### L04 — Factor Analysis

In addition to the priority findings:

- `notes/factor_analysis_notes.typ`, lines 183–190, presents \(n>p\), normal data, and no perfect multicollinearity as general PCA requirements. Descriptive PCA is defined for \(n<p\), rank-deficient, and non-normal data; these conditions matter for selected inferential interpretations or numerical goals, not PCA's existence.
- The Kaiser criterion is described as using eigenvalues of a reduced correlation matrix, but the implementation uses the original correlation matrix. The conventional \(\lambda>1\) rule is based on the original correlation matrix.
- Eigenvectors are called loadings, conflicting with L03's definition of loadings as variable-component correlations \(v_j\sqrt{\lambda_j}\). Distinguish component coefficients from correlation loadings.
- “86.7% of common variance” is misleading when calculated as the retained common-factor sum of squares divided by \(p\). With standardized variables, that quantity is the share of total observed variance attributed to the retained factors.
- The notes say factor analysis tests a specified theoretical structure; that is specifically CFA. EFA explores an unknown structure.

**Severity:** High overall.

### L05 — Discriminant Analysis

- `notes/discriminant_analysis_notes.typ`, line 245, says equal-prior LDA boundaries are perpendicular bisectors of centroid-connecting lines. This is true in Euclidean coordinates only when the shared covariance is proportional to the identity. General LDA uses Mahalanobis geometry, with boundary normal \(\Sigma^{-1}(\mu_k-\mu_l)\).
- The notebook standardizes the full dataset before the train/test split and cross-validation. Preprocessing should be fitted within each training split using a pipeline. An independent rerun with fold-local scaling produced the same scores here—LDA `[1, 1, 1, 1, 0.9958]` and QDA `[1, 1, 1, 1, 1]`—but the current workflow teaches leakage and can bias results on other data.
- A table labelled “Group Means on Original Features” displays `lda.means_`, which are means in standardized feature units.
- Maximum posterior probability is Bayes-optimal only under the stated loss; the usual rule assumes equal misclassification costs (0–1 loss).

**Severity:** Medium.

### L06 — Cluster Analysis

- `notes/cluster_analysis_notes.typ`, line 497, recommends the largest gap statistic. The standard one-standard-error rule chooses the smallest \(k\) satisfying \(\operatorname{Gap}(k)\geq\operatorname{Gap}(k+1)-s_{k+1}\).
- Ward's minimum-variance/SSE interpretation requires squared Euclidean dissimilarity and should state that restriction.
- Complete linkage should not be described generally as less sensitive to outliers; its maximum-pair distance can itself be driven by an outlier. K-medoids is more robust than k-means, not “unaffected” by extreme values.

The distance, linkage, WCSS, silhouette, Davies–Bouldin, and Dunn-index formulas reviewed are otherwise consistent.

**Severity:** Medium.

### L07 — Multivariate Regression and Modeling

In addition to the three priority findings:

- `notes/multivariate_regression_notes.typ`, line 158, calls \(p/(1-p)\) the “odds ratio.” It is the odds; a ratio compares two odds. \(e^{\beta_j}\) is the per-unit odds ratio.
- Follow-up univariate ANOVAs are reported with unadjusted p-values although the notes and slides instruct readers to correct for multiple testing. The present conclusions survive Bonferroni correction, but the code and reporting should implement it.
- The claim that \(r_1^2=40.8\%\) is “shared variance between domains” overstates a first canonical correlation. It is shared variance between the first pair of canonical variates; domain-level interpretation requires variance-extracted and redundancy indices.
- The statement that PCA is the special case of CCA with identical variable sets is false: identical sets yield canonical correlations of one and do not generally recover PCA directions.
- The stated regression factor-score formula adds a \((L'\Sigma^{-1}L)^{-1}\) term. For orthogonal factors, the Thomson/regression predictor is \(\hat f=L'\Sigma^{-1}x\) (or \(\Phi L'\Sigma^{-1}x\) in the correlated-factor form). The displayed expression resembles a generalized least-squares coefficient estimator, not the regression factor score.
- Claims that MANOVA or Hotelling's \(T^2\) necessarily “increase power” should be conditional; power depends on the effect pattern, covariance, sample size, and multiplicity strategy.

**Severity:** High overall.

## Recommended Remediation Order

1. Correct L07's Box's M procedure, circular Hotelling interpretation, synthetic-data claims, Wilks formula, and factor-score formula.
2. Rework L04 around an oblique primary rotation and replace validation/measurement-error overclaims.
3. Fix L05's LDA geometry explanation, preprocessing pipeline, and mislabeled means.
4. Correct L01/L02 terminology and transformation interpretations.
5. Align L03 parallel analysis and L06 model-selection/robustness guidance with standard definitions.
6. Re-run all notebooks and snippet tests, rebuild affected Typst PDFs, and inspect notes and slides for synchronized wording.

## Reference Basis

- Wilks' Lambda definition: [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08).
- Leakage-safe preprocessing: [scikit-learn, Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
- Gap statistic and one-standard-error selection: [Tibshirani, Walther, and Hastie technical report](https://statistics.stanford.edu/technical-reports/estimating-number-clusters-dataset-gap-statistic) and [Stanford course text](https://web.stanford.edu/class/bios221/book/05-chap.html).
- Rotation choice when factors may correlate: [Watkins, *Exploratory Factor Analysis: A Guide to Best Practice*](https://journals.sagepub.com/doi/10.1177/0095798418771807).

## Confidence

**High** for the formula, code-path, data-generation, and recomputation findings. **Moderate** for pedagogical-priority judgments, which depend on the intended depth of the course. No evidence of runtime failure was found; the primary risk is that executable material gives learners mathematically inaccurate interpretations or unsupported inferential conclusions.
