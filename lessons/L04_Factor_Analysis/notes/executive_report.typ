#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
)

#set par(justify: true)

#set heading(numbering: "1.")

#align(center)[
  #text(size: 20pt, weight: "bold")[
    Executive Report
  ]

  #v(0.5em)

  #text(size: 16pt)[
    Factor Analysis vs Principal Component Analysis
  ]

  #v(0.3em)

  #text(size: 14pt)[
    Recovering a Simulated Construct Structure: A Worked Example
  ]

  #v(1em)

  #line(length: 80%, stroke: 1pt)

  #v(1em)

  #text(size: 10pt, style: "italic")[
    Analysis of 200 Students Across Nine Assessment Variables
  ]
]

#v(2em)

= Executive Summary

#block(fill: rgb("#FFF3E0"), inset: 10pt, radius: 4pt, width: 100%)[
  *Synthetic data notice:* This report analyzes a single *simulated* sample (`fetch_educational.py`) generated from a known three-factor model with factor correlations of 0.20-0.30. The "students" and their scores are not real. The purpose of this report is to demonstrate whether Factor Analysis and PCA can *recover a structure we already know*, not to validate a real assessment instrument. Genuine construct validation requires independent real-world data, replication, and typically a confirmatory factor analysis (CFA).
]

This report presents a worked example examining whether nine simulated educational assessment variables recover three distinct underlying constructs: Quantitative Reasoning, Verbal Ability, and Interpersonal Skills. The analysis employs Factor Analysis (FA) with *Promax* (oblique) rotation as the primary method -- because the data were simulated with correlated factors, an oblique rotation is the appropriate default -- with Varimax (orthogonal) rotation fit alongside it as a sensitivity comparison, plus Principal Component Analysis (PCA) on data from 200 simulated students.

*Key Finding:* Factor Analysis with Promax rotation recovers the three simulated constructs with high clarity: each assessment loads strongly (>0.87) on exactly one factor, and the recovered factor correlations (Φ ≈ 0.19-0.36) are consistent with the 0.20-0.30 correlations used to generate the data.

*Recommendation:* This worked example illustrates the mechanics of FA/PCA and rotation choice. It does not, by itself, validate any real assessment battery; that would require independent data and further evidence (e.g. CFA, reliability, criterion validity).

= Research Question

*Can Factor Analysis and PCA recover the three simulated underlying constructs (Quantitative, Verbal, Interpersonal) in this synthetic dataset, and how does the choice between oblique and orthogonal rotation affect the answer?*

This worked example illustrates:
- How rotation choice (oblique vs. orthogonal) interacts with a data-generating process that has correlated factors
- How to read loadings, communalities, and a factor-correlation matrix
- The difference between a method *recovering a known structure* and *validating a real construct*

= Dataset Description

*Sample:* 200 simulated students (`fetch_educational.py`, synthetic data)

*Variables:* Nine assessment scores across three theoretical domains:
- *Quantitative Domain:* MathScore, AlgebraScore, GeometryScore
- *Verbal Domain:* ReadingComp, Vocabulary, Writing
- *Interpersonal Domain:* Collaboration, Leadership, Communication

*Data-generating process:* The three latent factors were simulated as *correlated* (0.20-0.30), not independent -- this is the key fact that motivates using an oblique rotation as the primary analysis below.

All variables were standardized (mean=0, std=1) prior to analysis to ensure equal contribution regardless of original measurement scales.

#pagebreak()

= Methodological Approach

== Factor Analysis

Factor Analysis (FA) is a statistical technique that identifies latent constructs underlying observed variables. Unlike PCA, FA distinguishes between:
- *Common variance:* Shared among variables, explained by factors
- *Unique variance:* Specific to each variable, combining variable-specific true variance and measurement error (these two cannot be separated from uniqueness alone)

*Key Parameters:*
- Extraction method: Principal Axis Factoring (PAF)
- Number of factors: 3 (theoretical expectation)
- Rotation: *Promax* (oblique, primary analysis -- allows correlated factors, matching the data-generating process), with *Varimax* (orthogonal) fit as a sensitivity comparison

== Principal Component Analysis

PCA identifies linear combinations of variables that capture maximum variance. It does not distinguish common from unique variance, using all available variance.

*Key Parameters:*
- All components extracted (9 total)
- No rotation applied
- Focus on first 3 components for comparison

== Statistical Assumptions

Two critical tests verified data suitability for Factor Analysis:

*Bartlett's Test of Sphericity*
- Chi-square = 1393.938, p < 0.001
- Confirms variables are sufficiently correlated for FA
- Rejects hypothesis that correlation matrix is an identity matrix

*Kaiser-Meyer-Olkin (KMO) Measure*
- Overall KMO = 0.799 (Acceptable)
- Individual MSA values: 0.728 to 0.850 (all adequate)
- Confirms sampling adequacy for factor extraction

*Conclusion:* Both tests passed. Data meets requirements for meaningful Factor Analysis.

#pagebreak()

= Factor Analysis Results

== Eigenvalues and Variance Explained

Three factors extracted with eigenvalues exceeding the Kaiser criterion (>1.0):

#table(
  columns: (auto, auto, auto),
  align: center,
  [*Factor*], [*Eigenvalue*], [*Cumulative Variance*],
  [1], [4.041], [44.9%],
  [2], [2.124], [68.5%],
  [3], [1.635], [86.7%]
)

The three retained factors' communalities sum to 86.7% of the *total standardized variance* (sum of communalities / 9 variables) -- not "86.7% of common variance," which would be a different (and here, trivially 100%) quantity. Remaining factors have eigenvalues \<1.0, indicating they capture primarily noise.

== Communalities

All variables showed high communalities (h²), indicating they are well-explained by the three-factor model:

#table(
  columns: (auto, auto, auto),
  [*Variable*], [*Communality (h²)*], [*Uniqueness (u²)*],
  [MathScore], [0.879], [0.121],
  [AlgebraScore], [0.905], [0.095],
  [GeometryScore], [0.848], [0.152],
  [ReadingComp], [0.852], [0.148],
  [Vocabulary], [0.838], [0.162],
  [Writing], [0.852], [0.148],
  [Collaboration], [0.874], [0.126],
  [Leadership], [0.873], [0.127],
  [Communication], [0.880], [0.120]
)

*Average communality = 0.867*: on average, 86.7% of each variable's variance is shared with the other variables through the three common factors, and 13.3% is unique to that variable (specific variance plus measurement error, not separable from communalities alone).

#pagebreak()

== Unrotated Factor Interpretation

The unrotated solution revealed three problematic patterns:

*Factor 1: General Educational Ability*
- All nine variables load positively (0.577 to 0.737)
- Represents overall academic competence, not specific abilities
- Problem: Cannot distinguish which skills a student possesses

*Factor 2: Interpersonal vs. Cognitive Contrast*
- Positive loadings: Collaboration (0.706), Leadership (0.710), Communication (0.650)
- Negative loadings: Math (-0.478), Algebra (-0.438), Geometry (-0.464)
- Problem: Bipolar structure creates ambiguous interpretation

*Factor 3: Quantitative vs. Verbal Contrast*
- Positive loadings: Math (0.422), Algebra (0.555), Geometry (0.338)
- Negative loadings: Reading (-0.527), Vocabulary (-0.588), Writing (-0.558)
- Problem: Suggests students cannot excel at both domains

*Critical Issues:*
1. Lack of simple structure (variables load on multiple factors)
2. Bipolar factors with confusing negative loadings
3. Not theory-aligned (expected three positive constructs, got contrasts + general factor)
4. Low practical utility for subscale creation

*Explanation:* Unrotated factors maximize variance extraction, not interpretability. This is why rotation is essential.

#pagebreak()

== Rotated Factor Solution (Promax, Primary)

Promax rotation transformed factors to achieve simple structure while allowing them to correlate:

#table(
  columns: (auto, auto, auto, auto),
  align: center,
  [*Variable*], [*Factor 1*], [*Factor 2*], [*Factor 3*],
  [MathScore], [*0.931*], [-0.018], [0.027],
  [AlgebraScore], [*0.984*], [0.036], [-0.134],
  [GeometryScore], [*0.874*], [-0.018], [0.121],
  [ReadingComp], [0.051], [-0.011], [*0.906*],
  [Vocabulary], [-0.032], [-0.032], [*0.934*],
  [Writing], [-0.023], [0.047], [*0.917*],
  [Collaboration], [0.001], [*0.945*], [-0.041],
  [Leadership], [-0.001], [*0.947*], [-0.052],
  [Communication], [0.002], [*0.902*], [0.110]
)

*Bold values* indicate salient loadings (>0.4 threshold).

*Factor Correlation Matrix (Φ):* Unlike Varimax, Promax does not force factors to be uncorrelated:

#table(
  columns: (auto, auto, auto, auto),
  align: center,
  [], [*Factor 1*], [*Factor 2*], [*Factor 3*],
  [*Factor 1*], [1.000], [0.194], [0.357],
  [*Factor 2*], [0.194], [1.000], [0.271],
  [*Factor 3*], [0.357], [0.271], [1.000],
)

These recovered correlations (0.19-0.36) are close to the 0.20-0.30 correlations actually used to simulate the data -- evidence that Promax is recovering the true generating structure, whereas Varimax's assumption of zero factor correlation would misrepresent it.

*Interpretation:*
- *Factor 1 = Quantitative Reasoning:* Math, Algebra, Geometry (loadings >0.87)
- *Factor 2 = Interpersonal Skills:* Collaboration, Leadership, Communication (loadings >0.90)
- *Factor 3 = Verbal Ability:* Reading, Vocabulary, Writing (loadings >0.89)

*Result:* Each variable loads strongly on exactly one factor, with small cross-loadings. In this synthetic sample, this recovers the theoretical three-construct model used to generate the data -- it demonstrates the method, and is not itself evidence that a *real* assessment battery would show the same pattern.

== Sensitivity Comparison: Varimax (Orthogonal)

#table(
  columns: (auto, auto, auto, auto),
  align: center,
  [*Variable*], [*Factor 1*], [*Factor 2*], [*Factor 3*],
  [MathScore], [*0.916*], [0.067], [0.188],
  [AlgebraScore], [*0.944*], [0.105], [0.048],
  [GeometryScore], [*0.877*], [0.075], [0.270],
  [ReadingComp], [0.210], [0.113], [*0.892*],
  [Vocabulary], [0.131], [0.089], [*0.901*],
  [Writing], [0.144], [0.166], [*0.897*],
  [Collaboration], [0.075], [*0.928*], [0.081],
  [Leadership], [0.072], [*0.929*], [0.071],
  [Communication], [0.099], [*0.906*], [0.223]
)

Varimax gives similar loadings and the same variable groupings here, but by construction reports zero correlation between factors (Φ = I), which does not match how this dataset was actually generated. We report it only as a sensitivity check, not as the primary result.

#pagebreak()

= Principal Component Analysis Results

== Eigenvalues and Variance

PCA extracted nine components with the following eigenvalue structure:

#table(
  columns: (auto, auto, auto),
  align: center,
  [*Component*], [*Eigenvalue*], [*Cumulative Variance*],
  [1], [4.062], [44.9%],
  [2], [2.135], [68.5%],
  [3], [1.643], [86.7%],
  [4-9], [\<1.0], [13.3%]
)

PCA eigenvalues closely match FA eigenvalues for the first three components, confirming three-dimensional structure. Components 4-9 have eigenvalues \<1.0 (Kaiser criterion), indicating they capture mostly non-shared (unique + residual) variance -- not necessarily "measurement noise" specifically.

== Component Loadings

Unlike FA's rotated solution, PCA loadings (unrotated) show more distributed patterns:

*PC1 (44.9% variance):* All variables load moderately (0.28-0.37), suggesting a general factor

*PC2 (23.6% variance):* Contrasts interpersonal skills (positive) vs. cognitive skills (negative)

*PC3 (18.2% variance):* Contrasts verbal (positive) vs. quantitative (negative)

*Observation:* Without rotation, PCA components are harder to interpret as distinct constructs. This demonstrates why rotation is useful for interpretability -- it does not, by itself, demonstrate construct validity.

#pagebreak()

= Comparative Analysis: FA vs PCA

#table(
  columns: (auto, auto, auto),
  [*Criterion*], [*Factor Analysis*], [*Principal Component Analysis*],
  [Variance explained], [86.7% of total standardized variance, via common factors], [86.7% of total variance, via first 3 components],
  [Rotation applied], [Yes (Promax primary, Varimax comparison)], [No],
  [Factor/component correlation], [Nonzero (Φ ≈ 0.19-0.36, Promax)], [Zero by construction],
  [Simple structure], [Clear (each variable loads mainly on one factor)], [Absent (distributed loadings)],
  [Interpretability], [High (clear factor labels in this simulation)], [Moderate (requires interpretation)],
  [Recovers known simulated structure], [Yes, closely], [Dimensionality yes; construct labels less clear],
  [Practical utility], [High (enables subscale creation), pending real-data validation], [Moderate (less clear assignments)]
)

*Key Insight:* Both methods identify three-dimensional structure. FA with rotation provides better interpretability here, and Promax additionally recovers the known factor correlations that Varimax cannot represent. This shows FA is well-suited to this *kind* of problem (correlated latent constructs) -- it is not, on its own, "construct validation."

= Visual Results

Three visualizations were generated to support findings:

*1. Scree Plots (fa_scree.png)*
- Clear "elbow" after third component/factor in both FA and PCA
- Eigenvalues 4-9 below Kaiser criterion (1.0)
- Visual confirmation of three-factor retention

*2. Loading Heatmaps (fa_loadings.png)*
- Unrotated loadings show mixed patterns with moderate values
- Rotated loadings show distinct blocks with strong values
- Dramatic visual demonstration of rotation's impact on interpretability

*3. PCA Biplot (pca_biplot.png)*
- Variable arrows cluster into three groups (Quantitative, Verbal, Interpersonal)
- Student scores distributed across PC1 and PC2
- Visual confirmation of three-domain structure

#pagebreak()

= Conclusions

== Answer to Research Question

*YES, in this synthetic sample, Factor Analysis recovers the three simulated constructs.*

Factor Analysis with Promax rotation (primary analysis):
- Each assessment loads strongly (>0.87) on exactly one factor
- Clear simple structure achieved
- Factors align precisely with the constructs used to generate the data:
  - Quantitative Reasoning (Math, Algebra, Geometry)
  - Interpersonal Skills (Collaboration, Leadership, Communication)
  - Verbal Ability (Reading, Vocabulary, Writing)
- Recovered factor correlations (Φ ≈ 0.19-0.36) are consistent with the 0.20-0.30 correlations used to generate the data
- High communalities (average = 0.867) mean most of each variable's variance is shared through the common factors -- this is not the same as "minimal measurement error," since uniqueness bundles specific variance and error together
- 86.7% of total standardized variance is attributed to the three common factors

This is a demonstration that the method works as expected on data with a known structure. It is *not*, by itself, evidence about any real assessment battery.

== What This Does and Does Not Establish

This worked example shows that FA with an appropriate (oblique) rotation can recover a known simulated structure, including its factor correlations. It does *not* establish:
- That any real assessment battery has this structure
- That the specific assessment items are valid measures of these constructs
- That the constructs are "separable and measurable" in a real population

Establishing those claims for a real instrument requires independent data, replication across samples, and typically a confirmatory factor analysis (CFA) that tests this specific three-factor model against the data rather than exploring for structure.

== Methodological Lessons

*Rotation choice is a modeling decision, not a default*
- Unrotated factors maximized variance but lacked interpretability
- Both Promax and Varimax achieve interpretable simple structure here, but only Promax's nonzero Φ matches how the data was actually generated
- When factors could plausibly correlate (the usual case for psychological/educational constructs), an oblique rotation should be the default choice, with an orthogonal rotation reported only as a comparison

*FA vs PCA Selection*
- Both methods identified three dimensions
- FA with rotation provided clearer factor interpretation and, via Promax, correctly represented factor correlation
- FA's separation of common and unique variance is useful when a latent-variable model is the actual object of interest

*Assumption Testing*
- KMO and Bartlett's tests confirmed data suitability
- All variables showed adequate sampling adequacy
- Proper assumption testing prevents meaningless results, but passing them does not by itself validate substantive conclusions

#pagebreak()

= Recommendations

== If This Structure Were Confirmed on Real Data

These applications would follow *only after* independent, real-data confirmation (e.g. CFA, replication) of a three-factor structure -- they are not licensed by this synthetic-data demonstration alone:

*1. Subscale Creation*
- Three subscale scores: Quantitative, Verbal, Interpersonal
- Each subscale with three indicators with high loadings
- Factor scores or simple sum scores for student reporting

*2. Diagnostic Assessment*
- Identify student strengths and weaknesses across three domains
- Target interventions to specific ability areas rather than general "academic support"
- Track domain-specific progress over time

*3. Program Evaluation*
- Assess effectiveness of interventions targeting specific constructs
- Determine whether programs improve targeted abilities without affecting others

*4. Research Applications*
- Motivate further construct-validity studies using real measures
- Motivate hypotheses about domain-specific effects, to be tested on independent data

== Cautions and Limitations

*This Is Synthetic Data*
- All 200 "students" and their scores were simulated from a known three-factor model
- Recovering the known structure demonstrates the method; it is not itself evidence about any real population
- None of the conclusions above should be applied to real assessment data without independent verification

*Sample Specificity (if applied to real data)*
- A single sample from one context is never sufficient for validation
- Replication with different, independent samples is required
- Factor structure may vary across populations

*Unique Variance*
- Despite high communalities (average = 0.867), 13.3% of variance is unique to each variable
- Unique variance combines measurement error and variable-specific true variance -- these cannot be separated from communalities/uniqueness alone
- Reliability analysis (e.g. Cronbach's alpha) is needed to estimate measurement error specifically

== Future Directions

*Confirmatory Factor Analysis*
- On real data, test this specific three-factor model using structural equation modeling
- Evaluate model fit with chi-square, CFI, RMSEA indices
- Compare alternative models (e.g., single-factor, hierarchical)

*Invariance Testing*
- Test whether factor structure holds across groups (gender, age, ethnicity)
- Establish measurement equivalence before making group comparisons

*Rotation Sensitivity*
- This report already uses Promax (oblique) as primary, given the correlated factor structure
- On real data, also compare other oblique rotations (e.g. Oblimin) to check robustness of the recovered Φ matrix

*Predictive Validity*
- On real data, examine whether subscales predict relevant outcomes (grades, career success)
- Establish criterion-related validity
- Test incremental validity of three separate scores vs. composite

#v(2em)

#line(length: 100%, stroke: 0.5pt)

#align(center)[
  #text(size: 9pt, style: "italic")[
    Report prepared from statistical analysis conducted using Python (factor_analyzer, scikit-learn)

    Dataset: Synthetic data, N=200 simulated students, p=9 assessment variables

    Methods: Factor Analysis (Principal Axis Factoring, Promax rotation primary, Varimax comparison), Principal Component Analysis
  ]
]
