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
    Educational Assessment Construct Validation Study
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

This report presents findings from a comprehensive statistical analysis examining whether nine educational assessment variables measure three distinct underlying constructs: Quantitative Reasoning, Verbal Ability, and Interpersonal Skills. The analysis employed both Factor Analysis (FA) with Varimax rotation and Principal Component Analysis (PCA) on data from 200 students.

*Key Finding:* Factor Analysis with rotation provides definitive evidence that the nine assessments measure three distinct constructs with exceptional clarity. Each assessment loads strongly (>0.87) on exactly one factor, achieving perfect simple structure.

*Recommendation:* The assessment battery successfully captures three separate ability domains and can be used confidently for diagnostic purposes, intervention design, and student profiling.

= Research Question

*Do the nine educational assessments measure three distinct underlying constructs (Quantitative, Verbal, Interpersonal), or do they reflect a different latent structure?*

This question has significant practical implications:
- Determining whether assessments capture separate abilities or redundant information
- Enabling targeted intervention programs for specific skill deficits
- Validating the theoretical framework underlying assessment design
- Supporting subscale creation for diagnostic reporting

= Dataset Description

*Sample:* 200 students

*Variables:* Nine assessment scores across three theoretical domains:
- *Quantitative Domain:* MathScore, AlgebraScore, GeometryScore
- *Verbal Domain:* ReadingComp, Vocabulary, Writing
- *Interpersonal Domain:* Collaboration, Leadership, Communication

All variables were standardized (mean=0, std=1) prior to analysis to ensure equal contribution regardless of original measurement scales.

#pagebreak()

= Methodological Approach

== Factor Analysis

Factor Analysis (FA) is a statistical technique that identifies latent constructs underlying observed variables. Unlike PCA, FA distinguishes between:
- *Common variance:* Shared among variables, explained by factors
- *Unique variance:* Specific to each variable plus measurement error

*Key Parameters:*
- Extraction method: Principal Axis Factoring (PAF)
- Number of factors: 3 (theoretical expectation)
- Rotation: Varimax (orthogonal rotation for simple structure)

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

The three-factor solution explains *86.7% of common variance*, with remaining factors having eigenvalues <1.0, indicating they capture primarily noise.

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

*Average communality = 0.867*, confirming minimal unique variance and strong factor relationships.

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

== Rotated Factor Solution (Varimax)

Varimax rotation transformed factors to achieve simple structure while preserving orthogonality:

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

*Bold values* indicate salient loadings (>0.4 threshold).

*Interpretation:*
- *Factor 1 = Quantitative Reasoning:* Math, Algebra, Geometry (loadings >0.87)
- *Factor 2 = Interpersonal Skills:* Collaboration, Leadership, Communication (loadings >0.90)
- *Factor 3 = Verbal Ability:* Reading, Vocabulary, Writing (loadings >0.89)

*Achievement:* Perfect simple structure. Each variable loads strongly on exactly ONE factor, with negligible cross-loadings. This confirms the theoretical three-construct model.

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
  [4-9], [<1.0], [13.3%]
)

PCA eigenvalues closely match FA eigenvalues for the first three components, confirming three-dimensional structure. Components 4-9 have eigenvalues <1.0 (Kaiser criterion), indicating they capture mostly measurement noise.

== Component Loadings

Unlike FA's rotated solution, PCA loadings (unrotated) show more distributed patterns:

*PC1 (44.9% variance):* All variables load moderately (0.28-0.37), suggesting a general factor

*PC2 (23.6% variance):* Contrasts interpersonal skills (positive) vs. cognitive skills (negative)

*PC3 (18.2% variance):* Contrasts verbal (positive) vs. quantitative (negative)

*Observation:* Without rotation, PCA components are harder to interpret as distinct constructs. This demonstrates why FA with rotation is preferred for construct validation.

#pagebreak()

= Comparative Analysis: FA vs PCA

#table(
  columns: (auto, auto, auto),
  [*Criterion*], [*Factor Analysis*], [*Principal Component Analysis*],
  [Variance explained], [86.7% of common variance], [86.7% of total variance],
  [Rotation applied], [Yes (Varimax)], [No],
  [Simple structure], [Perfect (each variable loads on one factor)], [Absent (distributed loadings)],
  [Interpretability], [Excellent (clear factor labels)], [Moderate (requires interpretation)],
  [Construct validation], [Strongly supports three-construct model], [Confirms dimensionality, less clear constructs],
  [Practical utility], [High (enables subscale creation)], [Moderate (less clear assignments)]
)

*Key Insight:* Both methods identify three-dimensional structure, but FA with rotation provides superior interpretability and construct validation. FA's ability to separate common from unique variance and achieve simple structure through rotation makes it the preferred method for this application.

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

*YES, the nine assessments measure three distinct constructs.*

Factor Analysis with Varimax rotation provides definitive evidence:
- Each assessment loads strongly (>0.87) on exactly one factor
- Perfect simple structure achieved
- Factors align precisely with theoretical expectations:
  - Quantitative Reasoning (Math, Algebra, Geometry)
  - Interpersonal Skills (Collaboration, Leadership, Communication)
  - Verbal Ability (Reading, Vocabulary, Writing)
- High communalities (average = 0.867) confirm minimal measurement error
- 86.7% of common variance explained

== Validation of Assessment Design

The assessment battery successfully captures three separate ability domains with minimal redundancy. This validates:
- The theoretical framework underlying assessment design
- The selection of specific assessment variables
- The assumption that these domains are separable and measurable

== Methodological Lessons

*Importance of Rotation*
- Unrotated factors maximized variance but lacked interpretability
- Rotation transformed factors to achieve simple structure without changing communalities or total variance
- Rotation is essential for construct validation applications

*FA vs PCA Selection*
- Both methods identified three dimensions
- FA with rotation provided superior construct interpretation
- FA's separation of common and unique variance is advantageous for measurement validation

*Assumption Testing*
- KMO and Bartlett's tests confirmed data suitability
- All variables showed adequate sampling adequacy
- Proper assumption testing prevents meaningless results

#pagebreak()

= Recommendations

== Practical Applications

The validated three-factor structure enables:

*1. Subscale Creation*
- Create three subscale scores: Quantitative, Verbal, Interpersonal
- Each subscale has three indicators with high loadings (>0.87)
- Use factor scores or simple sum scores for student reporting

*2. Diagnostic Assessment*
- Identify student strengths and weaknesses across three domains
- Target interventions to specific ability areas rather than general "academic support"
- Track domain-specific progress over time

*3. Program Evaluation*
- Assess effectiveness of interventions targeting specific constructs
- Determine whether programs improve targeted abilities without affecting others
- Validate construct-specific theories of change

*4. Research Applications*
- Establish construct validity for studies using these measures
- Support theoretical claims about ability structure
- Enable more precise hypothesis testing about domain-specific effects

== Cautions and Limitations

*Sample Specificity*
- Results based on 200 students from one context
- Replication with different samples recommended
- Factor structure may vary across populations

*Temporal Stability*
- Analysis represents one time point
- Longitudinal validation recommended
- Factor structure should be confirmed across developmental stages

*Measurement Error*
- Despite high communalities (average = 0.867), 13.3% variance remains unexplained
- Unique variance includes both measurement error and construct-specific variance
- Consider reliability analysis for individual assessments

== Future Directions

*Confirmatory Factor Analysis*
- Test the three-factor model using structural equation modeling
- Evaluate model fit with chi-square, CFI, RMSEA indices
- Compare alternative models (e.g., single-factor, hierarchical)

*Invariance Testing*
- Test whether factor structure holds across groups (gender, age, ethnicity)
- Establish measurement equivalence before making group comparisons

*Oblique Rotation*
- Explore whether factors are truly uncorrelated
- Consider Promax or other oblique rotations
- Examine factor correlations if theoretically relevant

*Predictive Validity*
- Examine whether subscales predict relevant outcomes (grades, career success)
- Establish criterion-related validity
- Test incremental validity of three separate scores vs. composite

#v(2em)

#line(length: 100%, stroke: 0.5pt)

#align(center)[
  #text(size: 9pt, style: "italic")[
    Report prepared from statistical analysis conducted using Python (factor_analyzer, scikit-learn)

    Dataset: N=200 students, p=9 assessment variables

    Methods: Factor Analysis (Principal Axis Factoring, Varimax rotation), Principal Component Analysis
  ]
]
