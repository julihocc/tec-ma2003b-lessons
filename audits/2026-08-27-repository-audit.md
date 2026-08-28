# Repository Audit — 2026-08-27

## Executive Summary

The repository is structurally sound, reproducible at the dataset level, and has healthy Git/LFS integrity. No critical security issue was identified. However, two of the seven primary notebooks fail from a clean kernel, and a third silently skips its validation section. The current snippet test does not detect these failures.

## Findings

### High — L04 dependency incompatibility

The locked environment installs `factor-analyzer 0.5.1` with `scikit-learn 1.9.0`, as permitted by [`pyproject.toml`](../pyproject.toml). `factor-analyzer` calls the removed `force_all_finite` argument, so `lessons/L04_Factor_Analysis/notebook/educational_analysis.ipynb` fails during factor extraction. The snippet harness masks this incompatibility with a runtime monkeypatch that the primary notebook does not apply.

**Recommendation:** pin a compatible scikit-learn release or replace/patch `factor-analyzer`, regenerate `uv.lock`, and execute the primary notebook from a fresh kernel.

### High — L07 missing import

`lessons/L07_Multivariate_Regression/notebook/health_risk_analysis.ipynb` calls `Path.cwd()` without importing `Path`. It fails immediately with `NameError` when executed from a clean kernel.

**Recommendation:** add `from pathlib import Path` to the import cell and rerun all cells.

### Medium — L06 validation is silently skipped

`lessons/L06_Cluster_Analysis/notebook/customer_clustering_analysis.ipynb` loads its main dataset from `../data/` but searches only the notebook directory for `customer_data_with_labels.csv`. The labeled file exists in `../data/`, yet the notebook reports it missing and omits the Adjusted Rand Index and confusion matrix.

**Recommendation:** resolve the validation file using the same fallback logic as the main dataset and assert that it exists when validation is expected.

### Medium — Snippet testing can produce false confidence

`lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py` is an execution smoke test, not an outcome test. It treats a notebook with no code cells as passing, injects an environment-specific compatibility patch, and has no assertions for expected numerical results. The PCA example reports a raw score difference of `4.0`; component sign ambiguity is not handled or explained.

**Recommendation:** add clean-kernel execution for every primary notebook and assertions for important pedagogical results. Fail on empty notebooks and unexpected skipped sections.

### Medium-Low — Typst workflow is not self-contained

[`README.md`](../README.md) documents `typst compile` but does not include Typst installation instructions. Typst was unavailable in the audited environment, so the 16 `.typ` sources could not be compiled. Nine sources currently have no corresponding committed PDF.

### Low — Saved notebook output is stale

Several notebooks contain old absolute Windows/Linux development paths, compatibility warnings, and kernel metadata from Python 3.10–3.12. The project requires Python 3.11 or newer. These outputs should be cleared or regenerated using the locked environment.

## Validation Evidence

- Five of seven primary notebooks executed successfully in isolated copies; L04 and L07 failed as described above.
- All seven dataset generators succeeded in a temporary copy.
- Every regenerated CSV was byte-identical to its tracked counterpart.
- All eight Python files parsed successfully.
- `uv lock --check`, core dependency imports, and `uv pip check` passed.
- The documented snippet harness reported 5/5 successful executions, subject to the limitations above.
- `git fsck`, `git lfs fsck`, and `git diff --check` passed.
- No broken local Markdown links or obvious secret patterns were found.
- The worktree was clean before the audit; a pre-existing `.gitignore` modification was present when this report was written and was not changed.

## Residual Risks

Typst compilation and PDF visual layout were not verified because Typst was not installed. No external dependency-vulnerability/CVE scan was performed. Basic secret-pattern scanning is not a substitute for a dedicated secret scanner.

## Recommended Remediation Order

1. Fix L07's missing import and L06's validation path.
2. Resolve the L04 dependency incompatibility in project metadata and the lockfile.
3. Add automated clean-kernel execution for all primary notebooks.
4. Strengthen snippet assertions and correct the PCA sign-comparison explanation.
5. Document Typst installation, compile all sources, and refresh stale notebook outputs.
