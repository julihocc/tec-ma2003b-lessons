#!/usr/bin/env python3
"""
Test All Factor Analysis Snippets
Runs all snippet notebook code cells in non-interactive mode, checks that they
execute cleanly, and asserts on the key numerical results each snippet teaches.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Ensure non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None


def check_pca_basic(ns):
    """01_pca_basic_example: sklearn PCA must match the manual eigendecomposition
    once component signs are aligned (PCA components are unique only up to sign)."""
    assert ns["raw_diff"] > 1.0, (
        "Expected the unaligned comparison to show a sign-flip mismatch; "
        "if this no longer reproduces, the sign-ambiguity explanation is stale."
    )
    assert ns["aligned_diff"] < 1e-8, (
        f"Sign-aligned PCA transform should match the manual eigendecomposition, "
        f"got max abs error {ns['aligned_diff']}"
    )
    ratio = ns["variance_ratio"]
    assert abs(ratio[0] - 0.75) < 1e-6 and abs(ratio[1] - 0.25) < 1e-6, (
        f"Expected explained variance ratios [0.75, 0.25] for this fixed dataset, got {ratio}"
    )


def check_component_retention(ns):
    """02_component_retention: cumulative variance ratio must reach 1.0, and both
    retention heuristics must select a valid number of components."""
    cumvar = ns["cumvar"]
    assert abs(cumvar[-1] - 1.0) < 1e-8, (
        f"Cumulative explained variance ratio should sum to 1.0, got {cumvar[-1]}"
    )
    n_features = ns["X"].shape[1]
    assert 1 <= ns["n_kaiser"] <= n_features, "Kaiser criterion picked an invalid component count"
    assert 1 <= ns["n_cumvar"] <= n_features, "Cumulative-variance criterion picked an invalid component count"


def check_factor_analysis_basic(ns):
    """03_factor_analysis_basic: FactorAnalyzer should recover the true single-factor
    loadings (up to sign) and communalities/uniqueness must sum to 1 per variable."""
    true_loadings = ns["true_loadings"]
    recovered = np.abs(ns["loadings"].flatten())
    max_loading_error = np.max(np.abs(true_loadings - recovered))
    assert max_loading_error < 0.15, (
        f"Recovered loadings should be close to the true loadings, "
        f"max abs error {max_loading_error:.3f}"
    )
    h2 = np.asarray(ns["communalities"])
    psi = np.asarray(ns["uniqueness"])
    assert np.allclose(h2 + psi, 1.0, atol=1e-6), (
        "Communalities + uniqueness should equal 1.0 for every variable"
    )
    correlation = ns["correlation"]
    assert abs(correlation) > 0.8, (
        f"Estimated factor scores should correlate strongly with the true latent factor, "
        f"got r={correlation:.3f}"
    )


def check_factor_rotation(ns):
    """04_factor_rotation: an orthogonal rotation (varimax) must preserve each
    variable's total communality, since it only rotates axes within the same
    retained factor space."""
    unrotated_h2 = (ns["loadings_before"] ** 2).sum(axis=1)
    varimax_h2 = (ns["loadings_varimax"] ** 2).sum(axis=1)
    assert np.allclose(unrotated_h2, varimax_h2, atol=1e-6), (
        "Varimax is an orthogonal rotation and must preserve each variable's communality"
    )


def check_complete_workflow(ns):
    """05_complete_workflow: KMO and Bartlett's test statistics must fall in their
    valid ranges, and at least one factor must be retained."""
    assert 0.0 <= ns["kmo_model"] <= 1.0, f"KMO must be in [0, 1], got {ns['kmo_model']}"
    assert 0.0 <= ns["p_value"] <= 1.0, f"Bartlett p-value must be in [0, 1], got {ns['p_value']}"
    assert ns["n_factors"] >= 1, "Kaiser criterion should retain at least one factor"


SNIPPET_CHECKS = {
    "01_pca_basic_example.ipynb": check_pca_basic,
    "02_component_retention.ipynb": check_component_retention,
    "03_factor_analysis_basic.ipynb": check_factor_analysis_basic,
    "04_factor_rotation.ipynb": check_factor_rotation,
    "05_complete_workflow.ipynb": check_complete_workflow,
}


def test_notebook(nb_path):
    """Execute a notebook's code cells and run its outcome assertions."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    if not code_cells:
        print(f"FAIL {nb_path.name} - NO CODE CELLS FOUND")
        return False

    full_code = "\n\n# --- CELL ---\n\n".join(code_cells)
    exec_namespace = {"__name__": "__main__"}

    try:
        exec(compile(full_code, str(nb_path), "exec"), exec_namespace)
    except Exception as e:
        print(f"FAIL {nb_path.name} - execution error")
        print(f"   Error: {type(e).__name__}: {e}")
        return False

    check = SNIPPET_CHECKS.get(nb_path.name)
    if check is not None:
        try:
            check(exec_namespace)
        except AssertionError as e:
            print(f"FAIL {nb_path.name} - assertion failed")
            print(f"   {e}")
            return False

    print(f"PASS {nb_path.name}")
    return True


def main():
    """Run all snippet tests."""
    print("Testing Factor Analysis Jupyter Notebook Snippets (Non-Interactive)")
    print("=" * 50)

    snippets_dir = Path(__file__).resolve().parent
    snippet_files = [snippets_dir / name for name in SNIPPET_CHECKS]

    success_count = 0
    total_count = len(snippet_files)

    for snippet in snippet_files:
        if snippet.exists():
            if test_notebook(snippet):
                success_count += 1
        else:
            print(f"FAIL {snippet.name} - FILE NOT FOUND")

    print("\n" + "=" * 50)
    print(f"Test Results: {success_count}/{total_count} snippets passed")

    if success_count == total_count:
        print("All snippets execute correctly and match their expected results.")
        return 0
    else:
        print("Some snippets have issues. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
