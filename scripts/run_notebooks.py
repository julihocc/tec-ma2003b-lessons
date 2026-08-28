#!/usr/bin/env python3
"""
Execute the primary lesson notebooks from a clean kernel and report failures.

Each notebook is run with a fresh IPython kernel (no leftover state from
previous cells or notebooks), catching import errors, undefined names, and
missing data files that a partially-run kernel can silently mask.
"""

import argparse
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent

PRIMARY_NOTEBOOKS = [
    "lessons/L01_Regression_Analysis/notebook/housing_regression_analysis.ipynb",
    "lessons/L02_Multivariate_Analysis/notebook/environmental_multivariate_analysis.ipynb",
    "lessons/L03_Principal_Component_Analysis/notebook/financial_pca_analysis.ipynb",
    "lessons/L04_Factor_Analysis/notebook/educational_analysis.ipynb",
    "lessons/L05_Discriminant_Analysis/notebook/marketing_discriminant_analysis.ipynb",
    "lessons/L06_Cluster_Analysis/notebook/customer_clustering_analysis.ipynb",
    "lessons/L07_Multivariate_Regression/notebook/health_risk_analysis.ipynb",
]


def run_one(rel_path: str, write_outputs: bool, timeout: int) -> bool:
    nb_path = REPO_ROOT / rel_path
    if not nb_path.exists():
        print(f"FAIL {rel_path} - FILE NOT FOUND")
        return False

    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
    except Exception as e:
        print(f"FAIL {rel_path}")
        print(f"   {type(e).__name__}: {e}")
        return False

    if write_outputs:
        nbformat.write(nb, nb_path)

    print(f"PASS {rel_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="Persist regenerated cell outputs back into the notebook files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell execution timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Subset of notebook paths (relative to the repo root) to run; defaults to all primary notebooks",
    )
    args = parser.parse_args()

    targets = args.notebooks or PRIMARY_NOTEBOOKS
    results = [run_one(t, args.write_outputs, args.timeout) for t in targets]

    passed = sum(results)
    print(f"\n{passed}/{len(results)} notebooks executed successfully")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
