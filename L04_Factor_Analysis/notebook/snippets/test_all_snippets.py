#!/usr/bin/env python3
"""
Test All Factor Analysis Snippets
Runs all snippet notebook code cells in non-interactive mode and reports their status
"""

import sys
import json
from pathlib import Path

# Ensure non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

# Compatibility patch for factor_analyzer with scikit-learn >= 1.6
import sklearn.utils.validation
_orig_check_array = sklearn.utils.validation.check_array
def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_array(*args, **kwargs)
sklearn.utils.validation.check_array = _patched_check_array
sklearn.utils.check_array = _patched_check_array

def test_notebook(nb_path):
    """Test a single Jupyter notebook by executing its code cells."""
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        ]

        if not code_cells:
            print(f"⚠️  {nb_path.name} - NO CODE CELLS FOUND")
            return True

        # Combine code cells and execute in isolated namespace
        setup_code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

import sklearn.utils.validation
_orig = sklearn.utils.validation.check_array
def _patched(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig(*args, **kwargs)
sklearn.utils.validation.check_array = _patched
sklearn.utils.check_array = _patched
"""
        full_code = setup_code + "\n\n# --- CELL ---\n\n".join(code_cells)
        exec_namespace = {"__name__": "__main__"}
        exec(compile(full_code, str(nb_path), "exec"), exec_namespace)

        print(f"✅ {nb_path.name} - SUCCESS")
        return True

    except Exception as e:
        print(f"❌ {nb_path.name} - FAILED")
        print(f"   Error: {e}")
        return False

def main():
    """Run all snippet tests."""
    print("Testing Factor Analysis Jupyter Notebook Snippets (Non-Interactive)")
    print("=" * 50)

    snippets_dir = Path(__file__).resolve().parent
    snippet_files = [
        snippets_dir / "01_pca_basic_example.ipynb",
        snippets_dir / "02_component_retention.ipynb",
        snippets_dir / "03_factor_analysis_basic.ipynb",
        snippets_dir / "04_factor_rotation.ipynb",
        snippets_dir / "05_complete_workflow.ipynb"
    ]

    success_count = 0
    total_count = len(snippet_files)

    for snippet in snippet_files:
        if snippet.exists():
            if test_notebook(snippet):
                success_count += 1
        else:
            print(f"❌ {snippet.name} - FILE NOT FOUND")

    print("\n" + "=" * 50)
    print(f"Test Results: {success_count}/{total_count} snippets passed")

    if success_count == total_count:
        print("🎉 All snippets are working correctly!")
        return 0
    else:
        print("⚠️  Some snippets have issues. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
