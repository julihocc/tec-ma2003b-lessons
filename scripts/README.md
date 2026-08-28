# Developer Scripts & Automation (`scripts/`)

This directory contains automated testing harnesses, execution utilities, and quality assurance scripts for the **MA2003B** course repository.

---

## 📂 Directory Contents

| Script | Purpose | Execution Command |
| :--- | :--- | :--- |
| [`run_notebooks.py`](run_notebooks.py) | Non-interactively executes the 7 primary lesson Jupyter notebooks from clean kernels, verifying path resolution, library compatibility, and runtime reproducibility without kernel pollution. | `uv run python scripts/run_notebooks.py` |

---

## 🛠️ Script Documentation

### `run_notebooks.py`

#### Overview
`run_notebooks.py` uses `nbformat` and `nbconvert.preprocessors.ExecutePreprocessor` to execute notebooks sequentially in fresh IPython kernels. It catches missing packages, broken relative dataset paths, syntax errors, and outdated API calls.

#### CLI Arguments
- `--write-outputs`: Persists newly executed cell outputs back into the `.ipynb` files.
- `--timeout <seconds>`: Maximum execution time per code cell (default: `600` seconds).
- `[notebooks ...]`: Optional positional paths to run a specific subset of notebooks.

#### Examples

```bash
# Execute all 7 primary notebooks
uv run python scripts/run_notebooks.py

# Execute a specific lesson notebook
uv run python scripts/run_notebooks.py lessons/L06_Cluster_Analysis/notebook/customer_clustering_analysis.ipynb

# Re-run all notebooks and save updated outputs back to files
uv run python scripts/run_notebooks.py --write-outputs
```

---

## 🧪 Additional Lesson-Specific Test Suites

For factor analysis modular snippets, execute the dedicated numerical verification suite:

```bash
uv run python lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py
```
