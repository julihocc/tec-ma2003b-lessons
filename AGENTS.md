# Repository Guidelines

## Project Structure & Module Organization

Course content lives under `lessons/`, organized as `L##_Topic_Name` (for example, `lessons/L03_Principal_Component_Analysis/`). Each lesson generally contains:

- `data/`: CSV datasets, data dictionaries, and `fetch_*.py` generators.
- `notebook/`: Jupyter case studies and, where applicable, reusable snippets.
- `docs/`: Typst lecture-note sources and compiled PDFs.
- `presentation/`: Typst slide sources and compiled PDFs.
- `README.md`: lesson objectives and usage notes.

Course-wide material is in `docs/`, including `SYLLABUS.md`. Python dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

## Build, Test, and Development Commands

- `uv sync`: create `.venv` with the locked Python 3.11 environment.
- `uv run jupyter notebook`: launch notebooks with project dependencies.
- `uv run python lessons/L01_Regression_Analysis/data/fetch_housing_regression.py`: regenerate a lesson dataset; use the corresponding `fetch_*.py` for other lessons.
- `uv run python lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py`: execute all factor-analysis snippet notebooks non-interactively.
- `typst compile lessons/L01_Regression_Analysis/docs/regression_analysis_notes.typ`: rebuild a Typst PDF. Apply the same pattern to presentations and other lessons.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions for Python. Prefer descriptive `snake_case` for functions, variables, scripts, and notebooks; use uppercase underscore-separated names for data dictionaries. Keep lesson directories aligned with `L##_Topic_Name`, and use relative paths so notebooks run from the repository root. No formatter or linter is configured, so keep imports organized and changes consistent with nearby files.

## Testing Guidelines

There is no coverage target or general test framework yet. Run the snippet harness after changing factor-analysis notebooks. For other notebooks, restart the kernel and run all cells from top to bottom, checking that outputs are reproducible and paths resolve. Recompile affected `.typ` files and inspect the resulting PDF for layout problems.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects: `feat(L03): ...`, `docs: ...`, `build: ...`, and `refactor(L06,L07): ...`. Use an imperative, concise summary and add lesson scopes when relevant. Pull requests should explain the educational or technical change, list affected lessons, and report validation commands. Include screenshots for visual slide or note changes and note regenerated datasets or PDFs. CSV, IPYNB, and PDF files are tracked with Git LFS; install LFS before committing or reviewing these assets, and avoid unrelated generated-file churn.
