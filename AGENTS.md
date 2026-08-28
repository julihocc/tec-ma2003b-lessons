# Repository Guidelines

## Project Structure & Module Organization

Course content lives under `lessons/`, organized as `L##_Topic_Name` (for example, `lessons/L03_Principal_Component_Analysis/`). Each lesson generally contains:

- `data/`: CSV datasets, data dictionaries, and `fetch_*.py` generators.
- `notebook/`: Jupyter case studies and, where applicable, reusable snippets.
- `notes/`: LaTeX lecture-note sources and compiled PDFs.
- `presentation/`: LaTeX Beamer slide sources and compiled PDFs.
- `README.md`: lesson objectives and usage notes.

Course-wide material is in `docs/`, including `SYLLABUS.md`. Python dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Shared LaTeX styles live in `latex/`: `ma2003b-common.sty` (fonts, Pandoc syntax-highlighting macros, table/hyperref plumbing shared by both classes), `ma2003b-notes.sty` (article-class notes styling), and `ma2003b-beamer.sty` (Beamer presentation styling, Tec de Monterrey Madrid theme/branding). Every lesson `.tex` source pulls these in via a single `\usepackage{ma2003b-notes}` or `\usepackage{ma2003b-beamer}` line instead of duplicating the preamble.

## Build, Test, and Development Commands

- `uv sync`: create `.venv` with the locked Python 3.11 environment.
- `uv run jupyter notebook`: launch notebooks with project dependencies.
- `uv run python lessons/L01_Regression_Analysis/data/fetch_housing_regression.py`: regenerate a lesson dataset; use the corresponding `fetch_*.py` for other lessons.
- `uv run python lessons/L04_Factor_Analysis/notebook/snippets/test_all_snippets.py`: execute all factor-analysis snippet notebooks non-interactively and check their key numerical results.
- `uv run python scripts/run_notebooks.py`: execute all seven primary lesson notebooks from a clean kernel; pass notebook paths to run a subset, or `--write-outputs` to persist regenerated outputs.
- `./compile.ps1 lessons/L01_Regression_Analysis/notes/regression_analysis_notes.tex` (PowerShell) or `./compile.sh lessons/L01_Regression_Analysis/notes/regression_analysis_notes.tex` (bash): rebuild a lecture-note or presentation PDF. Works from any working directory and applies to any lesson's notes/presentation source. Under the hood it points `TEXINPUTS` at the repo's `latex/` style directory and runs `latexmk -xelatex` (xelatex is required: some notes sources contain literal Unicode math symbols that plain `pdflatex` cannot typeset). To compile manually instead: set `TEXINPUTS` to `<repo-root>\latex//;` (note the trailing `;`, which re-appends the default TeX search path) and run `xelatex <file>.tex` twice.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions for Python. Prefer descriptive `snake_case` for functions, variables, scripts, and notebooks; use uppercase underscore-separated names for data dictionaries. Keep lesson directories aligned with `L##_Topic_Name`, and use relative paths so notebooks run from the repository root. No formatter or linter is configured, so keep imports organized and changes consistent with nearby files.

## Testing Guidelines

There is no coverage target or general test framework yet. Run the snippet harness after changing factor-analysis notebooks. For other notebooks, run `scripts/run_notebooks.py` (or restart the kernel and run all cells from top to bottom in Jupyter) to confirm outputs are reproducible and paths resolve. Recompile affected `.tex` files with `./compile.ps1`/`./compile.sh` and inspect the resulting PDF for layout problems. If you touch `latex/ma2003b-*.sty`, recompile at least one notes file and one presentation file, since both classes share it.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects: `feat(L03): ...`, `docs: ...`, `build: ...`, and `refactor(L06,L07): ...`. Use an imperative, concise summary and add lesson scopes when relevant. Pull requests should explain the educational or technical change, list affected lessons, and report validation commands. Include screenshots for visual slide or note changes and note regenerated datasets or PDFs. CSV, IPYNB, and PDF files are tracked with Git LFS; install LFS before committing or reviewing these assets, and avoid unrelated generated-file churn.
