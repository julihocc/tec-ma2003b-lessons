# Factor Analysis Interactive Notebooks

This folder contains **interactive Jupyter notebooks** converted from the original Python code snippets. These notebooks provide hands-on, executable examples for learning factor analysis concepts.

## 📓 Available Notebooks

1. **01_pca_basic_example.ipynb** - Basic PCA implementation with simple data
2. **02_component_retention.ipynb** - Component retention methods (Kaiser criterion, scree plot)
3. **03_factor_analysis_basic.ipynb** - Basic Factor Analysis with correlation matrix
4. **04_factor_rotation.ipynb** - Factor rotation examples (Varimax, Promax)
5. **05_complete_workflow.ipynb** - Complete end-to-end factor analysis workflow

## 🚀 Getting Started

### Prerequisites

Install required packages:

```bash
pip install numpy pandas scikit-learn matplotlib factor-analyzer jupyter
```

### Running the Notebooks

#### Option 1: VS Code (Recommended)

1. Open any `.ipynb` file in VS Code
2. Click "Run All" or execute cells individually
3. Notebooks use py-percent format for dual Python/Jupyter compatibility

#### Option 2: Jupyter Lab/Notebook

```bash
jupyter lab
# or
jupyter notebook
```

Then navigate to the `snippets/` folder and open any notebook.

#### Option 3: Command Line

```bash
# Run all notebooks non-interactively
jupyter nbconvert --execute --to notebook --inplace *.ipynb
```

## 📊 Notebook Features

- **Interactive execution**: Run code cells to see live results
- **Rich visualizations**: Matplotlib plots for scree plots, factor loadings, etc.
- **Educational structure**: Markdown explanations between code sections
- **Self-contained**: Each notebook includes data generation and analysis
- **Progressive learning**: From basic concepts to complete workflows

## ✅ Validation

All notebooks have been tested and validated. See `TESTING_RESULTS.md` for detailed test results.

## 🔧 Development

### Converting Notebooks to Scripts

```bash
# Convert back to Python scripts with py-percent cells
jupyter nbconvert --to script *.ipynb
```

### Converting Scripts to Notebooks

```bash
# Convert py-percent Python files to notebooks
jupytext --to notebook *.py
```

## 📈 Learning Path

Follow this sequence for optimal learning:

1. **Start with PCA basics** (`01_pca_basic_example.ipynb`)
2. **Learn component selection** (`02_component_retention.ipynb`)
3. **Explore factor analysis** (`03_factor_analysis_basic.ipynb`)
4. **Understand rotation** (`04_factor_rotation.ipynb`)
5. **Apply complete workflow** (`05_complete_workflow.ipynb`)

Each notebook builds on concepts from the previous ones while remaining self-contained.
