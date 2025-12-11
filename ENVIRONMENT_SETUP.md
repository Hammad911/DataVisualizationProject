# Conda Environment Setup Guide

## Environment Created: `dataviz`

### Quick Start

1. **Activate the environment:**
   ```bash
   conda activate dataviz
   ```

2. **Deactivate when done:**
   ```bash
   conda deactivate
   ```

### Using in Jupyter Notebook

The environment has been registered as a Jupyter kernel. To use it:

1. Open your Jupyter Notebook
2. Go to **Kernel** → **Change Kernel** → Select **"Python (DataViz)"**

Or if starting a new notebook, select **"Python (DataViz)"** as the kernel when creating it.

### Installed Packages

The following packages are installed in this environment:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization
- **pyarrow** - Apache Arrow (for Parquet file support)
- **jupyter** - Jupyter notebook
- **notebook** - Classic Jupyter notebook interface
- **ipykernel** - Jupyter kernel for Python

### Managing the Environment

**List all environments:**
```bash
conda env list
```

**Update packages:**
```bash
conda activate dataviz
conda update --all
```

**Remove the environment (if needed):**
```bash
conda env remove -n dataviz
```

**Recreate from requirements:**
```bash
conda env create -f requirements.txt
```

### Notes

- The environment uses Python 3.10
- All packages are installed from conda-forge for better compatibility
- The kernel is registered in your user Jupyter directory


















