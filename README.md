# 🎬 IMDb Analytics Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.14+-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A modern, interactive dashboard for exploring and analyzing IMDb dataset with 1.6M+ titles**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Screenshots](#-dashboard-preview) • [Documentation](#-documentation)

</div>

---

## 📊 Overview

This project provides comprehensive data analysis and visualization of the IMDb dataset, featuring:
- **1.6 Million+ titles** spanning over 100 years of cinema history
- **21+ interactive visualizations** across 6 analytical categories
- **Modern, responsive dashboard** built with Plotly Dash
- **Deep insights** into ratings, genres, temporal trends, and more

## ✨ Features

### 🎨 Modern Dashboard Design
- **Gradient-based UI** with smooth animations
- **Responsive layout** that works on all screen sizes
- **Interactive visualizations** with hover tooltips and zoom
- **Professional color schemes** using modern design principles
- **Tab-based navigation** for easy exploration

### 📈 Analysis Categories

1. **Temporal Trends** - Historical patterns and evolution over time
   - Title production volume trends
   - Rating trends across decades
   - Runtime evolution analysis
   - Volume vs. quality by decade

2. **Rating Analysis** - Quality metrics and distributions
   - Rating distribution across all titles
   - Ratings by title type (movies, TV shows, etc.)
   - Decade-wise rating patterns
   - Correlation matrix of numeric variables

3. **Genre Analysis** - Genre patterns and evolution
   - Genre evolution across 100+ years
   - Top 15 primary genres
   - Rating distribution by genre
   - Multi-genre title analysis

4. **Runtime Analysis** - Optimal movie length insights
   - Runtime vs. Rating sweet spot analysis
   - Runtime distribution patterns
   - Runtime by title type (violin plots)
   - Average runtime by genre

5. **Popularity Analysis** - Audience engagement metrics
   - Vote distribution patterns
   - Rating vs. popularity correlation
   - Popularity bias analysis
   - High-engagement title identification

6. **Advanced Insights** - Deep-dive analytics
   - Historical era genre distribution
   - Cinema "Golden Ages" analysis
   - Quality vs. quantity metrics
   - Cross-decade comparisons

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- 4GB+ RAM (for processing large dataset)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "Semester 5/Data Visualization/Project"
```

### Step 2: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or install key packages manually
pip install pandas numpy plotly dash dash-bootstrap-components scipy
```

### Step 3: Prepare the Data

Place the following IMDb dataset files in the project directory:
- `title.basics.tsv`
- `title.ratings.tsv`
- `title.crew.tsv`
- `title.akas.tsv`
- `title.principals.tsv`
- `title.episode.tsv`

**Download from:** [IMDb Datasets](https://datasets.imdbws.com/)

### Step 4: Process the Data

Run the Jupyter notebook to clean and process the data:

```bash
jupyter notebook finalahad.ipynb
```

Or run the Python script:

```bash
python Project.py
```

This will generate cleaned data files in the `processed/` directory.

## 🎯 Usage

### Running the Dashboard

```bash
python dashboard_app.py
```

The dashboard will start at: **http://127.0.0.1:8050/**

Open your web browser and navigate to the URL to explore the interactive dashboard.

### Using the Jupyter Notebook

```bash
jupyter notebook finalahad.ipynb
```

The notebook contains:
- Data loading and preprocessing
- Feature engineering
- Exploratory data analysis
- All 18+ visualizations
- Statistical analysis

## 📁 Project Structure

```
Project/
├── dashboard_app.py           # Interactive Dash dashboard
├── finalahad.ipynb           # Main analysis notebook
├── Project.py                # Python script version
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── processed/                # Generated output files
│   ├── cleaned_imdb_data.csv
│   ├── cleaned_imdb_data.parquet
│   ├── cleaning_report.txt
│   └── *.png                 # Visualization exports
│
├── *.tsv                     # Raw IMDb dataset files
│
└── docs/
    ├── Project Statement.pdf
    ├── Impactful Visualizations.docx
    └── ProjectDoc.pdf
```

## 🛠️ Technologies Used

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **plotly** - Interactive visualizations
- **dash** - Web dashboard framework
- **dash-bootstrap-components** - Modern UI components

### Visualization & Analysis
- **matplotlib** - Static plotting
- **seaborn** - Statistical visualizations
- **scipy** - Scientific computing
- **statsmodels** - Statistical modeling

### Data Processing
- **pyarrow** - Parquet file support
- **openpyxl** - Excel file handling

## 📊 Dashboard Preview

### Main Dashboard
- **Modern gradient header** with key statistics
- **4 summary cards** showing:
  - Total titles analyzed (1.6M+)
  - Average rating (7.2)
  - Time span (1900-2024)
  - Unique genres (20+)

### Interactive Features
- **Tab navigation** for different analysis categories
- **Hover tooltips** with detailed information
- **Responsive charts** that adapt to screen size
- **Clean, minimalist design** with professional color palette

## 📈 Key Insights Discovered

1. **Optimal Movie Length**: 90-120 minute films achieve highest ratings
2. **Genre Evolution**: Drama and Comedy dominate across all eras
3. **Rating Trends**: Average ratings have remained stable at ~7.0 since 1950
4. **Popularity Bias**: Higher vote counts correlate with more polarized ratings
5. **Golden Age**: 1990s-2000s show peak in both quantity and quality
6. **TV Show Patterns**: Quality tends to decline after Season 8

## 🔧 Configuration

### Performance Optimization

For large datasets (1M+ rows), the dashboard automatically samples data for optimal performance:

```python
# In dashboard_app.py
if len(df) > 500000:
    df = df.sample(n=500000, random_state=42)
```

### Customization

Modify the color scheme in `dashboard_app.py`:

```python
COLORS = {
    'primary': '#6366f1',    # Change to your brand color
    'secondary': '#8b5cf6',
    'success': '#10b981',
    # ... more colors
}
```

## 📝 Data Processing Pipeline

1. **Data Loading** - Read TSV files with optimized dtypes
2. **Data Cleaning** - Handle missing values, duplicates, outliers
3. **Feature Engineering** - Create derived features (decade, genre_count, etc.)
4. **Data Merging** - Combine multiple datasets on `tconst`
5. **Data Export** - Save cleaned data as CSV and Parquet
6. **Visualization** - Generate 21+ interactive plots

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install dev dependencies
pip install jupyter ipykernel black flake8

# Run linting
flake8 dashboard_app.py

# Format code
black dashboard_app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IMDb** for providing the comprehensive dataset
- **Plotly** for the excellent visualization library
- **Dash** for the powerful dashboard framework
- **Bootstrap** for the modern UI components

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for Data Visualization**

⭐ Star this repo if you find it useful!

</div>

# DataVisualizationProject
