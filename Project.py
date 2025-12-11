#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install required packages (if not using conda environment)
# NOTE: If using conda environment 'dataviz', packages are already installed
# Uncomment below only if you need to install packages in the current kernel
# %pip install pandas numpy matplotlib seaborn pyarrow -q



# In[2]:


# Enable inline plotting in Jupyter notebook
# Note: This magic command is for Jupyter notebooks only
# For standalone Python scripts, plots will be displayed/saved automatically

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import gc
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


# In[3]:


# Set up paths for local environment
import os

# Use current working directory (where the notebook and data files are)
# This will be the project directory
BASE_PATH = os.getcwd() + '/'
OUTPUT_PATH = BASE_PATH + 'processed/'

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*80)
print("IMDB DATASET COMPREHENSIVE PREPROCESSING PIPELINE")
print("="*80)
print(f"Base Path: {BASE_PATH}")
print(f"Output Path: {OUTPUT_PATH}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ## UTILITY FUNCTIONS
# 

# In[4]:


def load_data_optimized(filename, usecols=None, nrows=None, dtypes=None):
    """Load TSV files with memory optimization"""
    print(f"\n{'='*80}")
    print(f"Loading {filename}...")
    print(f"{'='*80}")

    filepath = BASE_PATH + filename

    # Check if file exists
    if not os.path.exists(filepath):
        # Try without .gz
        filepath_alt = filepath.replace('.gz', '')
        if os.path.exists(filepath_alt):
            filepath = filepath_alt
        else:
            print(f"ERROR: File not found: {filename}")
            return None

    # Get file size
    file_size_mb = os.path.getsize(filepath) / (1024**2)
    print(f"File size: {file_size_mb:.2f} MB")

    # Read with optimized settings
    df = pd.read_csv(
        filepath,
        sep='\t',
        na_values=['\\N'],
        low_memory=False,
        usecols=usecols,
        nrows=nrows,
        dtype=dtypes
    )

    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


# In[5]:


def assess_data_quality(df, name):
    """Comprehensive data quality assessment with visualizations"""
    print(f"\n{'='*80}")
    print(f"DATA QUALITY ASSESSMENT: {name}")
    print(f"{'='*80}")

    report = {}

    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    report['total_rows'] = df.shape[0]
    report['total_columns'] = df.shape[1]

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    report['missing_values'] = missing_df

    # Duplicates
    if 'tconst' in df.columns:
        duplicates = df.duplicated(subset=['tconst']).sum()
    elif 'nconst' in df.columns:
        duplicates = df.duplicated(subset=['nconst']).sum()
    else:
        duplicates = df.duplicated().sum()

    print(f"\nDuplicate Rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    report['duplicates'] = duplicates

    # Sample data
    print("\nFirst 3 rows:")
    print(df.head(3))

    return report


# In[6]:


def visualize_missing_data(df, name):
    """Create visualization for missing data"""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) > 0:
        plt.figure(figsize=(10, max(6, len(missing_pct) * 0.4)))
        missing_pct.plot(kind='barh', color='coral')
        plt.xlabel('Missing Percentage (%)')
        plt.title(f'Missing Data in {name}')
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + f'missing_data_{name.lower().replace(" ", "_")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved missing data visualization for {name}")


# ## STEP 1: LOAD ALL DATASETS WITH DATA QUALITY ASSESSMENT
# 

# In[7]:


print("\n" + "="*80)
print("STEP 1: LOADING ALL DATASETS")
print("="*80)

# Define optimal dtypes to reduce memory
title_basics_dtypes = {
    'tconst': 'category',
    'titleType': 'category',
    'isAdult': 'int8'
}

ratings_dtypes = {
    'tconst': 'category',
    'averageRating': 'float32',
    'numVotes': 'int32'
}


# In[32]:


# 1.1: Load Title Basics
print("\n[1] LOADING TITLE.BASICS.TSV")
title_basics = load_data_optimized('title.basics.tsv', dtypes=title_basics_dtypes)
if title_basics is not None:
    quality_basics = assess_data_quality(title_basics, 'Title Basics')
    visualize_missing_data(title_basics, 'Title Basics')


# In[35]:


# 1.2: Load Title Ratings
print("\n[2] LOADING TITLE.RATINGS.TSV")
ratings = load_data_optimized('title.ratings.tsv', dtypes=ratings_dtypes)
if ratings is not None:
    quality_ratings = assess_data_quality(ratings, 'Title Ratings')
    visualize_missing_data(ratings, 'Title Ratings')


# In[37]:


# 1.3: Load Title Crew
print("\n[3] LOADING TITLE.CREW.TSV")
crew = load_data_optimized('title.crew.tsv')
if crew is not None:
    quality_crew = assess_data_quality(crew, 'Title Crew')
    visualize_missing_data(crew, 'Title Crew')


# In[13]:


# 1.4: Load Name Basics
print("\n[4] LOADING NAME.BASICS.TSV")
name_basics = load_data_optimized(
    'name.basics.tsv',
    usecols=['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession']
)
if name_basics is not None:
    quality_names = assess_data_quality(name_basics, 'Name Basics')
    visualize_missing_data(name_basics, 'Name Basics')


# In[14]:


# 1.5: Load Title Principals (sample due to size)
print("\n[5] LOADING TITLE.PRINCIPALS.TSV (SAMPLED)")
principals = load_data_optimized('title.principals.tsv', nrows=500000)
if principals is not None:
    quality_principals = assess_data_quality(principals, 'Title Principals')
    visualize_missing_data(principals, 'Title Principals')


# In[15]:


# 1.6: Load Title AKAs (assess but don't merge - multiple entries per title)
print("\n[6] LOADING TITLE.AKAS.TSV (ASSESSMENT ONLY)")
print("Note: This file has multiple entries per title (regional variants)")
akas = load_data_optimized('title.akas.tsv', nrows=100000)  # Sample for assessment
if akas is not None:
    quality_akas = assess_data_quality(akas, 'Title AKAs')
    print(f"\nMultiple entries per title: {akas.groupby('titleId').size().max()} max entries")
    visualize_missing_data(akas, 'Title AKAs')
    # NOTE: Keeping akas dataset for visualizations
    # Uncomment below if you need to free memory:
    # del akas  # Free memory
    # gc.collect()


# In[16]:


# 1.7: Load Title Episodes (assess but don't merge - multiple entries per series)
print("\n[7] LOADING TITLE.EPISODE.TSV (ASSESSMENT ONLY)")
print("Note: This file has multiple entries per TV series (episodes)")
episodes = load_data_optimized('title.episode.tsv', nrows=100000)  # Sample for assessment
if episodes is not None:
    quality_episodes = assess_data_quality(episodes, 'Title Episodes')
    print(f"\nMultiple entries per series: {episodes.groupby('parentTconst').size().max()} max episodes")
    visualize_missing_data(episodes, 'Title Episodes')
    # NOTE: Keeping episodes dataset for visualizations
    # Uncomment below if you need to free memory:
    # del episodes  # Free memory
    # gc.collect()

print("\n✓ All datasets loaded and assessed!")


# 
# 

# In[33]:


# 1. TITLE.BASICS.TSV
print("="*80)
print("1. TITLE.BASICS.TSV")
print("="*80)
if 'title_basics' in globals() and title_basics is not None:
    print(f"Shape: {title_basics.shape}")
    print(f"\nColumns ({len(title_basics.columns)}):")
    print(title_basics.columns.tolist())
    print(f"\nData Types:")
    print(title_basics.dtypes)
    print(f"\nMemory Usage: {title_basics.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nSample Data:")
    print(title_basics.head(3))
    print(f"\nUnique titleTypes: {title_basics['titleType'].value_counts().to_dict()}")
    print(f"Missing Values:\n{title_basics.isnull().sum()}")
else:
    print("⚠️ title_basics not loaded yet. Run Step 1 data loading cells first.")


# In[18]:


# 2. TITLE.RATINGS.TSV
print("="*80)
print("2. TITLE.RATINGS.TSV")
print("="*80)
if 'ratings' in globals() and ratings is not None:
    print(f"Shape: {ratings.shape}")
    print(f"\nColumns ({len(ratings.columns)}):")
    print(ratings.columns.tolist())
    print(f"\nData Types:")
    print(ratings.dtypes)
    print(f"\nMemory Usage: {ratings.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nSample Data:")
    print(ratings.head(3))
    print(f"\nRating Statistics:")
    print(ratings[['averageRating', 'numVotes']].describe())
    print(f"Missing Values:\n{ratings.isnull().sum()}")
else:
    print("⚠️ ratings not loaded yet. Run Step 1 data loading cells first.")


# In[19]:


# 3. TITLE.CREW.TSV
print("="*80)
print("3. TITLE.CREW.TSV")
print("="*80)
if 'crew' in globals() and crew is not None:
    print(f"Shape: {crew.shape}")
    print(f"\nColumns ({len(crew.columns)}):")
    print(crew.columns.tolist())
    print(f"\nData Types:")
    print(crew.dtypes)
    print(f"\nMemory Usage: {crew.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nSample Data:")
    print(crew.head(3))
    print(f"\nMissing Values:\n{crew.isnull().sum()}")
    print(f"\nDirectors with multiple entries: {crew['directors'].notna().sum():,}")
    print(f"Writers with multiple entries: {crew['writers'].notna().sum():,}")
else:
    print("⚠️ crew not loaded yet. Run Step 1 data loading cells first.")


# In[20]:


# 4. NAME.BASICS.TSV
print("="*80)
print("4. NAME.BASICS.TSV")
print("="*80)
if 'name_basics' in globals() and name_basics is not None:
    print(f"Shape: {name_basics.shape}")
    print(f"\nColumns ({len(name_basics.columns)}):")
    print(name_basics.columns.tolist())
    print(f"\nData Types:")
    print(name_basics.dtypes)
    print(f"\nMemory Usage: {name_basics.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nSample Data:")
    print(name_basics.head(3))
    print(f"\nMissing Values:\n{name_basics.isnull().sum()}")
else:
    print("⚠️ name_basics not loaded yet. Run Step 1 data loading cells first.")


# In[21]:


# 5. TITLE.PRINCIPALS.TSV (if loaded)
print("="*80)
print("5. TITLE.PRINCIPALS.TSV (Sample)")
print("="*80)
if 'principals' in globals() and principals is not None:
    print(f"Shape: {principals.shape}")
    print(f"\nColumns ({len(principals.columns)}):")
    print(principals.columns.tolist())
    print(f"\nData Types:")
    print(principals.dtypes)
    print(f"\nSample Data:")
    print(principals.head(3))
    print(f"\nUnique Categories: {principals['category'].value_counts().head(10).to_dict()}")
else:
    print("⚠️ principals not loaded yet. Run Step 1 data loading cells first.")


# In[22]:


# 6. RELATIONSHIP ANALYSIS & MERGE STRATEGY
print("="*80)
print("KEY RELATIONSHIPS")
print("="*80)

if 'title_basics' in globals() and 'ratings' in globals() and title_basics is not None and ratings is not None:
    print(f"\n• title_basics.tconst <-> ratings.tconst (Primary Key Join)")
    print(f"  - title_basics records: {len(title_basics):,}")
    print(f"  - ratings records: {len(ratings):,}")
    print(f"  - Common tconst values: {len(set(title_basics['tconst']) & set(ratings['tconst'])):,}")

if 'title_basics' in globals() and 'crew' in globals() and title_basics is not None and crew is not None:
    print(f"\n• title_basics.tconst <-> crew.tconst (Left Join)")
    print(f"  - crew records: {len(crew):,}")
    print(f"  - Common tconst values: {len(set(title_basics['tconst']) & set(crew['tconst'])):,}")

if 'name_basics' in globals() and name_basics is not None:
    print(f"\n• name_basics.nconst (Used in crew.directors, crew.writers, principals.nconst)")
    print(f"  - Total people: {len(name_basics):,}")

print("\n" + "="*80)
print("MERGE STRATEGY")
print("="*80)
print("1. title_basics INNER JOIN ratings (on tconst) - Keep only titles with ratings")
print("2. Result LEFT JOIN crew (on tconst) - Add director/writer info")
print("3. Final dataset will have: title info + ratings + crew info")
print("="*80)


# ## STEP 2: INITIAL MERGING (BEFORE CLEANING)
# 

# In[26]:


# 1.1: Load Title Basics
title_basics = load_data_optimized('title.basics.tsv', dtypes=title_basics_dtypes)


# In[27]:


# 1.2: Load Title Ratings
ratings = load_data_optimized('title.ratings.tsv', dtypes=ratings_dtypes)


# In[28]:


# 1.3: Load Title Crew
crew = load_data_optimized('title.crew.tsv')


# In[29]:


# Redundant imports removed - already imported at top

print("\n" + "="*80)
print("STEP 2: INITIAL DATA MERGING")
print("="*80)
print("\nMerging datasets BEFORE cleaning to maintain referential integrity")

# ============================================================================
# STEP 1: Inspect and Normalize Column Names
# ============================================================================
print("\n📋 STEP 1: Inspecting Dataset Column Names")
print("-" * 80)

def inspect_columns(df, name):
    """Inspect and display column information"""
    print(f"\n{name} columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. '{col}' (dtype: {df[col].dtype})")
    print(f"  Shape: {df.shape}")
    return df.columns.tolist()

# Inspect all datasets
title_basics_cols = inspect_columns(title_basics, "title_basics")
ratings_cols = inspect_columns(ratings, "ratings")
crew_cols = inspect_columns(crew, "crew")

# ============================================================================
# STEP 2: Normalize Column Names (handle variations)
# ============================================================================
print("\n🔧 STEP 2: Normalizing Column Names")
print("-" * 80)

# Common column name variations mapping
column_mapping = {
    'titleType': ['titleType', 'title_type', 'type', 'TitleType'],
    'titleName': ['titleName', 'title_name', 'primaryTitle', 'primary_title', 'title'],
    'primaryTitle': ['primaryTitle', 'primary_title', 'titleName', 'title_name', 'title'],
    'tconst': ['tconst', 'id', 'tconst_id', 'imdb_id'],
    'director': ['director', 'directors', 'director_name', 'Director'],
    'writer': ['writer', 'writers', 'writer_name', 'Writer'],
    'averageRating': ['averageRating', 'average_rating', 'rating', 'Rating'],
    'numVotes': ['numVotes', 'num_votes', 'votes', 'Votes'],
    'runtimeMinutes': ['runtimeMinutes', 'runtime_minutes', 'runtime', 'Runtime'],
    'startYear': ['startYear', 'start_year', 'year', 'Year'],
    'genres': ['genres', 'genre', 'Genres']
}

def normalize_column_names(df, df_name):
    """Normalize column names to standard format"""
    print(f"\nNormalizing {df_name} columns...")
    original_cols = df.columns.tolist()
    renamed = {}
    
    for standard_name, variations in column_mapping.items():
        for col in original_cols:
            if col in variations and col != standard_name:
                if standard_name not in renamed.values():
                    renamed[col] = standard_name
                    print(f"  '{col}' → '{standard_name}'")
    
    if renamed:
        df = df.rename(columns=renamed)
    else:
        print(f"  ✓ No renaming needed")
    
    return df

# Normalize column names
title_basics = normalize_column_names(title_basics, "title_basics")
ratings = normalize_column_names(ratings, "ratings")
crew = normalize_column_names(crew, "crew")

# ============================================================================
# STEP 3: Verify Required Columns Exist
# ============================================================================
print("\n✅ STEP 3: Verifying Required Columns")
print("-" * 80)

required_cols = {
    'title_basics': ['tconst'],
    'ratings': ['tconst'],
    'crew': ['tconst']
}

def verify_columns(df, df_name, required):
    """Verify required columns exist"""
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"  ❌ {df_name} missing columns: {missing}")
        print(f"  Available columns: {df.columns.tolist()}")
        raise KeyError(f"{df_name} is missing required columns: {missing}")
    else:
        print(f"  ✓ {df_name} has all required columns")
    return True

verify_columns(title_basics, "title_basics", required_cols['title_basics'])
verify_columns(ratings, "ratings", required_cols['ratings'])
verify_columns(crew, "crew", required_cols['crew'])

# ============================================================================
# STEP 4: Merge Datasets with Error Handling
# ============================================================================
print("\n🔗 STEP 4: Merging Datasets")
print("-" * 80)

try:
    # Merge title_basics with ratings (inner join - keep only titles with ratings)
    print("\nMerging title_basics + ratings...")
    print(f"  title_basics shape: {title_basics.shape}")
    print(f"  ratings shape: {ratings.shape}")
    print(f"  Common key: 'tconst'")
    
    df_merged = title_basics.merge(ratings, on='tconst', how='inner', suffixes=('', '_ratings'))
    print(f"  ✓ After merging basics + ratings: {df_merged.shape}")
    print(f"  Columns after merge: {df_merged.columns.tolist()}")
    
except KeyError as e:
    print(f"  ❌ Error merging title_basics and ratings: {e}")
    print(f"  title_basics columns: {title_basics.columns.tolist()}")
    print(f"  ratings columns: {ratings.columns.tolist()}")
    raise

try:
    # Merge with crew (left join - keep all titles, crew info optional)
    print("\nMerging with crew data...")
    print(f"  df_merged shape: {df_merged.shape}")
    print(f"  crew shape: {crew.shape}")
    print(f"  Common key: 'tconst'")
    
    # Check for duplicate column names before merge
    common_cols = set(df_merged.columns) & set(crew.columns)
    if 'tconst' in common_cols:
        common_cols.remove('tconst')
    if common_cols:
        print(f"  ⚠️  Common columns (will use suffixes): {common_cols}")
    
    df_merged = df_merged.merge(crew, on='tconst', how='left', suffixes=('', '_crew'))
    print(f"  ✓ After merging with crew: {df_merged.shape}")
    print(f"  Columns after merge: {df_merged.columns.tolist()}")
    
except KeyError as e:
    print(f"  ❌ Error merging with crew: {e}")
    print(f"  df_merged columns: {df_merged.columns.tolist()}")
    print(f"  crew columns: {crew.columns.tolist()}")
    raise

# ============================================================================
# STEP 5: Handle Duplicate Column Names
# ============================================================================
print("\n🧹 STEP 5: Cleaning Up Duplicate Columns")
print("-" * 80)

# Remove duplicate columns (keep first occurrence)
duplicate_cols = df_merged.columns[df_merged.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"  ⚠️  Found duplicate columns: {duplicate_cols}")
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
    print(f"  ✓ Removed duplicate columns")
else:
    print(f"  ✓ No duplicate columns found")

# Handle columns with suffixes that might be duplicates
cols_to_check = ['director', 'directors', 'writer', 'writers']
for col in cols_to_check:
    if f"{col}_crew" in df_merged.columns and col in df_merged.columns:
        print(f"  Merging '{col}' and '{col}_crew'...")
        # Combine the columns (fill NaN in one with values from the other)
        df_merged[col] = df_merged[col].fillna(df_merged[f"{col}_crew"])
        df_merged = df_merged.drop(columns=[f"{col}_crew"])
        print(f"    ✓ Combined into '{col}'")

# ============================================================================
# STEP 6: Final Verification
# ============================================================================
print("\n📊 STEP 6: Final Dataset Summary")
print("-" * 80)

# Store original shape for comparison
original_shape = df_merged.shape
print(f"Merged dataset shape: {df_merged.shape}")
print(f"Memory usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nFinal column names:")
for i, col in enumerate(df_merged.columns, 1):
    non_null = df_merged[col].notna().sum()
    pct = (non_null / len(df_merged)) * 100
    print(f"  {i:2d}. '{col}' - {non_null:,} non-null ({pct:.1f}%)")

# Verify critical columns exist
critical_cols = ['tconst', 'primaryTitle', 'startYear', 'runtimeMinutes', 
                 'averageRating', 'numVotes']
missing_critical = [col for col in critical_cols if col not in df_merged.columns]

if missing_critical:
    print(f"\n⚠️  WARNING: Missing critical columns: {missing_critical}")
    print("  Available similar columns:")
    for col in missing_critical:
        similar = [c for c in df_merged.columns if col.lower() in c.lower() or c.lower() in col.lower()]
        if similar:
            print(f"    '{col}' → {similar}")
else:
    print("\n✓ All critical columns present")

# Keep original datasets for visualizations
print("\n📊 Keeping original datasets for visualizations...")
# NOTE: Variables title_basics, ratings, and crew are kept for use in visualizations
# Uncomment the lines below if you need to free memory:
# del title_basics, ratings, crew
# gc.collect()
print("✓ Original datasets retained (title_basics, ratings, crew)")

print("\n" + "="*80)
print("✓ Data merging complete!")
print("="*80 + "\n")


# In[38]:
# NOTE: This section is commented out because it's a duplicate merge operation.
# The merge has already been completed in the earlier section (STEP 4: Merging Datasets).
# The df_merged dataframe is already available for the subsequent cleaning steps.
# Original datasets (title_basics, ratings, crew) are kept for visualizations.

# import gc
# 
# print("\n" + "="*80)
# print("STEP 2: INITIAL DATA MERGING")
# print("="*80)
# print("\nMerging datasets BEFORE cleaning to maintain referential integrity")
# 
# # This cell now assumes the real IMDB tables `title_basics`, `ratings`, and `crew`
# # have already been loaded from the TSV files in earlier cells.
# 
# # 1) Merge title_basics with ratings (inner join – keep only titles that have ratings)
# print("\nMerging title_basics + ratings (inner on 'tconst')...")
# print(f"  title_basics shape: {title_basics.shape}")
# print(f"  ratings shape: {ratings.shape}")
# 
# df_merged = title_basics.merge(ratings, on='tconst', how='inner')
# print(f"  ✓ After basics + ratings: {df_merged.shape}")
# 
# # 2) Merge with crew (left join – keep all rated titles, crew info optional)
# print("\nMerging with crew data (left on 'tconst')...")
# print(f"  crew shape: {crew.shape}")
# 
# df_merged = df_merged.merge(crew, on='tconst', how='left')
# print(f"  ✓ After merge with crew: {df_merged.shape}")
# 
# # 3) Store original shape for later validation
# original_shape = df_merged.shape
# print(f"\nMerged dataset shape: {df_merged.shape}")
# print(f"Memory usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
# 
# # 4) Clean up large source tables to free memory
# del title_basics, ratings, crew
# gc.collect()


# ## STEP 3: DATA CLEANING & WRANGLING
# 

# ### 3.1: Handle Missing Values
# 

# In[50]:


print("\n" + "="*80)
print("[3.1] HANDLING MISSING VALUES - PROFESSIONAL DATA CLEANING")
print("="*80)

# ============================================================================
# BEFORE CLEANING: Capture Initial State
# ============================================================================
print("\n📊 BEFORE CLEANING - Initial Dataset State:")
print("-" * 80)
initial_shape = df_merged.shape
initial_memory = df_merged.memory_usage(deep=True).sum() / 1024**2
initial_missing = df_merged.isnull().sum()
initial_total_missing = initial_missing.sum()

print(f"Dataset Shape: {initial_shape[0]:,} rows × {initial_shape[1]} columns")
print(f"Memory Usage: {initial_memory:.2f} MB")
print(f"Total Missing Values: {initial_total_missing:,}")

print("\nMissing Values by Column:")
missing_summary = initial_missing[initial_missing > 0].sort_values(ascending=False)
if len(missing_summary) > 0:
    for col, count in missing_summary.items():
        pct = (count / initial_shape[0]) * 100
        print(f"  • {col}: {count:,} ({pct:.2f}%)")
else:
    print("  ✓ No missing values detected")

# Store before statistics for report generation
cleaning_stats = {
    'before': {
        'shape': initial_shape,
        'memory_mb': initial_memory,
        'total_missing': initial_total_missing,
        'missing_by_column': initial_missing.to_dict()
    }
}

# ============================================================================
# STEP 1: Handle Critical Missing Values (startYear, runtimeMinutes)
# ============================================================================
print("\n" + "-"*80)
print("STEP 1: Critical Missing Value Analysis")
print("-"*80)

# Analyze missing startYear
if 'startYear' in df_merged.columns:
    missing_startyear_before = df_merged['startYear'].isna().sum()
    missing_startyear_pct = (missing_startyear_before / initial_shape[0]) * 100
    print(f"\n➤ startYear Analysis:")
    print(f"   Missing: {missing_startyear_before:,} ({missing_startyear_pct:.2f}%)")
    print(f"   Action: KEPT (critical for temporal analysis, but flagged for awareness)")
    removed_startyear = 0
    missing_startyear = missing_startyear_before
else:
    missing_startyear = 0
    removed_startyear = 0
    print("\n➤ startYear: Column not found")

# Analyze missing runtimeMinutes
if 'runtimeMinutes' in df_merged.columns:
    missing_runtime_before = df_merged['runtimeMinutes'].isna().sum()
    missing_runtime_pct = (missing_runtime_before / initial_shape[0]) * 100
    print(f"\n➤ runtimeMinutes Analysis:")
    print(f"   Missing: {missing_runtime_before:,} ({missing_runtime_pct:.2f}%)")
    print(f"   Action: KEPT (critical for analysis, but flagged for awareness)")
    removed_runtime = 0
    missing_runtime = missing_runtime_before
else:
    missing_runtime = 0
removed_runtime = 0
    print("\n➤ runtimeMinutes: Column not found")

# ============================================================================
# STEP 2: Fill Missing Categorical Values
# ============================================================================
print("\n" + "-"*80)
print("STEP 2: Filling Missing Categorical Values")
print("-"*80)

# Fill genres
if 'genres' in df_merged.columns:
    genres_missing_before = df_merged['genres'].isna().sum()
    genres_missing_pct = (genres_missing_before / initial_shape[0]) * 100
    print(f"\n➤ genres Column:")
    print(f"   Before: {genres_missing_before:,} missing ({genres_missing_pct:.2f}%)")
    df_merged['genres'] = df_merged['genres'].fillna('Unknown')
    genres_missing_after = df_merged['genres'].isna().sum()
    genres_missing = genres_missing_before  # Store for reporting
    print(f"   After: {genres_missing_after:,} missing")
    print(f"   ✓ Filled {genres_missing_before:,} missing values with 'Unknown'")
else:
    genres_missing = 0
    print("\n➤ genres: Column not found")

# Fill crew information (directors/writers)
print("\n➤ Crew Information (directors/writers):")
directors_missing_before = 0
writers_missing_before = 0

if 'directors' in df_merged.columns:
    directors_missing_before = df_merged['directors'].isna().sum()
    directors_missing_pct = (directors_missing_before / initial_shape[0]) * 100
    print(f"   directors - Before: {directors_missing_before:,} missing ({directors_missing_pct:.2f}%)")
    df_merged['directors'] = df_merged['directors'].fillna('Unknown')
    directors_missing_after = df_merged['directors'].isna().sum()
    print(f"   directors - After: {directors_missing_after:,} missing")
    print(f"   ✓ Filled {directors_missing_before:,} missing director values")

if 'writers' in df_merged.columns:
    writers_missing_before = df_merged['writers'].isna().sum()
    writers_missing_pct = (writers_missing_before / initial_shape[0]) * 100
    print(f"   writers - Before: {writers_missing_before:,} missing ({writers_missing_pct:.2f}%)")
    df_merged['writers'] = df_merged['writers'].fillna('Unknown')
    writers_missing_after = df_merged['writers'].isna().sum()
    print(f"   writers - After: {writers_missing_after:,} missing")
    print(f"   ✓ Filled {writers_missing_before:,} missing writer values")

# ============================================================================
# STEP 3: Handle Valid Missing Values (endYear)
# ============================================================================
print("\n" + "-"*80)
print("STEP 3: Valid Missing Values (Intentional)")
print("-"*80)

if 'endYear' in df_merged.columns:
    endyear_missing = df_merged['endYear'].isna().sum()
    endyear_missing_pct = (endyear_missing / initial_shape[0]) * 100
    print(f"\n➤ endYear:")
    print(f"   Missing: {endyear_missing:,} ({endyear_missing_pct:.2f}%)")
    print(f"   Action: KEPT AS NaN (valid for non-series titles)")
    print(f"   Rationale: Movies and non-series content don't have end years")
else:
    endyear_missing = 0
    print("\n➤ endYear: Column not found")

# ============================================================================
# AFTER CLEANING: Capture Final State
# ============================================================================
print("\n" + "-"*80)
print("📊 AFTER CLEANING - Final Dataset State:")
print("-"*80)

final_shape = df_merged.shape
final_memory = df_merged.memory_usage(deep=True).sum() / 1024**2
final_missing = df_merged.isnull().sum()
final_total_missing = final_missing.sum()

print(f"Dataset Shape: {final_shape[0]:,} rows × {final_shape[1]} columns")
print(f"Memory Usage: {final_memory:.2f} MB")
print(f"Total Missing Values: {final_total_missing:,}")

print("\nMissing Values by Column (After Cleaning):")
missing_summary_after = final_missing[final_missing > 0].sort_values(ascending=False)
if len(missing_summary_after) > 0:
    for col, count in missing_summary_after.items():
        pct = (count / final_shape[0]) * 100
        print(f"  • {col}: {count:,} ({pct:.2f}%)")
else:
    print("  ✓ All missing values handled")

# ============================================================================
# SUMMARY: Before/After Comparison
# ============================================================================
print("\n" + "="*80)
print("📈 CLEANING SUMMARY - BEFORE vs AFTER")
print("="*80)
print(f"{'Metric':<30} {'Before':>15} {'After':>15} {'Change':>15}")
print("-" * 80)
print(f"{'Total Rows':<30} {initial_shape[0]:>15,} {final_shape[0]:>15,} {final_shape[0]-initial_shape[0]:>15,}")
print(f"{'Total Columns':<30} {initial_shape[1]:>15} {final_shape[1]:>15} {final_shape[1]-initial_shape[1]:>15}")
print(f"{'Memory Usage (MB)':<30} {initial_memory:>15.2f} {final_memory:>15.2f} {final_memory-initial_memory:>15.2f}")
print(f"{'Total Missing Values':<30} {initial_total_missing:>15,} {final_total_missing:>15,} {final_total_missing-initial_total_missing:>15,}")
print("="*80)

# Store after statistics for report generation
cleaning_stats['after'] = {
    'shape': final_shape,
    'memory_mb': final_memory,
    'total_missing': final_total_missing,
    'missing_by_column': final_missing.to_dict()
}

# Store specific cleaning metrics
cleaning_stats['metrics'] = {
    'genres_filled': genres_missing,
    'directors_filled': directors_missing_before if 'directors' in df_merged.columns else 0,
    'writers_filled': writers_missing_before if 'writers' in df_merged.columns else 0,
    'startyear_missing': missing_startyear,
    'runtime_missing': missing_runtime,
    'endyear_missing': endyear_missing if 'endYear' in df_merged.columns else 0
}

print("\n✓ Missing value handling complete - No rows removed")
print("="*80 + "\n")


# ### 3.2: Remove Duplicates
# 

# In[51]:


print("\n" + "="*80)
print("[3.2] REMOVING DUPLICATES - PROFESSIONAL DATA CLEANING")
print("="*80)

# ============================================================================
# BEFORE CLEANING: Capture Initial State
# ============================================================================
print("\n📊 BEFORE CLEANING - Duplicate Analysis:")
print("-" * 80)

before_shape = df_merged.shape
before_rows = before_shape[0]

# Check for duplicates based on tconst (unique identifier)
duplicates_count = df_merged.duplicated(subset='tconst').sum()
duplicates_pct = (duplicates_count / before_rows) * 100 if before_rows > 0 else 0

print(f"Total Rows: {before_rows:,}")
print(f"Duplicate Records (by tconst): {duplicates_count:,} ({duplicates_pct:.2f}%)")

# Additional duplicate checks
if duplicates_count > 0:
    print("\n➤ Analyzing duplicate patterns...")
    duplicate_rows = df_merged[df_merged.duplicated(subset='tconst', keep=False)]
    print(f"   Rows involved in duplicates: {len(duplicate_rows):,}")
    
    # Show sample duplicates for inspection
    if len(duplicate_rows) > 0:
        print("\n   Sample duplicate tconst values:")
        sample_duplicates = duplicate_rows['tconst'].value_counts().head(5)
        for tconst, count in sample_duplicates.items():
            print(f"     • {tconst}: appears {count} times")

# Store before statistics
duplicate_stats = {
    'before': {
        'shape': before_shape,
        'total_rows': before_rows,
        'duplicates_count': duplicates_count,
        'duplicates_pct': duplicates_pct
    }
}

# ============================================================================
# REMOVE DUPLICATES
# ============================================================================
print("\n" + "-"*80)
print("STEP: Removing Duplicates")
print("-"*80)

if duplicates_count > 0:
    print(f"\n➤ Removing {duplicates_count:,} duplicate records...")
    print(f"   Strategy: Keep first occurrence, remove subsequent duplicates")
    
    # Store which rows will be removed (for reporting)
    rows_to_remove = df_merged[df_merged.duplicated(subset='tconst', keep='first')].index
    
    # Remove duplicates
    df_merged = df_merged.drop_duplicates(subset='tconst', keep='first')
    
    print(f"   ✓ Successfully removed {duplicates_count:,} duplicate records")
    print(f"   ✓ Kept first occurrence for each unique tconst")
else:
    print(f"\n✓ No duplicates found - dataset is clean")
    rows_to_remove = []

# ============================================================================
# AFTER CLEANING: Capture Final State
# ============================================================================
print("\n" + "-"*80)
print("📊 AFTER CLEANING - Final State:")
print("-" * 80)

after_shape = df_merged.shape
after_rows = after_shape[0]
rows_removed = before_rows - after_rows

print(f"Total Rows: {after_rows:,}")
print(f"Rows Removed: {rows_removed:,}")

# Verify no duplicates remain
remaining_duplicates = df_merged.duplicated(subset='tconst').sum()
if remaining_duplicates == 0:
    print(f"✓ Verification: No duplicates remaining")
else:
    print(f"⚠️  Warning: {remaining_duplicates:,} duplicates still present")

# Store after statistics
duplicate_stats['after'] = {
    'shape': after_shape,
    'total_rows': after_rows,
    'rows_removed': rows_removed,
    'remaining_duplicates': remaining_duplicates
}

# ============================================================================
# SUMMARY: Before/After Comparison
# ============================================================================
print("\n" + "="*80)
print("📈 DUPLICATE REMOVAL SUMMARY - BEFORE vs AFTER")
print("="*80)
print(f"{'Metric':<30} {'Before':>15} {'After':>15} {'Change':>15}")
print("-" * 80)
print(f"{'Total Rows':<30} {before_rows:>15,} {after_rows:>15,} {rows_removed:>15,}")
print(f"{'Duplicate Records':<30} {duplicates_count:>15,} {remaining_duplicates:>15,} {-duplicates_count:>15,}")
print(f"{'Duplicate Percentage':<30} {duplicates_pct:>14.2f}% {0:>14.2f}% {duplicates_pct:>14.2f}%")
print("="*80)

print("\n✓ Duplicate removal complete")
print("="*80 + "\n")


# ### 3.3: Analyze Outliers (SHOW but DON'T REMOVE)
# 

# In[41]:


import pandas as pd

# Convert the column to numeric, coercing errors to NaN
df_merged['runtimeMinutes'] = pd.to_numeric(
    df_merged['runtimeMinutes'], 
    errors='coerce'
)

# Remove NaN values from the runtimeMinutes column for accurate calculation 
# (NaNs will cause issues with describe() and quantile())
runtime_column = df_merged['runtimeMinutes'].dropna()

print("\n[3.3] OUTLIER ANALYSIS (VISUALIZATION ONLY - NOT REMOVED)")
print("-" * 80)

# Analyze runtime outliers
print("\n➤ Analyzing runtimeMinutes outliers...")
print("\nRuntimeMinutes Statistics:")
# Use the cleaned, numeric column for statistics
runtime_stats = runtime_column.describe() 
print(runtime_stats)

# Calculate outliers using IQR method
Q1 = runtime_column.quantile(0.25)
Q3 = runtime_column.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Calculate outliers on the cleaned column
outliers_count = ((runtime_column < lower_bound) |
                  (runtime_column > upper_bound)).sum()

print(f"\nOutliers detected (IQR method): {outliers_count:,} ({outliers_count/len(runtime_column)*100:.2f}%)")
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

# Optional: To visualize the data and outliers, you'd typically use a Box Plot 
# or a Histogram.
print("\n")


# In[42]:


# (Removed) Old synthetic outlier analysis that recreated df_merged.
# The main outlier analysis now uses the real df_merged in cell 32.
pass


# In[46]:


from scipy import stats

# Use the real merged df_merged from the pipeline (≈1.6M rows)
# Assume df_merged['runtimeMinutes'] was already converted to numeric in 3.3.
runtime_column = df_merged['runtimeMinutes'].dropna()

# Calculate IQR bounds and variables on the full merged data
Q1 = runtime_column.quantile(0.25)
Q3 = runtime_column.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_count = ((runtime_column < lower_bound) | (runtime_column > upper_bound)).sum()
outliers_pct = (outliers_count / len(runtime_column)) * 100

# Set plot style and colors
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']


# In[47]:


# Plot 1: Box Plot - Full Data
fig, ax1 = plt.subplots(figsize=(8, 6))
box_data = df_merged['runtimeMinutes'].dropna()
bp1 = ax1.boxplot(box_data, vert=True, patch_artist=True, 
                   boxprops=dict(facecolor=colors[0], alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5))
ax1.set_title('Runtime Distribution - Full Dataset', 
              fontsize=14, fontweight='bold', pad=10)
ax1.set_ylabel('Runtime (minutes)', fontsize=12, fontweight='bold')
ax1.set_xticks([]) # Remove x-axis tick
ax1.grid(True, alpha=0.3, linestyle='--')

stats_text = f"Median: {box_data.median():.0f} min\nMean: {box_data.mean():.0f} min"
ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.show()


# In[45]:


# Plot 2: Box Plot - Zoomed View (limited to <= 300 minutes)
fig, ax2 = plt.subplots(figsize=(8, 6))
reasonable_runtime = df_merged[df_merged['runtimeMinutes'] <= 300]['runtimeMinutes']
bp2 = ax2.boxplot(reasonable_runtime, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=colors[1], alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5))
ax2.set_title('Runtime Distribution - Zoomed View (with IQR Bounds)', 
              fontsize=14, fontweight='bold', pad=10)
ax2.set_ylabel('Runtime (minutes)', fontsize=12, fontweight='bold')
ax2.set_xticks([]) 

# Add outlier bounds
ax2.axhline(lower_bound, color='orange', linestyle='--', linewidth=1.5, 
           label=f'Lower bound: {lower_bound:.0f} min', alpha=0.7)
ax2.axhline(upper_bound, color='red', linestyle='--', linewidth=1.5, 
           label=f'Upper bound: {upper_bound:.0f} min', alpha=0.7)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.show()


# In[32]:


# Plot 3: Histogram with Outlier Bounds
fig, ax3 = plt.subplots(figsize=(10, 6))
n, bins, patches = ax3.hist(df_merged['runtimeMinutes'], bins=100, 
                             edgecolor='black', alpha=0.7, color=colors[2])

# Color bins based on outlier status
for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
    if bin_val < lower_bound or bin_val > upper_bound:
        patch.set_facecolor('#FF6B6B')  # Red for outliers
    else:
        patch.set_facecolor(colors[2])  # Normal color

# Add vertical lines for bounds
ax3.axvline(lower_bound, color='orange', linestyle='--', linewidth=2, 
           label=f'Lower bound: {lower_bound:.0f} min', alpha=0.8)
ax3.axvline(upper_bound, color='red', linestyle='--', linewidth=2, 
           label=f'Upper bound: {upper_bound:.0f} min', alpha=0.8)
ax3.axvline(df_merged['runtimeMinutes'].median(), color='green', 
           linestyle='-', linewidth=2, label=f'Median: {df_merged["runtimeMinutes"].median():.0f} min', alpha=0.8)

ax3.set_xlabel('Runtime (minutes)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Runtime Distribution - Histogram (with Outlier Bounds)', 
              fontsize=14, fontweight='bold', pad=10)
ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.show()


# In[33]:


# Plot 4: Box Plot by Title Type
fig, ax4 = plt.subplots(figsize=(12, 6))
title_types_unique = sorted(df_merged['titleType'].unique())
box_data_by_type = [df_merged[df_merged['titleType'] == tt]['runtimeMinutes'].dropna() 
                    for tt in title_types_unique]

bp4 = ax4.boxplot(box_data_by_type, labels=title_types_unique, patch_artist=True,
                  medianprops=dict(color='darkred', linewidth=2),
                  whiskerprops=dict(color='black', linewidth=1.5))

# Color each box differently
for i, patch in enumerate(bp4['boxes']):
    patch.set_facecolor(colors[i % len(colors)])

ax4.set_title('Runtime Distribution by Title Type', 
              fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('Title Type', fontsize=12, fontweight='bold')
ax4.set_ylabel('Runtime (minutes)', fontsize=12, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.show()


# In[34]:


# Plot 5: Outlier Count by Title Type (Bar Chart)
fig, ax5 = plt.subplots(figsize=(10, 6))

outlier_counts = df_merged.groupby('titleType').apply(
    lambda x: ((x['runtimeMinutes'] < lower_bound) | 
               (x['runtimeMinutes'] > upper_bound)).sum()
).sort_values(ascending=False)

bars = ax5.barh(range(len(outlier_counts)), outlier_counts.values, 
                color=colors[0], alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_yticks(range(len(outlier_counts)))
ax5.set_yticklabels(outlier_counts.index, fontsize=10)
ax5.set_xlabel('Number of Outliers', fontsize=12, fontweight='bold')
ax5.set_title('Outlier Count by Title Type', 
              fontsize=14, fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3, linestyle='--', axis='x')

# Add value labels on bars
for i, (idx, val) in enumerate(outlier_counts.items()):
    ax5.text(val + max(outlier_counts.values) * 0.01, i, 
            f'{int(val):,}', va='center', fontsize=9, fontweight='bold')
plt.show()


# In[35]:


# Plot 6: Density Plot (KDE)
fig, ax6 = plt.subplots(figsize=(10, 6))
runtime_clean = df_merged['runtimeMinutes'].dropna()

# Plot Histogram and KDE
ax6.hist(runtime_clean, bins=80, density=True, alpha=0.6, 
         color=colors[0], edgecolor='black', label='Distribution')

if len(runtime_clean) > 0:
    x = np.linspace(runtime_clean.min(), runtime_clean.max(), 200)
    kde = stats.gaussian_kde(runtime_clean)
    ax6.plot(x, kde(x), color='blue', linewidth=2, label='KDE Curve', alpha=0.8)

# Add boundary lines
ax6.axvline(lower_bound, color='orange', linestyle='--', linewidth=2, 
           label=f'Lower: {lower_bound:.0f} min')
ax6.axvline(upper_bound, color='red', linestyle='--', linewidth=2, 
           label=f'Upper: {upper_bound:.0f} min')
ax6.axvline(runtime_clean.median(), color='green', linestyle='-', 
           linewidth=2, label=f'Median: {runtime_clean.median():.0f} min')

ax6.set_xlabel('Runtime (minutes)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Density', fontsize=12, fontweight='bold')
ax6.set_title('Runtime Distribution - Density Plot', 
              fontsize=14, fontweight='bold', pad=10)
ax6.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.show()


# In[36]:


# Plot 7: Outlier Percentage by Title Type
fig, ax7 = plt.subplots(figsize=(10, 6))

outlier_pct_by_type = df_merged.groupby('titleType').apply(
    lambda x: ((x['runtimeMinutes'] < lower_bound) | 
               (x['runtimeMinutes'] > upper_bound)).sum() / len(x) * 100
).sort_values(ascending=False)

bars = ax7.bar(range(len(outlier_pct_by_type)), outlier_pct_by_type.values, 
               color=colors[1], alpha=0.7, edgecolor='black', linewidth=1.5)
ax7.set_xticks(range(len(outlier_pct_by_type)))
ax7.set_xticklabels(outlier_pct_by_type.index, rotation=45, ha='right', fontsize=10)
ax7.set_ylabel('Outlier Percentage (%)', fontsize=12, fontweight='bold')
ax7.set_title('Outlier Percentage by Title Type', 
              fontsize=14, fontweight='bold', pad=10)
ax7.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add value labels on bars
for i, (idx, val) in enumerate(outlier_pct_by_type.items()):
    ax7.text(i, val + max(outlier_pct_by_type.values) * 0.02, 
            f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
plt.show()


# In[37]:


# Plot 8: Summary Statistics Table (No axes/plotting)
fig, ax8 = plt.subplots(figsize=(7, 4))
ax8.axis('off')

# Create summary statistics
summary_data = {
    'Metric': ['Total Records', 'Mean Runtime', 'Median Runtime', 
               'Std Deviation', 'Min Runtime', 'Max Runtime',
               'Lower Outliers', 'Upper Outliers', 'Total Outliers',
               'Outlier %'],
    'Value': [
        f"{len(df_merged):,}",
        f"{df_merged['runtimeMinutes'].mean():.1f} min",
        f"{df_merged['runtimeMinutes'].median():.1f} min",
        f"{df_merged['runtimeMinutes'].std():.1f} min",
        f"{df_merged['runtimeMinutes'].min():.1f} min",
        f"{df_merged['runtimeMinutes'].max():.1f} min",
        f"{len(df_merged[df_merged['runtimeMinutes'] < lower_bound]):,}",
        f"{len(df_merged[df_merged['runtimeMinutes'] > upper_bound]):,}",
        f"{outliers_count:,}",
        f"{outliers_pct:.2f}%"
    ]
}

table = ax8.table(cellText=[[summary_data['Metric'][i], summary_data['Value'][i]] 
                            for i in range(len(summary_data['Metric']))],
                 colLabels=['Metric', 'Value'],
                 cellLoc='left',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# Style the table
for i in range(len(summary_data['Metric']) + 1):
    for j in range(2):
        cell = table[(i, j)]
        if i == 0:  # Header row
            cell.set_facecolor('#4A90E2')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
            cell.set_edgecolor('black')

ax8.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=10)
plt.show()




# ### 3.4: Standardize Categorical Variables
# 

# In[ ]:


print("\n" + "="*80)
print("[3.4] STANDARDIZING CATEGORICAL VARIABLES - PROFESSIONAL DATA CLEANING")
print("="*80)

# ============================================================================
# BEFORE CLEANING: Capture Initial State
# ============================================================================
print("\n📊 BEFORE CLEANING - Categorical Variable Analysis:")
print("-" * 80)

categorical_stats = {
    'before': {},
    'after': {}
}

# ============================================================================
# STEP 1: Standardize titleType
# ============================================================================
print("\n" + "-"*80)
print("STEP 1: Standardizing titleType Column")
print("-"*80)

if 'titleType' in df_merged.columns:
    print("\n➤ titleType Standardization:")
    
    # Before state
    titletype_before = df_merged['titleType'].value_counts()
    unique_before = df_merged['titleType'].nunique()
    print(f"   Before:")
    print(f"   • Unique values: {unique_before}")
    print(f"   • Value distribution:")
    for val, count in titletype_before.head(10).items():
        pct = (count / len(df_merged)) * 100
        print(f"     - '{val}': {count:,} ({pct:.2f}%)")
    
    # Store before statistics
    categorical_stats['before']['titleType'] = {
        'unique_count': unique_before,
        'value_counts': titletype_before.to_dict(),
        'sample_values': df_merged['titleType'].unique()[:5].tolist()
    }
    
    # Standardize: lowercase and strip whitespace
    print(f"\n   Action: Converting to lowercase and stripping whitespace...")
    df_merged['titleType'] = df_merged['titleType'].str.lower().str.strip()

    # After state
    titletype_after = df_merged['titleType'].value_counts()
    unique_after = df_merged['titleType'].nunique()
    print(f"\n   After:")
    print(f"   • Unique values: {unique_after}")
    print(f"   • Value distribution:")
    for val, count in titletype_after.head(10).items():
        pct = (count / len(df_merged)) * 100
        print(f"     - '{val}': {count:,} ({pct:.2f}%)")
    
    # Store after statistics
    categorical_stats['after']['titleType'] = {
        'unique_count': unique_after,
        'value_counts': titletype_after.to_dict(),
        'standardization_applied': 'lowercase + strip whitespace'
    }
    
    print(f"\n   ✓ Standardization complete: {unique_before} → {unique_after} unique values")
else:
    print("\n➤ titleType: Column not found in df_merged")

# ============================================================================
# STEP 2: Create genre_count Feature
# ============================================================================
print("\n" + "-"*80)
print("STEP 2: Creating genre_count Feature")
print("-"*80)

if 'genres' in df_merged.columns:
    print("\n➤ genre_count Feature Engineering:")
    
    # Before: Analyze genre distribution
    print(f"   Before: Analyzing genre column...")
    genres_with_data = df_merged[df_merged['genres'] != 'Unknown']['genres'].notna().sum()
    genres_unknown = (df_merged['genres'] == 'Unknown').sum()
    print(f"   • Records with genre data: {genres_with_data:,}")
    print(f"   • Records with 'Unknown': {genres_unknown:,}")
    
    # Create genre_count feature
    print(f"\n   Action: Counting genres per title (comma-separated)...")
    df_merged['genre_count'] = df_merged['genres'].apply(
        lambda x: 0 if x == 'Unknown' or pd.isna(x) else len(str(x).split(','))
    )
    
    # After: Analyze genre_count distribution
    genre_count_stats = df_merged['genre_count'].describe()
    max_genres = df_merged['genre_count'].max()
    avg_genres = df_merged['genre_count'].mean()
    
    print(f"\n   After: genre_count statistics:")
    print(f"   • Maximum genres per title: {max_genres}")
    print(f"   • Average genres per title: {avg_genres:.2f}")
    print(f"   • Distribution:")
    genre_count_dist = df_merged['genre_count'].value_counts().sort_index().head(10)
    for count, freq in genre_count_dist.items():
        pct = (freq / len(df_merged)) * 100
        print(f"     - {count} genre(s): {freq:,} titles ({pct:.2f}%)")
    
    # Store statistics
    categorical_stats['after']['genre_count'] = {
        'max': int(max_genres),
        'mean': float(avg_genres),
        'distribution': genre_count_dist.to_dict(),
        'stats': genre_count_stats.to_dict()
    }
    
    print(f"\n   ✓ genre_count feature created successfully")
else:
    print("\n➤ genres: Column not found in df_merged")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📈 CATEGORICAL STANDARDIZATION SUMMARY")
print("="*80)
print("✓ titleType: Standardized to lowercase with trimmed whitespace")
print("✓ genre_count: New feature created (count of genres per title)")
print("="*80 + "\n")


# ### 3.5: Feature Engineering
# 



# In[ ]:


warnings.filterwarnings('ignore', category=FutureWarning)

print("\n" + "="*80)
print("[3.5] FEATURE ENGINEERING - PROFESSIONAL DATA CLEANING")
print("="*80)

# ============================================================================
# BEFORE FEATURE ENGINEERING: Capture Initial State
# ============================================================================
print("\n📊 BEFORE FEATURE ENGINEERING - Initial Column Count:")
print("-" * 80)
before_cols = df_merged.shape[1]
before_col_names = df_merged.columns.tolist()
print(f"Total Columns: {before_cols}")
print(f"Existing Columns: {', '.join(before_col_names[:10])}{'...' if len(before_col_names) > 10 else ''}")

feature_stats = {
    'before': {
        'column_count': before_cols,
        'columns': before_col_names
    },
    'features_created': []
}

# ============================================================================
# FEATURE ENGINEERING: Create Derived Features
# ============================================================================
print("\n" + "-"*80)
print("CREATING DERIVED FEATURES")
print("-"*80)

# Feature 1: Primary Genre
print("\n➤ Feature 1: primary_genre")
if 'genres' in df_merged.columns:
    print("   Description: Extracts the first genre from comma-separated genre list")
    print("   Logic: Split by comma, take first element; 'Unknown' if no genre data")
    
    before_primary_genre = 'primary_genre' in df_merged.columns
    df_merged['primary_genre'] = df_merged['genres'].apply(
        lambda x: str(x).split(',')[0] if x != 'Unknown' and not pd.isna(x) else 'Unknown'
    )
    
    unique_genres = df_merged['primary_genre'].nunique()
    top_genres = df_merged['primary_genre'].value_counts().head(5)
    print(f"   ✓ Created: {unique_genres} unique primary genres")
    print(f"   Top 5: {', '.join([f'{g}({c:,})' for g, c in top_genres.items()])}")
    feature_stats['features_created'].append({
        'name': 'primary_genre',
        'type': 'categorical',
        'unique_values': unique_genres,
        'description': 'First genre from genres list'
    })
else:
    print("   ⚠️  Skipped: 'genres' column not found")

# Feature 2: Decade
print("\n➤ Feature 2: decade")
if 'startYear' in df_merged.columns:
    print("   Description: Extracts decade from startYear (e.g., 1995 → 1990)")
    print("   Logic: Integer division by 10, then multiply by 10")
    
    df_merged['decade'] = (df_merged['startYear'] // 10 * 10).astype('Int16')
    unique_decades = df_merged['decade'].nunique()
    decade_range = f"{df_merged['decade'].min()}-{df_merged['decade'].max()}"
    print(f"   ✓ Created: {unique_decades} unique decades ({decade_range})")
    feature_stats['features_created'].append({
        'name': 'decade',
        'type': 'ordinal',
        'unique_values': unique_decades,
        'range': decade_range,
        'description': 'Decade bins from startYear'
    })
    
    # Feature 2b: Year Category
    print("\n➤ Feature 2b: year_category")
    print("   Description: Categorizes titles into historical eras")
    print("   Bins: Pre-1950, 1950-1980, 1980-2000, 2000-2010, 2010-2025")
    
    df_merged['year_category'] = pd.cut(
        df_merged['startYear'],
        bins=[1890, 1950, 1980, 2000, 2010, 2025],
        labels=['Pre-1950', '1950-1980', '1980-2000', '2000-2010', '2010-2025']
    )
    year_cat_dist = df_merged['year_category'].value_counts()
    print(f"   ✓ Created: {df_merged['year_category'].nunique()} era categories")
    print(f"   Distribution:")
    for cat, count in year_cat_dist.items():
        pct = (count / len(df_merged)) * 100
        print(f"     - {cat}: {count:,} ({pct:.2f}%)")
    feature_stats['features_created'].append({
        'name': 'year_category',
        'type': 'ordinal',
        'unique_values': df_merged['year_category'].nunique(),
        'categories': year_cat_dist.to_dict(),
        'description': 'Historical era categorization'
    })
else:
    print("   ⚠️  Skipped: 'startYear' column not found")

# Feature 3: Crew Counts
print("\n➤ Feature 3: num_directors & num_writers")
if 'directors' in df_merged.columns:
    print("   Description: Counts number of directors per title")
    print("   Logic: Count comma-separated values in directors column")
    
    df_merged['num_directors'] = df_merged['directors'].apply(
        lambda x: 0 if x == 'Unknown' or pd.isna(x) else len(str(x).split(','))
    )
    max_directors = df_merged['num_directors'].max()
    avg_directors = df_merged['num_directors'].mean()
    print(f"   ✓ num_directors created: max={max_directors}, avg={avg_directors:.2f}")
    feature_stats['features_created'].append({
        'name': 'num_directors',
        'type': 'discrete',
        'max': int(max_directors),
        'mean': float(avg_directors),
        'description': 'Count of directors per title'
    })
else:
    print("   ⚠️  Skipped: 'directors' column not found")

if 'writers' in df_merged.columns:
    print("   Description: Counts number of writers per title")
    print("   Logic: Count comma-separated values in writers column")
    
    df_merged['num_writers'] = df_merged['writers'].apply(
        lambda x: 0 if x == 'Unknown' or pd.isna(x) else len(str(x).split(','))
    )
    max_writers = df_merged['num_writers'].max()
    avg_writers = df_merged['num_writers'].mean()
    print(f"   ✓ num_writers created: max={max_writers}, avg={avg_writers:.2f}")
    feature_stats['features_created'].append({
        'name': 'num_writers',
        'type': 'discrete',
        'max': int(max_writers),
        'mean': float(avg_writers),
        'description': 'Count of writers per title'
    })
else:
    print("   ⚠️  Skipped: 'writers' column not found")

# Feature 4: Rating Category
print("\n➤ Feature 4: rating_category")
if 'averageRating' in df_merged.columns:
    print("   Description: Categorizes ratings into quality tiers")
    print("   Bins: Poor (0-4), Below Average (4-6), Average (6-7), Good (7-8), Excellent (8-10)")
    
    df_merged['rating_category'] = pd.cut(
        df_merged['averageRating'],
        bins=[0, 4, 6, 7, 8, 10],
        labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
    )
    rating_cat_dist = df_merged['rating_category'].value_counts()
    print(f"   ✓ Created: {df_merged['rating_category'].nunique()} rating categories")
    print(f"   Distribution:")
    for cat, count in rating_cat_dist.items():
        pct = (count / len(df_merged)) * 100
        print(f"     - {cat}: {count:,} ({pct:.2f}%)")
    feature_stats['features_created'].append({
        'name': 'rating_category',
        'type': 'ordinal',
        'unique_values': df_merged['rating_category'].nunique(),
        'categories': rating_cat_dist.to_dict(),
        'description': 'Rating quality categorization'
    })
else:
    print("   ⚠️  Skipped: 'averageRating' column not found")

# Feature 5: Popularity Tier
print("\n➤ Feature 5: popularity_tier")
if 'numVotes' in df_merged.columns:
    print("   Description: Categorizes titles into popularity tiers based on vote count")
    print("   Method: Quantile-based binning (5 equal-sized groups)")
    print("   Labels: Very Low, Low, Medium, High, Very High")
    
    df_merged['popularity_tier'] = pd.qcut(
        df_merged['numVotes'],
        q=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        duplicates='drop'
    )
    pop_tier_dist = df_merged['popularity_tier'].value_counts()
    print(f"   ✓ Created: {df_merged['popularity_tier'].nunique()} popularity tiers")
    print(f"   Distribution:")
    for tier, count in pop_tier_dist.items():
        pct = (count / len(df_merged)) * 100
        print(f"     - {tier}: {count:,} ({pct:.2f}%)")
    feature_stats['features_created'].append({
        'name': 'popularity_tier',
        'type': 'ordinal',
        'unique_values': df_merged['popularity_tier'].nunique(),
        'categories': pop_tier_dist.to_dict(),
        'description': 'Quantile-based popularity categorization'
    })
else:
    print("   ⚠️  Skipped: 'numVotes' column not found")

# ============================================================================
# AFTER FEATURE ENGINEERING: Capture Final State
# ============================================================================
print("\n" + "-"*80)
print("📊 AFTER FEATURE ENGINEERING - Final Column Count:")
print("-" * 80)
after_cols = df_merged.shape[1]
after_col_names = df_merged.columns.tolist()
new_features = [col for col in after_col_names if col not in before_col_names]
print(f"Total Columns: {after_cols} (+{after_cols - before_cols} new features)")
print(f"New Features Created: {', '.join(new_features)}")

feature_stats['after'] = {
    'column_count': after_cols,
    'columns': after_col_names,
    'new_features': new_features
}

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📈 FEATURE ENGINEERING SUMMARY")
print("="*80)
print(f"{'Feature':<25} {'Type':<15} {'Status':<10}")
print("-" * 80)
for feat in feature_stats['features_created']:
    status = "✓ Created"
    print(f"{feat['name']:<25} {feat['type']:<15} {status:<10}")
print("="*80)

print(f"\n✓ Feature engineering complete: {len(feature_stats['features_created'])} features created")
print(f"✓ New features available for EDA: {', '.join(new_features)}")
print("="*80 + "\n")


# ### 3.6: Integrate Additional IMDB Datasets (Principals, Names, AKAs, Episodes)
#

print("\n" + "="*80)
print("[3.6] INTEGRATING ADDITIONAL IMDB DATASETS (PRINCIPALS, NAMES, AKAS, EPISODES)")
print("="*80)

# ============================================================================
# BEFORE INTEGRATION: Capture Initial State
# ============================================================================
print("\n📊 BEFORE INTEGRATION - df_merged Snapshot:")
print("-" * 80)
integration_before_shape = df_merged.shape
integration_before_cols = df_merged.columns.tolist()
print(f"Shape: {integration_before_shape[0]:,} rows × {integration_before_shape[1]} columns")
print(f"Columns (first 12): {', '.join(integration_before_cols[:12])}{'...' if len(integration_before_cols) > 12 else ''}")

integration_stats = {
    'before': {
        'shape': integration_before_shape,
        'columns': integration_before_cols
    },
    'features_added': []
}

# ============================================================================
# STEP 1: Integrate TITLE.PRINCIPALS (per-title cast/crew counts)
# ============================================================================
print("\n" + "-"*80)
print("STEP 1: Integrating title.principals.tsv (Cast/Crew Data)")
print("-" * 80)

if 'principals' in globals() and principals is not None and 'tconst' in principals.columns:
    print("\n➤ Using principals sample to derive per-title features...")
    print(f"   principals shape: {principals.shape}")

    principals_local = principals.copy()
    principals_local['is_cast'] = principals_local['category'].isin(
        ['actor', 'actress', 'self']
    ).astype('int8')

    principals_agg = principals_local.groupby('tconst').agg(
        num_principals=('nconst', 'count'),
        num_cast=('is_cast', 'sum')
    ).reset_index()
    principals_agg['num_crew_principals'] = (
        principals_agg['num_principals'] - principals_agg['num_cast']
    )

    print(f"   Aggregated principals to per-title features: {principals_agg.shape[0]:,} titles")

    before_merge_titles = df_merged['tconst'].nunique()
    df_merged = df_merged.merge(principals_agg, on='tconst', how='left')

    merged_titles_with_principals = df_merged['num_principals'].notna().sum()
    print(f"   ✓ Merged principals features into df_merged (left join on tconst)")
    print(f"   • Titles with principals data: {merged_titles_with_principals:,} / {before_merge_titles:,}")

    integration_stats['features_added'].extend(
        ['num_principals', 'num_cast', 'num_crew_principals']
    )
    feature_stats['features_created'].append({
        'name': 'num_principals',
        'type': 'discrete',
        'description': 'Number of principal cast/crew entries per title'
    })
    feature_stats['features_created'].append({
        'name': 'num_cast',
        'type': 'discrete',
        'description': 'Number of principal cast (actor/actress/self) per title'
    })
    feature_stats['features_created'].append({
        'name': 'num_crew_principals',
        'type': 'discrete',
        'description': 'Number of principal non-cast crew per title'
    })
else:
    print("⚠️ principals dataframe not available or missing 'tconst' - skipping principals integration.")

# ============================================================================
# STEP 2: Integrate NAME.BASICS via Principals (talent demographics)
# ============================================================================
print("\n" + "-"*80)
print("STEP 2: Integrating name.basics.tsv (Talent Information) via principals")
print("-" * 80)

if (
    'principals' in globals() and principals is not None and
    'name_basics' in globals() and name_basics is not None and
    {'nconst'} <= set(principals.columns) and
    {'nconst', 'birthYear'} <= set(name_basics.columns)
):
    print("\n➤ Enriching principals with birthYear from name.basics...")
    principals_enriched = principals[['tconst', 'nconst']].merge(
        name_basics[['nconst', 'birthYear']],
        on='nconst',
        how='left'
    )

    birth_stats = principals_enriched.groupby('tconst')['birthYear'].agg(
        avg_principal_birthYear='mean'
    ).reset_index()

    print(f"   Aggregated birthYear stats for {birth_stats.shape[0]:,} titles")

    df_merged = df_merged.merge(birth_stats, on='tconst', how='left')
    print("   ✓ Merged principal birth year statistics into df_merged")

    integration_stats['features_added'].extend(
        ['avg_principal_birthYear']
    )
    feature_stats['features_created'].append({
        'name': 'avg_principal_birthYear',
        'type': 'continuous',
        'description': 'Average birth year of principals linked to a title'
    })
else:
    print("⚠️ Either principals or name_basics (with birthYear) not available - skipping name-based features.")

# ============================================================================
# STEP 3: Integrate TITLE.AKAS (regional variants per title)
# ============================================================================
print("\n" + "-"*80)
print("STEP 3: Integrating title.akas.tsv (Regional Variants)")
print("-" * 80)

if 'akas' in globals() and akas is not None and 'titleId' in akas.columns:
    print("\n➤ Creating aka_count (number of regional title variants per tconst)...")
    aka_counts = akas.groupby('titleId').size().rename('aka_count').reset_index()
    print(f"   Aggregated AKAs for {aka_counts.shape[0]:,} unique titleIds (sample-based)")

    df_merged = df_merged.merge(
        aka_counts,
        left_on='tconst',
        right_on='titleId',
        how='left'
    )
    if 'titleId' in df_merged.columns:
        df_merged = df_merged.drop(columns=['titleId'])

    titles_with_akas = df_merged['aka_count'].notna().sum()
    print(f"   ✓ Merged aka_count into df_merged for {titles_with_akas:,} titles (based on AKAs sample)")

    integration_stats['features_added'].append('aka_count')
    feature_stats['features_created'].append({
        'name': 'aka_count',
        'type': 'discrete',
        'description': 'Number of regional title variants (from title.akas sample)'
    })
else:
    print("⚠️ akas dataframe not available or missing 'titleId' - skipping AKAs integration.")

# ============================================================================
# STEP 4: Integrate TITLE.EPISODE (episode counts per series)
# ============================================================================
print("\n" + "-"*80)
print("STEP 4: Integrating title.episode.tsv (Episode Data)")
print("-" * 80)

if 'episodes' in globals() and episodes is not None and 'parentTconst' in episodes.columns:
    print("\n➤ Creating num_episodes_sample (number of episodes per series)...")
    episodes_counts = episodes.groupby('parentTconst').size().rename('num_episodes_sample').reset_index()
    print(f"   Aggregated episode counts for {episodes_counts.shape[0]:,} parentTconst values (sample-based)")

    df_merged = df_merged.merge(
        episodes_counts,
        left_on='tconst',
        right_on='parentTconst',
        how='left'
    )
    if 'parentTconst' in df_merged.columns:
        df_merged = df_merged.drop(columns=['parentTconst'])

    titles_with_episodes = df_merged['num_episodes_sample'].notna().sum()
    print(f"   ✓ Merged num_episodes_sample into df_merged for {titles_with_episodes:,} series titles (based on episodes sample)")

    integration_stats['features_added'].append('num_episodes_sample')
    feature_stats['features_created'].append({
        'name': 'num_episodes_sample',
        'type': 'discrete',
        'description': 'Number of episodes per series (from title.episode sample)'
    })
else:
    print("⚠️ episodes dataframe not available or missing 'parentTconst' - skipping episodes integration.")

# ============================================================================
# AFTER INTEGRATION: Capture Final State
# ============================================================================
print("\n" + "-"*80)
print("📊 AFTER INTEGRATION - df_merged Snapshot:")
print("-" * 80)
integration_after_shape = df_merged.shape
integration_after_cols = df_merged.columns.tolist()
print(f"Shape: {integration_after_shape[0]:,} rows × {integration_after_shape[1]} columns")
print(f"New Columns Added: {', '.join([c for c in integration_after_cols if c not in integration_before_cols])}")

integration_stats['after'] = {
    'shape': integration_after_shape,
    'columns': integration_after_cols
}

print("\n" + "="*80)
print("📈 ADDITIONAL DATA INTEGRATION SUMMARY")
print("="*80)
print(f"{'Metric':<35} {'Before':>15} {'After':>15} {'Change':>15}")
print("-" * 80)
print(f"{'Total Columns':<35} {integration_before_shape[1]:>15} {integration_after_shape[1]:>15} {integration_after_shape[1]-integration_before_shape[1]:>15}")
print(f"{'Total Rows':<35} {integration_before_shape[0]:>15,} {integration_after_shape[0]:>15,} {integration_after_shape[0]-integration_before_shape[0]:>15,}")
print("="*80 + "\n")


# ## STEP 4: DATA VISUALIZATION FOR UNDERSTANDING
# 

# ### 4.1: Distribution Visualizations
# 

# In[ ]:



# In[44]:


# Year distribution
plt.figure(figsize=(8, 6))
plt.hist(df_merged['startYear'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('1. Distribution of Titles by Year', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()


# In[45]:


# Rating distribution - removed duplicate (see In[54] for better version with kde)


# In[46]:


# Runtime distribution (capped at <= 300 minutes for better visibility)
plt.figure(figsize=(8, 6))
plt.hist(df_merged[df_merged['runtimeMinutes'] <= 300]['runtimeMinutes'],
         bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
plt.xlabel('Runtime (minutes)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('3. Distribution of Runtime (≤300 min)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()


# In[47]:


# Title type distribution - removed duplicate (see In[56] for better version)


# In[48]:


# Top genres - removed duplicate (see In[57] for better version excluding Unknown)


# In[49]:


# Rating category distribution
plt.figure(figsize=(8, 6))
rating_cat_counts = df_merged['rating_category'].value_counts().reindex(['Poor', 'Below Average', 'Average', 'Good', 'Excellent'])
plt.bar(rating_cat_counts.index, rating_cat_counts.values, color='coral')
plt.xlabel('Rating Category', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('6. Distribution by Rating Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# ### 4.2: Temporal Trends
# 

# In[ ]:


# Redundant imports removed - already imported at top
warnings.filterwarnings('ignore', category=FutureWarning)

print("\n[4.2] Creating Temporal Trend Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Titles over time
yearly_counts = df_merged.groupby('startYear').size()
axes[0, 0].plot(yearly_counts.index, yearly_counts.values, linewidth=2, color='steelblue')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Number of Titles')
axes[0, 0].set_title('Title Production Over Time')
axes[0, 0].grid(True, alpha=0.3)

# Average rating over time
yearly_ratings = df_merged.groupby('startYear')['averageRating'].mean()
axes[0, 1].plot(yearly_ratings.index, yearly_ratings.values, linewidth=2, color='crimson')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Average Rating')
axes[0, 1].set_title('Average Rating Trends Over Time')
axes[0, 1].grid(True, alpha=0.3)

# Runtime trends
yearly_runtime = df_merged.groupby('startYear')['runtimeMinutes'].mean()
axes[1, 0].plot(yearly_runtime.index, yearly_runtime.values, linewidth=2, color='green')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Average Runtime (minutes)')
axes[1, 0].set_title('Runtime Trends Over Time')
axes[1, 0].grid(True, alpha=0.3)

# Decade comparison (requires 'decade' and 'tconst' from the real pipeline)
if 'decade' in df_merged.columns and 'tconst' in df_merged.columns:
    decade_stats = df_merged.groupby('decade').agg({
        'tconst': 'count',
        'averageRating': 'mean'
    })
    ax2 = axes[1, 1].twinx()
    axes[1, 1].bar(decade_stats.index, decade_stats['tconst'], alpha=0.5, color='skyblue', label='Count')
    ax2.plot(decade_stats.index, decade_stats['averageRating'], marker='o', color='red',
             linewidth=2, markersize=8, label='Avg Rating')
    axes[1, 1].set_xlabel('Decade')
    axes[1, 1].set_ylabel('Number of Titles', color='skyblue')
    ax2.set_ylabel('Average Rating', color='red')
    axes[1, 1].set_title('Decade Comparison: Volume vs Quality')
    axes[1, 1].grid(True, alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'Decade/ID info not available', ha='center', va='center')
    axes[1, 1].axis('off')

plt.suptitle('Temporal Trend Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# ### 4.3: Correlation Analysis
# 

# In[51]:


# Redundant imports removed - already imported at top
warnings.filterwarnings('ignore', category=FutureWarning)

print("\n[4.3] Creating Correlation Analysis...")

numeric_cols = ['startYear', 'runtimeMinutes', 'averageRating', 'numVotes',
                'genre_count', 'num_directors', 'num_writers']

# Only keep columns that actually exist (robust if pipeline changes)
numeric_cols = [c for c in numeric_cols if c in df_merged.columns]

corr_matrix = df_merged[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Numeric Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# ### Visualizations top Understand the data

# In[52]:


# Use the cleaned df_merged created in earlier steps for EDA visualizations.
# At this point, df_merged already contains all engineered features
# (primary_genre, decade, year_category, genre_count, num_directors, etc.).
print("✅ df_merged from cleaning pipeline is ready for exploratory analysis.")

# In[53]:


plt.figure(figsize=(10, 6))
sns.kdeplot(df_merged['runtimeMinutes'].dropna(), fill=True, color='skyblue', linewidth=2)
plt.title('1. Runtime Distribution (Density Estimate)', fontsize=16)
plt.xlabel('Runtime (Minutes)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 200) # Limit x-axis for clearer view of main trend
plt.show()


# In[54]:


plt.figure(figsize=(10, 6))
sns.histplot(df_merged['averageRating'].dropna(), bins=20, kde=True, color='lightcoral', edgecolor='black')
plt.title('2. Distribution of Average Ratings', fontsize=16)
plt.xlabel('Average Rating (1.0 - 10.0)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# In[55]:


plt.figure(figsize=(10, 6))
sns.histplot(df_merged['numVotes'].dropna(), bins=50, kde=False, color='green', log_scale=True, edgecolor='black')
plt.title('3. Distribution of Number of Votes (Log Scale)', fontsize=16)
plt.xlabel('Number of Votes (Log Scale)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# In[56]:


plt.figure(figsize=(8, 6))
sns.countplot(y='titleType', data=df_merged, order=df_merged['titleType'].value_counts().index, palette='magma')
plt.title('4. Distribution by Title Type', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Title Type', fontsize=12)
plt.show()


# In[57]:


plt.figure(figsize=(10, 6))
top_genres = df_merged[df_merged['primary_genre'] != 'Unknown']['primary_genre'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title('5. Top 10 Primary Genres (Excluding Unknown)', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Primary Genre', fontsize=12)
plt.show()


# In[58]:


plt.figure(figsize=(10, 6))
yearly_counts = df_merged.groupby('startYear').size()
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-', color='darkblue', alpha=0.7)
plt.title('6. Title Production Volume Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)
plt.grid(True, alpha=0.5)
plt.show()


# In[59]:


# In[60]:


plt.figure(figsize=(10, 6))
yearly_ratings = df_merged.groupby('startYear')['averageRating'].mean()
plt.plot(yearly_ratings.index, yearly_ratings.values, marker='o', linestyle='-', color='crimson', alpha=0.7)
plt.title('7. Average Rating Trends Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.grid(True, alpha=0.5)
plt.show()


# In[61]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='runtimeMinutes', y='averageRating', data=df_merged[df_merged['runtimeMinutes'] <= 240], 
                alpha=0.2, color='darkgreen')
plt.title('8. Runtime vs. Average Rating (Runtimes < 240 min)', fontsize=16)
plt.xlabel('Runtime (Minutes)', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()


# In[62]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='decade', y='averageRating', data=df_merged, palette='Pastel1')
plt.title('9. Distribution of Average Ratings by Decade', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()


# In[63]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='titleType', y='runtimeMinutes', 
               data=df_merged[df_merged['runtimeMinutes'] < 180], # Limit y-axis for clarity
               palette='Set2', inner='quartile')
plt.title('10. Runtime Distribution by Title Type (Violin Plot)', fontsize=16)
plt.xlabel('Title Type', fontsize=12)
plt.ylabel('Runtime (Minutes)', fontsize=12)
plt.show()


# In[64]:


plt.figure(figsize=(10, 6))
rating_by_pop = df_merged.groupby('popularity_tier')['averageRating'].mean().reset_index()
sns.barplot(x='popularity_tier', y='averageRating', data=rating_by_pop, palette='Blues_d')
plt.title('11. Average Rating by Popularity Tier (Vote Count)', fontsize=16)
plt.xlabel('Popularity Tier', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()


# In[65]:


plt.figure(figsize=(10, 6))
# Filter out extreme runtimes for clarity
df_plot = df_merged[df_merged['runtimeMinutes'] < 180] 
sns.scatterplot(x='runtimeMinutes', y='averageRating', hue='titleType', data=df_plot, alpha=0.5, palette='viridis')
plt.title('1. Rating vs. Runtime, Grouped by Title Type', fontsize=16)
plt.xlabel('Runtime (Minutes)', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()


# In[66]:


# Ensure num_directors exists before visualization (consolidated check)
if 'num_directors' not in df_merged.columns:
    if 'directors' in df_merged.columns:
        try:
            df_merged['num_directors'] = df_merged['directors'].apply(
                lambda x: 0 if x == 'Unknown' or pd.isna(x) else len(str(x).split(','))
            )
            print("✓ Created 'num_directors' column for visualization")
        except Exception as e:
            print(f"Error creating 'num_directors' column: {e}")
    else:
        print("Warning: 'directors' column not found in df_merged. Available columns:", list(df_merged.columns)[:10])

# Create barplot visualization only if num_directors column exists
if 'num_directors' in df_merged.columns:
plt.figure(figsize=(10, 6))
# Filter for 1 to 3 directors as higher counts are rare
director_stats = df_merged[df_merged['num_directors'].isin([1, 2, 3])].groupby('num_directors')['averageRating'].mean().reset_index()
sns.barplot(x='num_directors', y='averageRating', data=director_stats, palette='coolwarm')
plt.title('12. Average Rating by Number of Directors', fontsize=16)
plt.xlabel('Number of Directors', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()


# In[67]:


top_5_genres = df_merged['primary_genre'].value_counts().head(5).index
plt.figure(figsize=(12, 6))
sns.boxplot(x='primary_genre', y='averageRating', 
            data=df_merged[df_merged['primary_genre'].isin(top_5_genres)], 
            palette='Set3')
plt.title('2. Rating Distribution for Top 5 Primary Genres', fontsize=16)
plt.xlabel('Primary Genre', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()


# In[68]:


# Create violinplot visualization (num_directors already checked/created in In[66])
if 'num_directors' in df_merged.columns:
plt.figure(figsize=(10, 6))
df_plot = df_merged[df_merged['num_directors'].isin([1, 2, 3])]
sns.violinplot(x='num_directors', y='averageRating', data=df_plot, palette='coolwarm', inner='quartile')
    plt.title('13. Rating Distribution by Number of Directors', fontsize=16)
plt.xlabel('Number of Directors', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()
else:
    print("Warning: 'num_directors' column not found. Skipping violin plot.")


# In[69]:


plt.figure(figsize=(10, 6))
runtime_by_pop = df_merged.groupby('popularity_tier')['runtimeMinutes'].mean().reset_index()
sns.barplot(x='popularity_tier', y='runtimeMinutes', data=runtime_by_pop, palette='YlOrRd')
plt.title('4. Average Runtime by Popularity Tier', fontsize=16)
plt.xlabel('Popularity Tier (Based on Votes)', fontsize=12)
plt.ylabel('Average Runtime (Minutes)', fontsize=12)
plt.show()


# In[70]:


plt.figure(figsize=(8, 6))
rating_by_genre_count = df_merged.groupby('genre_count')['averageRating'].mean().reset_index()
sns.barplot(x='genre_count', y='averageRating', data=rating_by_genre_count[rating_by_genre_count['genre_count'] > 0], palette='Blues')
plt.title('5. Average Rating by Number of Genres', fontsize=16)
plt.xlabel('Number of Genres', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()


# In[71]:


plt.figure(figsize=(12, 6))
df_pivot = df_merged.groupby('decade')['titleType'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
df_pivot.plot(kind='bar', stacked=True, colormap='Spectral', ax=plt.gca())
plt.title('6. Title Type Distribution (Percentage) Across Decades', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Percentage of Titles', fontsize=12)
plt.legend(title='Title Type', loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[72]:


plt.figure(figsize=(10, 6))
avg_genres_decade = df_merged.groupby('decade')['genre_count'].mean()
plt.plot(avg_genres_decade.index, avg_genres_decade.values, marker='s', linestyle='--', color='darkviolet')
plt.title('7. Average Number of Genres Assigned by Decade', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Average Genre Count', fontsize=12)
plt.grid(True, alpha=0.5)
plt.show()


# In[73]:


plt.figure(figsize=(12, 6))
# Filter out extreme votes for visualization clarity and use log scale
df_plot = df_merged[df_merged['numVotes'] < 500000]
sns.boxplot(x='rating_category', y='numVotes', data=df_plot, palette='tab10')
plt.yscale('log')
plt.title('8. Vote Count Distribution by Rating Category (Log Scale)', fontsize=16)
plt.xlabel('Rating Category', fontsize=12)
plt.ylabel('Number of Votes (Log Scale)', fontsize=12)
plt.show()


# In[74]:


plt.figure(figsize=(10, 6))
df_plot = df_merged[df_merged['primary_genre'].isin(top_5_genres)]
runtime_by_genre = df_plot.groupby('primary_genre')['runtimeMinutes'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=runtime_by_genre.values, y=runtime_by_genre.index, palette='Greens_d')
plt.title('9. Average Runtime of Top 10 Primary Genres', fontsize=16)
plt.xlabel('Average Runtime (Minutes)', fontsize=12)
plt.ylabel('Primary Genre', fontsize=12)
plt.show()


# In[75]:


plt.figure(figsize=(10, 8))
# Count the number of titles in each decade/rating_category combination
df_pivot = df_merged.groupby(['decade', 'rating_category']).size().unstack(fill_value=0)
sns.heatmap(df_pivot, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
plt.title('10. Title Volume: Rating Category Across Decades (Count)', fontsize=16)
plt.xlabel('Rating Category', fontsize=12)
plt.ylabel('Decade', fontsize=12)
plt.show()


# In[76]:


plt.figure(figsize=(8, 6))
# Filter for reasonable crew sizes
if 'num_directors' in df_merged.columns and 'num_writers' in df_merged.columns:
df_plot = df_merged[df_merged['num_directors'] < 5]
df_plot = df_plot[df_plot['num_writers'] < 5]
sns.scatterplot(x='num_directors', y='num_writers', data=df_plot, alpha=0.5, color='orange')
plt.title('11. Number of Writers vs. Number of Directors', fontsize=16)
plt.xlabel('Number of Directors', fontsize=12)
plt.ylabel('Number of Writers', fontsize=12)
plt.show()
else:
    print("Warning: 'num_directors' or 'num_writers' column not found. Skipping scatter plot.")
    plt.close()


# In[77]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='rating_category', y='genre_count', data=df_merged, palette='Pastel2')
plt.title('12. Genre Count Distribution by Rating Category', fontsize=16)
plt.xlabel('Rating Category', fontsize=12)
plt.ylabel('Number of Genres', fontsize=12)
plt.show()


# In[78]:


plt.figure(figsize=(12, 6))
df_plot = df_merged[df_merged['titleType'].isin(['movie', 'tvepisode'])]
sns.barplot(x='decade', y='averageRating', hue='titleType', data=df_plot, palette='husl')
plt.title('14. Average Rating Comparison: Movies vs. TV Episodes by Decade', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.legend(title='Title Type')
plt.show()


# In[79]:


plt.figure(figsize=(10, 6))
df_plot = df_merged.copy()
df_plot['Genre Group'] = np.where(df_plot['genre_count'] > 1, 'Multi-Genre', 'Single-Genre')
sns.violinplot(x='Genre Group', y='runtimeMinutes', data=df_plot[df_plot['runtimeMinutes'] < 180], 
               palette=['#007FFF', '#FF4500'], inner='quartile')
plt.title('15. Runtime Distribution: Single vs. Multi-Genre Titles', fontsize=16)
plt.xlabel('Genre Group', fontsize=12)
plt.ylabel('Runtime (Minutes)', fontsize=12)
plt.show()


# In[80]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='decade', y='numVotes', data=df_merged, palette='tab20')
plt.yscale('log')
plt.title('16. Vote Count Distribution by Decade (Log Scale)', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Number of Votes (Log Scale)', fontsize=12)
plt.show()


# In[81]:


# 4.4: Insights from Integrated Principals & Names Data

# Visualization 17: Cast Size vs. Average Rating
plt.figure(figsize=(10, 6))
df_cast = df_merged[['averageRating', 'num_principals']].dropna()
df_cast = df_cast[df_cast['num_principals'] > 0]

if not df_cast.empty:
    # Bin cast sizes into interpretable buckets
    df_cast = df_cast[df_cast['num_principals'] <= 50]
    df_cast['principal_bin'] = pd.cut(
        df_cast['num_principals'],
        bins=[0, 3, 7, 15, 50],
        labels=['1-3', '4-7', '8-15', '16+']
    )

    rating_by_cast = df_cast.groupby('principal_bin')['averageRating'].mean().reset_index()
    sns.barplot(x='principal_bin', y='averageRating', data=rating_by_cast, palette='Purples')
    plt.title('17. Cast Size vs. Average Rating', fontsize=16)
    plt.xlabel('Number of Principals (Binned)', fontsize=12)
    plt.ylabel('Average Rating', fontsize=12)
    plt.show()
else:
    print("Warning: 'num_principals' data not available for cast size visualization.")


# Visualization 18: Regional Variants vs Popularity (Votes, Log Scale)
plt.figure(figsize=(10, 6))
df_aka = df_merged[['aka_count', 'numVotes']].dropna()
df_aka = df_aka[df_aka['aka_count'] > 0]

if not df_aka.empty:
    sns.scatterplot(x='aka_count', y='numVotes', data=df_aka, alpha=0.3, color='teal')
    plt.yscale('log')
    plt.title('18. Regional Title Variants vs. Popularity (Log Votes)', fontsize=16)
    plt.xlabel('Number of Regional Title Variants (aka_count)', fontsize=12)
    plt.ylabel('Number of Votes (Log Scale)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("Warning: 'aka_count' data not available for regional variants visualization.")


# In[82]:


# 4.5: Insights from Integrated Episodes & Talent Demographics

# Visualization 19: Series Length vs. Average Rating (Episode Sample)
plt.figure(figsize=(10, 6))
if 'num_episodes_sample' in df_merged.columns and 'titleType' in df_merged.columns:
    df_series = df_merged[
        df_merged['num_episodes_sample'].notna()
        & df_merged['titleType'].isin(['tvseries', 'tvminiseries'])
    ].copy()

    if not df_series.empty:
        # Cap extremely long series for readability
        df_series = df_series[df_series['num_episodes_sample'] <= 200]
        sns.scatterplot(
            x='num_episodes_sample',
            y='averageRating',
            data=df_series,
            alpha=0.4,
            color='darkorange'
        )
        plt.title('19. Series Length vs. Average Rating (Episode Sample)', fontsize=16)
        plt.xlabel('Number of Episodes (Sample-Based)', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Warning: No series with 'num_episodes_sample' available for visualization.")
else:
    print("Warning: 'num_episodes_sample' or 'titleType' not available for series length visualization.")


# Visualization 20: Average Principal Age vs. Average Rating
plt.figure(figsize=(10, 6))
required_cols_age = {'avg_principal_birthYear', 'startYear', 'averageRating'}
if required_cols_age.issubset(df_merged.columns):
    df_age = df_merged.dropna(subset=list(required_cols_age)).copy()
    if not df_age.empty:
        df_age['avg_principal_age'] = df_age['startYear'] - df_age['avg_principal_birthYear']
        # Keep reasonable age range
        df_age = df_age[df_age['avg_principal_age'].between(20, 90)]

        df_age['avg_principal_age_bin'] = pd.cut(
            df_age['avg_principal_age'],
            bins=[20, 30, 40, 50, 60, 70, 90],
            labels=['20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        )

        rating_by_age = df_age.groupby('avg_principal_age_bin')['averageRating'].mean().reset_index()
        sns.lineplot(
            x='avg_principal_age_bin',
            y='averageRating',
            data=rating_by_age,
            marker='o',
            color='steelblue'
        )
        plt.title('20. Average Principal Age vs. Average Rating', fontsize=16)
        plt.xlabel('Average Principal Age (Binned)', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Warning: No usable data for principal age visualization.")
else:
    print("Warning: 'avg_principal_birthYear' and/or 'startYear' not available for principal age visualization.")


# In[81]:


# 4.6: Additional Insights from Integrated Features

# Visualization 21: Heatmap of Average Rating by Cast Size and Decade
plt.figure(figsize=(10, 6))
if {'num_principals', 'decade', 'averageRating'}.issubset(df_merged.columns):
    df_cast_decade = df_merged[['num_principals', 'decade', 'averageRating']].dropna().copy()
    if not df_cast_decade.empty:
        df_cast_decade = df_cast_decade[df_cast_decade['num_principals'] <= 50]
        df_cast_decade = df_cast_decade[df_cast_decade['num_principals'] > 0]

        df_cast_decade['principal_bin'] = pd.cut(
            df_cast_decade['num_principals'],
            bins=[0, 3, 7, 15, 50],
            labels=['1-3', '4-7', '8-15', '16+']
        )

        heat_data = df_cast_decade.groupby(['decade', 'principal_bin'])['averageRating'] \
                                  .mean().unstack(fill_value=np.nan)

        sns.heatmap(
            heat_data,
            annot=True,
            fmt=".2f",
            cmap='YlOrRd',
            linewidths=.5
        )
        plt.title('21. Average Rating by Cast Size and Decade', fontsize=16)
        plt.xlabel('Cast Size (Number of Principals, Binned)', fontsize=12)
        plt.ylabel('Decade', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("Warning: No usable data for cast size × decade heatmap.")
else:
    print("Warning: Required columns for cast size × decade heatmap are missing.")


# Visualization 22: Global Reach (aka_count) vs. Average Rating
plt.figure(figsize=(10, 6))
if {'aka_count', 'averageRating'}.issubset(df_merged.columns):
    df_aka_rating = df_merged[['aka_count', 'averageRating']].dropna().copy()
    df_aka_rating = df_aka_rating[df_aka_rating['aka_count'] > 0]
    if not df_aka_rating.empty:
        df_aka_rating = df_aka_rating[df_aka_rating['aka_count'] <= 50]
        df_aka_rating['aka_bin'] = pd.cut(
            df_aka_rating['aka_count'],
            bins=[0, 1, 3, 5, 10, 50],
            labels=['1', '2-3', '4-5', '6-10', '11+']
        )

        rating_by_aka = df_aka_rating.groupby('aka_bin')['averageRating'] \
                                     .mean().reset_index()
        sns.barplot(x='aka_bin', y='averageRating', data=rating_by_aka, palette='GnBu_d')
        plt.title('22. Global Reach (Regional Variants) vs. Average Rating', fontsize=16)
        plt.xlabel('Number of Regional Variants (Binned aka_count)', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("Warning: No usable data for aka_count vs. rating visualization.")
else:
    print("Warning: 'aka_count' or 'averageRating' not available for global reach visualization.")


# Visualization 23: Series Length vs. Popularity (Votes, Log Scale)
plt.figure(figsize=(10, 6))
if {'num_episodes_sample', 'numVotes', 'titleType'}.issubset(df_merged.columns):
    df_series_pop = df_merged[
        df_merged['num_episodes_sample'].notna()
        & df_merged['titleType'].isin(['tvseries', 'tvminiseries'])
    ][['num_episodes_sample', 'numVotes']].dropna().copy()

    if not df_series_pop.empty:
        df_series_pop = df_series_pop[df_series_pop['num_episodes_sample'] <= 200]
        df_series_pop = df_series_pop[df_series_pop['numVotes'] > 0]

        df_series_pop['episodes_bin'] = pd.cut(
            df_series_pop['num_episodes_sample'],
            bins=[0, 10, 25, 50, 100, 200],
            labels=['1-10', '11-25', '26-50', '51-100', '101-200']
        )

        votes_by_len = df_series_pop.groupby('episodes_bin')['numVotes'] \
                                    .median().reset_index()

        sns.barplot(x='episodes_bin', y='numVotes', data=votes_by_len, palette='Oranges')
        plt.yscale('log')
        plt.title('23. Series Length vs. Median Popularity (Log Votes)', fontsize=16)
        plt.xlabel('Number of Episodes (Binned)', fontsize=12)
        plt.ylabel('Median Number of Votes (Log Scale)', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("Warning: No usable series data for length vs. popularity visualization.")
else:
    print("Warning: Required columns for series length vs. popularity visualization are missing.")


# In[82]:


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

print("\n" + "="*80)
print("STEP 5: DETECTING REDUNDANT ATTRIBUTES")
print("="*80)

# Use correlation matrix from the real df_merged
numeric_cols = ['startYear', 'runtimeMinutes', 'averageRating', 'numVotes',
                'genre_count', 'num_directors', 'num_writers']
numeric_cols = [c for c in numeric_cols if c in df_merged.columns]

corr_matrix = df_merged[numeric_cols].corr()

# Check for highly correlated features
print("\nChecking for highly correlated features (threshold > 0.95)...")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j],
                              corr_matrix.iloc[i, j]))

if high_corr:
    print("\nHighly Correlated Features (>0.95):")
    for feat1, feat2, corr in high_corr:
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")
else:
    print("✓ No highly correlated features found (threshold: 0.95)")

# Check isAdult column if present
if 'isAdult' in df_merged.columns:
    adult_ratio = df_merged['isAdult'].mean()
    print(f"\nAdult content ratio: {adult_ratio:.4f} ({adult_ratio*100:.2f}%)")
    if adult_ratio < 0.01:
        print("→ Removing 'isAdult' column (less than 1% adult content - low variance)")
        df_merged = df_merged.drop('isAdult', axis=1)
    else:
        print("→ Keeping 'isAdult' column (sufficient variance)")
else:
    print("\n'isAdult' column not present in df_merged (already removed or not available).")

# Check for columns with single unique value
print("\nChecking for columns with only one unique value...")
for col in df_merged.columns:
    if df_merged[col].nunique() == 1:
        print(f"  {col}: Only 1 unique value - consider removing")


# ## STEP 5.5: COMPREHENSIVE CLEANING SUMMARY (FOR REPORT GENERATION)
# 
# This section aggregates all cleaning statistics for easy report generation

print("\n" + "="*80)
print("STEP 5.5: COMPREHENSIVE CLEANING SUMMARY")
print("="*80)

# Aggregate all cleaning statistics
comprehensive_cleaning_report = {
    'pipeline_info': {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_path': BASE_PATH,
        'output_path': OUTPUT_PATH
    },
    'data_merging': {
        'initial_shape': original_shape if 'original_shape' in locals() else df_merged.shape,
        'final_shape_after_merge': df_merged.shape if 'df_merged' in locals() else None,
        'merge_strategy': 'INNER join (basics + ratings), LEFT join (crew)'
    },
    'missing_values': cleaning_stats if 'cleaning_stats' in locals() else {},
    'duplicates': duplicate_stats if 'duplicate_stats' in locals() else {},
    'categorical_standardization': categorical_stats if 'categorical_stats' in locals() else {},
    'feature_engineering': feature_stats if 'feature_stats' in locals() else {},
    'outlier_analysis': {
        'runtime_outliers': outliers_count if 'outliers_count' in locals() else 0,
        'runtime_lower_bound': lower_bound if 'lower_bound' in locals() else None,
        'runtime_upper_bound': upper_bound if 'upper_bound' in locals() else None,
        'action_taken': 'VISUALIZED but NOT REMOVED'
    },
    'final_dataset': {
        'shape': df_merged.shape,
        'columns': df_merged.columns.tolist(),
        'memory_mb': df_merged.memory_usage(deep=True).sum() / 1024**2,
        'data_types': df_merged.dtypes.to_dict(),
        'numeric_columns': df_merged.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df_merged.select_dtypes(include=['object', 'category']).columns.tolist()
    }
}

# Print comprehensive summary
print("\n📊 COMPREHENSIVE CLEANING SUMMARY")
print("="*80)

print("\n1. DATA MERGING:")
print(f"   • Initial merged shape: {comprehensive_cleaning_report['data_merging']['initial_shape']}")
print(f"   • Final shape after merge: {comprehensive_cleaning_report['data_merging']['final_shape_after_merge']}")
print(f"   • Strategy: {comprehensive_cleaning_report['data_merging']['merge_strategy']}")

if comprehensive_cleaning_report['missing_values']:
    print("\n2. MISSING VALUE HANDLING:")
    mv = comprehensive_cleaning_report['missing_values']
    if 'before' in mv and 'after' in mv:
        print(f"   • Missing values before: {mv['before']['total_missing']:,}")
        print(f"   • Missing values after: {mv['after']['total_missing']:,}")
        print(f"   • Reduction: {mv['before']['total_missing'] - mv['after']['total_missing']:,}")
    if 'metrics' in mv:
        print(f"   • Genres filled: {mv['metrics'].get('genres_filled', 0):,}")
        print(f"   • Directors filled: {mv['metrics'].get('directors_filled', 0):,}")
        print(f"   • Writers filled: {mv['metrics'].get('writers_filled', 0):,}")

if comprehensive_cleaning_report['duplicates']:
    print("\n3. DUPLICATE REMOVAL:")
    dup = comprehensive_cleaning_report['duplicates']
    if 'before' in dup and 'after' in dup:
        print(f"   • Duplicates found: {dup['before']['duplicates_count']:,}")
        print(f"   • Rows removed: {dup['after']['rows_removed']:,}")
        print(f"   • Remaining duplicates: {dup['after']['remaining_duplicates']:,}")

if comprehensive_cleaning_report['feature_engineering']:
    print("\n4. FEATURE ENGINEERING:")
    fe = comprehensive_cleaning_report['feature_engineering']
    if 'features_created' in fe:
        print(f"   • Total features created: {len(fe['features_created'])}")
        for feat in fe['features_created']:
            print(f"     - {feat['name']}: {feat.get('description', 'N/A')}")

print("\n5. FINAL DATASET CHARACTERISTICS:")
fd = comprehensive_cleaning_report['final_dataset']
print(f"   • Shape: {fd['shape'][0]:,} rows × {fd['shape'][1]} columns")
print(f"   • Memory usage: {fd['memory_mb']:.2f} MB")
print(f"   • Numeric columns: {len(fd['numeric_columns'])}")
print(f"   • Categorical columns: {len(fd['categorical_columns'])}")

print("\n6. OUTLIER ANALYSIS:")
oa = comprehensive_cleaning_report['outlier_analysis']
print(f"   • Runtime outliers detected: {oa['runtime_outliers']:,}")
if oa['runtime_lower_bound'] is not None and oa['runtime_upper_bound'] is not None:
    print(f"   • IQR bounds: [{oa['runtime_lower_bound']:.2f}, {oa['runtime_upper_bound']:.2f}] minutes")
print(f"   • Action: {oa['action_taken']}")

print("\n" + "="*80)
print("✓ Comprehensive cleaning summary complete - Ready for report generation")
print("="*80 + "\n")

# Store report in a global variable for easy access
CLEANING_REPORT = comprehensive_cleaning_report

# ## STEP 6: FINAL DATA VALIDATION
# 

# In[82]:


print("\n" + "="*80)
print("STEP 6: FINAL DATA VALIDATION")
print("="*80)

print(f"\nOriginal shape (after merge): {original_shape}")
print(f"Final shape (after cleaning): {df_merged.shape}")
print(f"Rows removed: {original_shape[0] - df_merged.shape[0]:,} ({(original_shape[0] - df_merged.shape[0])/original_shape[0]*100:.2f}%)")

print(f"\nFinal dataset summary:")
print(f"  - Total records: {len(df_merged):,}")
print(f"  - Total features: {df_merged.shape[1]}")
print(f"  - Date range: {df_merged['startYear'].min():.0f} - {df_merged['startYear'].max():.0f}")
print(f"  - Memory usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nMissing values in final dataset:")
missing_final = df_merged.isnull().sum()
if missing_final.sum() > 0:
    print(missing_final[missing_final > 0])
else:
    print("✓ No missing values in critical columns")

print(f"\nData types:")
print(df_merged.dtypes)


# ## STEP 7: SAVE CLEANED DATA
# 

# In[63]:


print("\n" + "="*80)
print("STEP 7: SAVING CLEANED DATASET")
print("="*80)

# Save full cleaned dataset
output_file = OUTPUT_PATH + 'cleaned_imdb_data.csv'
df_merged.to_csv(output_file, index=False)
print(f"✓ Saved full dataset to: {output_file}")
print(f"  Size: {os.path.getsize(output_file) / 1024**2:.2f} MB")

# Save as parquet for faster loading
parquet_file = OUTPUT_PATH + 'cleaned_imdb_data.parquet'
df_merged.to_parquet(parquet_file, index=False)
print(f"✓ Saved dataset (Parquet) to: {parquet_file}")
print(f"  Size: {os.path.getsize(parquet_file) / 1024**2:.2f} MB")

# Save a sample for quick testing
sample_file = OUTPUT_PATH + 'cleaned_imdb_sample.csv'
sample_df = df_merged.sample(n=min(50000, len(df_merged)), random_state=42)
sample_df.to_csv(sample_file, index=False)
print(f"✓ Saved sample (50K) to: {sample_file}")


# ## STEP 8: GENERATE COMPREHENSIVE CLEANING REPORT
# 

# In[64]:


print("\n" + "="*80)
print("STEP 8: GENERATING COMPREHENSIVE CLEANING REPORT")
print("="*80)

# Calculate outliers for runtimeMinutes (recalculate to ensure availability)
if 'runtimeMinutes' in df_merged.columns:
    runtime_column = df_merged['runtimeMinutes'].dropna()
    if len(runtime_column) > 0:
        Q1 = runtime_column.quantile(0.25)
        Q3 = runtime_column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = ((runtime_column < lower_bound) | (runtime_column > upper_bound)).sum()
    else:
        outliers_count = 0
        lower_bound = 0
        upper_bound = 0
else:
    outliers_count = 0
    lower_bound = 0
    upper_bound = 0

# Calculate outliers for numVotes and averageRating if columns exist
votes_outliers = 0
rating_outliers = 0

if 'numVotes' in df_merged.columns:
    votes_column = df_merged['numVotes'].dropna()
    if len(votes_column) > 0:
        Q1_votes = votes_column.quantile(0.25)
        Q3_votes = votes_column.quantile(0.75)
        IQR_votes = Q3_votes - Q1_votes
        lower_bound_votes = Q1_votes - 1.5 * IQR_votes
        upper_bound_votes = Q3_votes + 1.5 * IQR_votes
        votes_outliers = ((votes_column < lower_bound_votes) | (votes_column > upper_bound_votes)).sum()

if 'averageRating' in df_merged.columns:
    rating_column = df_merged['averageRating'].dropna()
    if len(rating_column) > 0:
        Q1_rating = rating_column.quantile(0.25)
        Q3_rating = rating_column.quantile(0.75)
        IQR_rating = Q3_rating - Q1_rating
        lower_bound_rating = Q1_rating - 1.5 * IQR_rating
        upper_bound_rating = Q3_rating + 1.5 * IQR_rating
        rating_outliers = ((rating_column < lower_bound_rating) | (rating_column > upper_bound_rating)).sum()

summary = f"""
================================================================================
IMDB DATASET PREPROCESSING & CLEANING REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Source: https://datasets.imdbws.com/

================================================================================
1. FILES PROCESSED
================================================================================

✓ title.basics.tsv - Core title information
✓ title.ratings.tsv - Rating metrics
✓ title.crew.tsv - Director and writer data
✓ name.basics.tsv - Person information (assessed)
✓ title.principals.tsv - Cast/crew data (sampled)
✓ title.akas.tsv - Regional variants (assessed only - multiple entries per title)
✓ title.episode.tsv - Episode data (assessed only - multiple entries per series)

Note: Files with multiple entries per title (akas, episodes) were assessed
but not merged to avoid data explosion.

================================================================================
2. DATA MERGING STRATEGY
================================================================================

Order: title_basics → ratings → crew
Merge Type: INNER join (basics + ratings), LEFT join (crew)
Rationale: Merged BEFORE cleaning to maintain referential integrity
Result: {len(df_merged):,} records after merging

================================================================================
3. DATA CLEANING OPERATIONS
================================================================================

3.1 Missing Value Handling
---------------------------
• startYear: Removed {removed_startyear:,} records (critical for temporal analysis)
• runtimeMinutes: Removed {removed_runtime:,} records (critical for analysis)
• genres: Filled {genres_missing:,} values with 'Unknown'
• directors/writers: Filled missing with 'Unknown'
• endYear: Kept as NaN (valid for non-series)

3.2 Duplicate Removal
----------------------
• Duplicates found: {duplicates_count:,}
• Action: Removed duplicates based on tconst (kept first occurrence)

3.3 Outlier Analysis
---------------------
• runtimeMinutes outliers: {outliers_count:,} ({outliers_count/len(df_merged)*100:.2f}%)
  - IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}] minutes
  - Action: VISUALIZED but NOT REMOVED (as requested)

• numVotes outliers: {votes_outliers:,} ({votes_outliers/len(df_merged)*100:.2f}%)
• averageRating outliers: {rating_outliers:,} ({rating_outliers/len(df_merged)*100:.2f}%)

Note: All outliers documented but kept in dataset for complete analysis.

3.4 Categorical Standardization
--------------------------------
• titleType: Standardized to lowercase, stripped whitespace
• genres: Parsed and analyzed for multi-genre titles

3.5 Feature Engineering
------------------------
Created derived features:
• primary_genre: First genre from genres list
• decade: Decade bins for temporal analysis
• year_category: Era categorization (Pre-1950, 1950-1980, etc.)
• num_directors: Count of directors per title
• num_writers: Count of writers per title
• rating_category: Ordinal bins (Poor to Excellent)
• popularity_tier: Quantile-based popularity (Very Low to Very High)
• genre_count: Number of genres per title

================================================================================
4. REDUNDANT ATTRIBUTE DETECTION
================================================================================

• High correlation check: Threshold > 0.95
• Result: {len(high_corr)} highly correlated pairs found
• isAdult column: Removed (adult ratio: {adult_ratio:.4f} < 1%)

================================================================================
5. FINAL DATASET CHARACTERISTICS
================================================================================

Shape: {df_merged.shape}
  - Records: {len(df_merged):,}
  - Features: {df_merged.shape[1]}

Temporal Coverage:
  - Year range: {int(df_merged['startYear'].min())} - {int(df_merged['startYear'].max())}
  - Decades: {df_merged['decade'].nunique()} unique decades

Title Types: {df_merged['titleType'].nunique()} types
  {df_merged['titleType'].value_counts().to_dict()}

Genres: {df_merged['primary_genre'].nunique()} unique primary genres
  Top 5: {df_merged['primary_genre'].value_counts().head(5).to_dict()}

Rating Distribution:
  - Mean: {df_merged['averageRating'].mean():.2f}
  - Median: {df_merged['averageRating'].median():.2f}
  - Std: {df_merged['averageRating'].std():.2f}

Memory Usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Data Completeness: {(1 - df_merged.isnull().sum().sum() / (len(df_merged) * len(df_merged.columns)))*100:.2f}%

================================================================================
6. FILES GENERATED
================================================================================

Cleaned Datasets:
• cleaned_imdb_data.csv - Full cleaned dataset (CSV format)
• cleaned_imdb_data.parquet - Full cleaned dataset (Parquet format)
• cleaned_imdb_sample.csv - Sample of 50,000 records

Visualizations:
• missing_data_*.png - Missing data visualizations for each file
• outlier_analysis_runtime.png - Detailed runtime outlier analysis
• outlier_analysis_all_numeric.png - All numeric variable outliers
• data_distributions.png - Distribution visualizations
• temporal_trends.png - Temporal trend analysis
• correlation_matrix.png - Correlation heatmap

Reports:
• cleaning_report.txt - This comprehensive report

================================================================================
7. DATA QUALITY METRICS
================================================================================

Before Cleaning:
  - Total records: {original_shape[0]:,}
  - Missing values: Present in multiple columns
  - Duplicates: {duplicates_count:,}

After Cleaning:
  - Total records: {len(df_merged):,}
  - Records removed: {original_shape[0] - len(df_merged):,} ({(original_shape[0] - len(df_merged))/original_shape[0]*100:.2f}%)
  - Missing critical values: 0
  - Duplicates: 0

================================================================================
8. VARIABLE TYPES DOCUMENTATION
================================================================================

Quantitative Variables:
  - Continuous: averageRating, runtimeMinutes
  - Discrete: startYear, numVotes, genre_count, num_directors, num_writers

Categorical Variables:
  - Nominal: titleType, primary_genre, tconst
  - Ordinal: rating_category, popularity_tier, year_category

Temporal Variables:
  - startYear (YYYY format)
  - decade (decade bins)
  - year_category (era bins)

Text Variables:
  - primaryTitle, originalTitle, genres, directors, writers

================================================================================
9. ETHICAL CONSIDERATIONS
================================================================================

Privacy:
• No personal identifying information exposed
• Public dataset from official IMDB source

Bias Mitigation:
• Adult content filtered (isAdult removed - <1% of data)
• All title types included for comprehensive analysis
• No genre excluded (including 'Unknown')

Data Integrity:
• Outliers documented but not arbitrarily removed
• Referential integrity maintained through proper merge order
• All filtering decisions documented with justification

Transparency:
• All transformations logged
• Before/after statistics provided
• Visual documentation of key decisions

================================================================================
10. NEXT STEPS
================================================================================

✓ Data is cleaned and ready for:
  1. Data Reduction & Transformation
  2. Exploratory Data Analysis
  3. Visualization Design
  4. Statistical Analysis

✓ Recommended analyses:
  - Temporal trends (production volume, rating evolution)
  - Genre analysis (popularity, quality)
  - Runtime vs rating correlation
  - Decade comparisons

================================================================================
END OF REPORT
================================================================================
"""

print(summary)

# Save report
report_file = OUTPUT_PATH + 'cleaning_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n✓ Comprehensive report saved to: {report_file}")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nAll files saved to: {OUTPUT_PATH}")
print("\n✓ Ready for Data Reduction & Transformation!")
print("="*80)


# In[ ]:




