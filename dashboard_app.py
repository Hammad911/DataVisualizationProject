"""
IMDb Data Visualization Dashboard
===================================
Modern, professional dashboard showcasing key findings from the IMDb dataset analysis.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from scipy.stats import gaussian_kde
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODERN COLOR PALETTE
# ============================================================================
COLORS = {
    'primary': '#6366f1',      # Modern indigo
    'secondary': '#8b5cf6',    # Purple
    'success': '#10b981',      # Green
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'info': '#3b82f6',         # Blue
    'dark': '#1e293b',         # Slate dark
    'light': '#f8fafc',        # Slate light
    'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
}

# ============================================================================
# LOAD DATA (OPTIMIZED WITH SAMPLING FOR LARGE DATASETS)
# ============================================================================
print("🔄 Loading IMDb dataset...")
try:
    # Try parquet first (faster)
    df = pd.read_parquet('processed/cleaned_imdb_data.parquet')
    print(f"✓ Loaded from Parquet: {len(df):,} records")
except:
    # Fallback to CSV with sampling if too large
    df = pd.read_csv('processed/cleaned_imdb_data.csv')
    print(f"✓ Loaded from CSV: {len(df):,} records")
    
    # Sample if dataset is very large (for performance)
    if len(df) > 500000:
        print(f"⚡ Sampling 500K records for optimal performance...")
        df = df.sample(n=500000, random_state=42)

print(f"📊 Dataset: {len(df):,} records × {df.shape[1]} features")

# Convert numeric columns
numeric_cols = ['startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 
                'genre_count', 'num_directors', 'num_writers', 'decade']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================================
# HELPER FUNCTIONS FOR DATA PROCESSING
# ============================================================================

def create_genre_evolution_data():
    """Process data for genre evolution visualization"""
    genre_data = []
    for idx, row in df.iterrows():
        if pd.notna(row['genres']) and pd.notna(row['decade']):
            genres_list = str(row['genres']).split(',')
            for genre in genres_list:
                genre = genre.strip()
                if genre and genre != 'Unknown':
                    genre_data.append({'decade': row['decade'], 'genre': genre})
    
    genre_df = pd.DataFrame(genre_data)
    genre_evolution = genre_df.groupby(['decade', 'genre']).size().reset_index(name='count')
    top_genres = genre_df['genre'].value_counts().head(10).index.tolist()
    return genre_evolution[genre_evolution['genre'].isin(top_genres)]

def create_genre_cooccurrence_data():
    """Create genre co-occurrence matrix"""
    df_genres = df[df['genres'].notna() & (df['genres'] != 'Unknown')].copy()
    genre_pairs = []
    genre_ratings = []
    
    for idx, row in df_genres.iterrows():
        genres_list = [g.strip() for g in str(row['genres']).split(',')]
        if len(genres_list) >= 2:
            for g1, g2 in combinations(sorted(genres_list), 2):
                genre_pairs.append((g1, g2))
                if pd.notna(row['averageRating']):
                    genre_ratings.append((g1, g2, row['averageRating']))
    
    pair_counts = Counter(genre_pairs)
    top_pairs = pair_counts.most_common(30)
    return pd.DataFrame(top_pairs, columns=['pair', 'count'])

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_overview_stats():
    """Create overview statistics cards"""
    total_titles = len(df)
    avg_rating = df['averageRating'].mean()
    year_range = f"{int(df['startYear'].min())}-{int(df['startYear'].max())}"
    unique_genres = df['primary_genre'].nunique()
    
    return {
        'total_titles': f"{total_titles:,}",
        'avg_rating': f"{avg_rating:.2f}",
        'year_range': year_range,
        'unique_genres': unique_genres
    }

def create_temporal_trends():
    """Create modern temporal trend visualizations"""
    yearly_counts = df.groupby('startYear').size().reset_index(name='count')
    yearly_ratings = df.groupby('startYear')['averageRating'].mean().reset_index()
    yearly_runtime = df.groupby('startYear')['runtimeMinutes'].mean().reset_index()
    
    # Titles over time with modern styling
    fig1 = px.area(yearly_counts, x='startYear', y='count',
                   title='📊 Title Production Over Time',
                   labels={'startYear': 'Year', 'count': 'Number of Titles'})
    fig1.update_traces(fillcolor=COLORS['primary'], line_color=COLORS['primary'], opacity=0.6)
    fig1.update_layout(template='plotly_white', hovermode='x unified')
    
    # Average rating over time
    fig2 = px.line(yearly_ratings, x='startYear', y='averageRating',
                   title='⭐ Average Rating Trends',
                   labels={'startYear': 'Year', 'averageRating': 'Average Rating'})
    fig2.update_traces(line_color=COLORS['warning'], line_width=3)
    fig2.update_layout(template='plotly_white', hovermode='x unified')
    
    # Runtime trends
    fig3 = px.line(yearly_runtime, x='startYear', y='runtimeMinutes',
                   title='⏱️ Runtime Trends',
                   labels={'startYear': 'Year', 'runtimeMinutes': 'Average Runtime (minutes)'})
    fig3.update_traces(line_color=COLORS['success'], line_width=3)
    fig3.update_layout(template='plotly_white', hovermode='x unified')
    
    return fig1, fig2, fig3

def create_genre_evolution():
    """Create modern genre evolution area chart"""
    genre_data = create_genre_evolution_data()
    
    fig = px.area(genre_data, x='decade', y='count', color='genre',
                  title='🎭 Genre Evolution Across Decades',
                  labels={'decade': 'Decade', 'count': 'Number of Titles', 'genre': 'Genre'},
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_runtime_vs_rating():
    """Create modern runtime vs rating sweet spot analysis"""
    df_runtime = df[(df['titleType'] == 'movie') & 
                    (df['runtimeMinutes'].notna()) &
                    (df['averageRating'].notna()) &
                    (df['runtimeMinutes'] > 0) &
                    (df['runtimeMinutes'] < 300)].copy()
    
    # Calculate averages per runtime bin
    runtime_bins = np.arange(60, 180, 5)
    rating_means = []
    
    for i in range(len(runtime_bins)-1):
        mask = (df_runtime['runtimeMinutes'] >= runtime_bins[i]) & \
               (df_runtime['runtimeMinutes'] < runtime_bins[i+1])
        if mask.sum() > 5:
            rating_means.append(df_runtime[mask]['averageRating'].mean())
        else:
            rating_means.append(np.nan)
    
    runtime_centers = runtime_bins[:-1] + 2.5
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=runtime_centers, 
        y=rating_means,
        mode='lines+markers',
        name='Average Rating',
        line=dict(color=COLORS['danger'], width=4, shape='spline'),
        marker=dict(size=10, color=COLORS['danger'], line=dict(width=2, color='white')),
        fill='tonexty',
        fillcolor=f'rgba(239, 68, 68, 0.1)'
    ))
    
    fig.update_layout(
        title='🎯 Runtime vs Rating - The Sweet Spot',
        xaxis_title='Runtime (minutes)',
        yaxis_title='Average Rating',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

def create_rating_distributions():
    """Create modern rating distribution visualizations"""
    # Rating histogram with gradient
    fig1 = px.histogram(df, x='averageRating', nbins=20,
                       title='⭐ Distribution of Average Ratings',
                       labels={'averageRating': 'Average Rating', 'count': 'Frequency'})
    fig1.update_traces(marker_color=COLORS['warning'], marker_line_color='white', marker_line_width=1.5)
    fig1.update_layout(template='plotly_white')
    
    # Rating by title type with modern colors
    fig2 = px.box(df, x='titleType', y='averageRating',
                  title='📺 Rating Distribution by Title Type',
                  labels={'titleType': 'Title Type', 'averageRating': 'Average Rating'},
                  color='titleType',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(template='plotly_white', showlegend=False)
    
    # Rating by decade
    fig3 = px.violin(df, x='decade', y='averageRating',
                  title='📅 Rating Distribution by Decade',
                  labels={'decade': 'Decade', 'averageRating': 'Average Rating'},
                  color='decade',
                  color_discrete_sequence=px.colors.sequential.Viridis)
    fig3.update_layout(template='plotly_white', showlegend=False)
    
    return fig1, fig2, fig3

def create_genre_analysis():
    """Create modern genre-related visualizations"""
    # Top genres with gradient colors
    top_genres_data = df[df['primary_genre'] != 'Unknown']['primary_genre'].value_counts().head(15)
    fig1 = px.bar(x=top_genres_data.values, y=top_genres_data.index,
                  orientation='h',
                  title='🏆 Top 15 Primary Genres',
                  labels={'x': 'Count', 'y': 'Genre'})
    fig1.update_traces(marker_color=top_genres_data.values, 
                      marker_colorscale='Viridis',
                      marker_line_color='white',
                      marker_line_width=1.5)
    fig1.update_layout(template='plotly_white')
    
    # Rating by genre with modern styling
    top_5_genres = df['primary_genre'].value_counts().head(5).index
    df_top_genres = df[df['primary_genre'].isin(top_5_genres)]
    fig2 = px.violin(df_top_genres, x='primary_genre', y='averageRating',
                  title='🎭 Rating Distribution for Top 5 Genres',
                  labels={'primary_genre': 'Genre', 'averageRating': 'Rating'},
                  color='primary_genre',
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    fig2.update_layout(template='plotly_white', showlegend=False)
    
    # Genre count analysis with modern bars
    genre_count_stats = df[df['genre_count'] > 0].groupby('genre_count')['averageRating'].mean().reset_index()
    fig3 = px.bar(genre_count_stats, x='genre_count', y='averageRating',
                  title='📊 Average Rating by Number of Genres',
                  labels={'genre_count': 'Number of Genres', 'averageRating': 'Average Rating'})
    fig3.update_traces(marker_color=COLORS['secondary'], marker_line_color='white', marker_line_width=1.5)
    fig3.update_layout(template='plotly_white')
    
    return fig1, fig2, fig3

def create_popularity_analysis():
    """Create popularity-related visualizations"""
    # Votes distribution (log scale)
    fig1 = px.histogram(df, x='numVotes', log_y=True,
                       title='Distribution of Number of Votes (Log Scale)',
                       labels={'numVotes': 'Number of Votes', 'count': 'Frequency'})
    
    # Rating by popularity tier
    if 'popularity_tier' in df.columns:
        rating_by_pop = df.groupby('popularity_tier')['averageRating'].mean().reset_index()
        fig2 = px.bar(rating_by_pop, x='popularity_tier', y='averageRating',
                     title='Average Rating by Popularity Tier',
                     labels={'popularity_tier': 'Popularity Tier', 'averageRating': 'Average Rating'})
    else:
        fig2 = go.Figure()
    
    # Scatter: Rating vs Votes
    sample_df = df.sample(min(10000, len(df)), random_state=42)
    fig3 = px.scatter(sample_df, x='numVotes', y='averageRating',
                     color='primary_genre', opacity=0.5,
                     title='Rating vs Popularity (10K Sample)',
                     labels={'numVotes': 'Number of Votes', 'averageRating': 'Rating'},
                     log_x=True)
    
    return fig1, fig2, fig3

def create_decade_analysis():
    """Create decade comparison visualizations"""
    decade_stats = df.groupby('decade').agg({
        'tconst': 'count',
        'averageRating': 'mean'
    }).reset_index()
    decade_stats.columns = ['decade', 'count', 'avg_rating']
    
    # Dual axis: volume vs quality
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=decade_stats['decade'], y=decade_stats['count'],
               name='Number of Titles', marker_color='lightblue'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=decade_stats['decade'], y=decade_stats['avg_rating'],
                  name='Avg Rating', mode='lines+markers',
                  line=dict(color='crimson', width=3),
                  marker=dict(size=10)),
        secondary_y=True
    )
    
    fig.update_layout(title='Decade Analysis: Volume vs Quality',
                     height=500)
    fig.update_xaxes(title_text='Decade')
    fig.update_yaxes(title_text='Number of Titles', secondary_y=False)
    fig.update_yaxes(title_text='Average Rating', secondary_y=True)
    
    return fig

def create_runtime_analysis():
    """Create runtime-related visualizations"""
    # Runtime distribution
    df_runtime = df[(df['runtimeMinutes'] > 0) & (df['runtimeMinutes'] < 300)]
    fig1 = px.histogram(df_runtime, x='runtimeMinutes', nbins=50,
                       title='Runtime Distribution (< 300 min)',
                       labels={'runtimeMinutes': 'Runtime (minutes)', 'count': 'Frequency'})
    
    # Runtime by title type
    fig2 = px.violin(df_runtime, x='titleType', y='runtimeMinutes',
                    title='Runtime Distribution by Title Type',
                    labels={'titleType': 'Title Type', 'runtimeMinutes': 'Runtime (minutes)'})
    
    # Runtime by genre
    top_genres = df['primary_genre'].value_counts().head(10).index
    df_genre_runtime = df[df['primary_genre'].isin(top_genres)]
    genre_runtime_stats = df_genre_runtime.groupby('primary_genre')['runtimeMinutes'].mean().sort_values(ascending=False)
    
    fig3 = px.bar(x=genre_runtime_stats.values, y=genre_runtime_stats.index,
                  orientation='h',
                  title='Average Runtime by Genre (Top 10)',
                  labels={'x': 'Average Runtime (minutes)', 'y': 'Genre'})
    
    return fig1, fig2, fig3

def create_correlation_heatmap():
    """Create modern correlation matrix heatmap"""
    numeric_cols = ['startYear', 'runtimeMinutes', 'averageRating', 'numVotes',
                    'genre_count', 'num_directors', 'num_writers']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix,
                   labels=dict(color="Correlation"),
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   color_continuous_scale='RdBu_r',
                   zmin=-1, zmax=1,
                   title='🔗 Correlation Matrix of Numeric Variables',
                   aspect="auto")
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        xaxis_title='',
        yaxis_title=''
    )
    fig.update_traces(text=corr_matrix.round(2).values, texttemplate='%{text}')
    return fig

def create_advanced_insights():
    """Create advanced insight visualizations"""
    # Historical era genre distribution
    if 'year_category' in df.columns:
        era_genre_data = df[df['year_category'].notna()].groupby(['year_category', 'primary_genre']).size().reset_index(name='count')
        top_genres = df['primary_genre'].value_counts().head(8).index
        era_genre_filtered = era_genre_data[era_genre_data['primary_genre'].isin(top_genres)]
        
        fig1 = px.bar(era_genre_filtered, x='year_category', y='count', color='primary_genre',
                     title='Genre Distribution Across Historical Eras',
                     labels={'year_category': 'Era', 'count': 'Number of Titles'})
    else:
        fig1 = go.Figure()
    
    # Quality vs Quantity by decade
    decade_quality = df.groupby('decade').agg({
        'tconst': 'count',
        'averageRating': 'mean'
    }).reset_index()
    decade_quality['high_quality_pct'] = df[df['averageRating'] >= 7.5].groupby('decade').size() / df.groupby('decade').size() * 100
    
    fig2 = make_subplots(rows=1, cols=2,
                        subplot_titles=('Volume by Decade', 'Quality Percentage by Decade'))
    
    fig2.add_trace(
        go.Bar(x=decade_quality['decade'], y=decade_quality['tconst'],
               marker_color='steelblue'),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(x=decade_quality['decade'], y=decade_quality['high_quality_pct'],
                  mode='lines+markers', marker_color='green', line=dict(width=3)),
        row=1, col=2
    )
    
    fig2.update_xaxes(title_text='Decade', row=1, col=1)
    fig2.update_xaxes(title_text='Decade', row=1, col=2)
    fig2.update_yaxes(title_text='Number of Titles', row=1, col=1)
    fig2.update_yaxes(title_text='% High Quality (≥7.5)', row=1, col=2)
    fig2.update_layout(height=500, showlegend=False, title_text='Cinema Golden Ages Analysis')
    
    return fig1, fig2

# ============================================================================
# DASH APP SETUP
# ============================================================================

# Initialize Dash app with modern Bootstrap theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "IMDb Analytics Dashboard"

# Custom CSS for modern styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }
            body {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                margin: 0;
                padding: 0;
            }
            .stat-card {
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                padding: 24px;
                transition: all 0.3s ease;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
            .stat-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            }
            .stat-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 8px 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .stat-label {
                font-size: 0.875rem;
                font-weight: 500;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .dashboard-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 48px 32px;
                border-radius: 24px;
                margin-bottom: 32px;
                box-shadow: 0 20px 25px -5px rgba(102, 126, 234, 0.3);
            }
            .dashboard-title {
                font-size: 3rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .dashboard-subtitle {
                font-size: 1.125rem;
                font-weight: 400;
                opacity: 0.9;
                margin-top: 8px;
            }
            .nav-tabs .nav-link {
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                margin: 0 4px;
                font-weight: 500;
                transition: all 0.3s ease;
                background: white;
                color: #64748b;
            }
            .nav-tabs .nav-link:hover {
                background: #f1f5f9;
                transform: translateY(-2px);
            }
            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.3);
            }
            .tab-content {
                background: white;
                border-radius: 16px;
                padding: 24px;
                margin-top: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .graph-container {
                background: white;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 24px;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create visualizations
stats = create_overview_stats()
temp_fig1, temp_fig2, temp_fig3 = create_temporal_trends()
rating_fig1, rating_fig2, rating_fig3 = create_rating_distributions()
genre_fig1, genre_fig2, genre_fig3 = create_genre_analysis()
pop_fig1, pop_fig2, pop_fig3 = create_popularity_analysis()
runtime_fig1, runtime_fig2, runtime_fig3 = create_runtime_analysis()

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = dbc.Container([
    # Modern Header with Gradient
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🎬 IMDb Analytics Dashboard", className="dashboard-title"),
                html.P("Comprehensive insights from millions of titles across cinema history", 
                      className="dashboard-subtitle")
            ], className="dashboard-header")
        ])
    ]),
    
    # Modern Statistics Cards with Icons
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("📊 Total Titles", className="stat-label"),
                html.Div(stats['total_titles'], className="stat-value"),
                html.P("Movies, TV Shows & More", style={'color': '#94a3b8', 'fontSize': '0.875rem', 'margin': '4px 0 0 0'})
            ], className="stat-card")
        ], width=3, xs=12, sm=6, md=3),
        dbc.Col([
            html.Div([
                html.Div("⭐ Average Rating", className="stat-label"),
                html.Div(stats['avg_rating'], className="stat-value"),
                html.P("Across All Titles", style={'color': '#94a3b8', 'fontSize': '0.875rem', 'margin': '4px 0 0 0'})
            ], className="stat-card")
        ], width=3, xs=12, sm=6, md=3),
        dbc.Col([
            html.Div([
                html.Div("📅 Time Span", className="stat-label"),
                html.Div(stats['year_range'], className="stat-value", style={'fontSize': '1.75rem'}),
                html.P("Years of Cinema", style={'color': '#94a3b8', 'fontSize': '0.875rem', 'margin': '4px 0 0 0'})
            ], className="stat-card")
        ], width=3, xs=12, sm=6, md=3),
        dbc.Col([
            html.Div([
                html.Div("🎭 Unique Genres", className="stat-label"),
                html.Div(str(stats['unique_genres']), className="stat-value"),
                html.P("Genre Categories", style={'color': '#94a3b8', 'fontSize': '0.875rem', 'margin': '4px 0 0 0'})
            ], className="stat-card")
        ], width=3, xs=12, sm=6, md=3),
    ], style={'marginBottom': '32px'}),
    
    # Modern Navigation Tabs
    html.Div([
        dbc.Tabs([
            # TAB 1: Temporal Trends
            dbc.Tab([
                html.Div([
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=temp_fig1, config={'displayModeBar': False})], className="graph-container")], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=temp_fig2, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12),
                        dbc.Col([html.Div([dcc.Graph(figure=temp_fig3, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=create_decade_analysis(), config={'displayModeBar': False})], className="graph-container")], width=12)
                    ])
                ], className="tab-content")
            ], label="📈 Temporal Trends", tab_id="temporal"),
            
            # TAB 2: Rating Analysis
            dbc.Tab([
                html.Div([
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=rating_fig1, config={'displayModeBar': False})], className="graph-container")], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=rating_fig2, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12),
                        dbc.Col([html.Div([dcc.Graph(figure=rating_fig3, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=create_correlation_heatmap(), config={'displayModeBar': False})], className="graph-container")], width=12)
                    ])
                ], className="tab-content")
            ], label="⭐ Ratings", tab_id="ratings"),
            
            # TAB 3: Genre Analysis
            dbc.Tab([
                html.Div([
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=create_genre_evolution(), config={'displayModeBar': False})], className="graph-container")], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=genre_fig1, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12),
                        dbc.Col([html.Div([dcc.Graph(figure=genre_fig2, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=genre_fig3, config={'displayModeBar': False})], className="graph-container")], width=12)
                    ])
                ], className="tab-content")
            ], label="🎭 Genres", tab_id="genres"),
            
            # TAB 4: Runtime Analysis
            dbc.Tab([
                html.Div([
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=create_runtime_vs_rating(), config={'displayModeBar': False})], className="graph-container")], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=runtime_fig1, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12),
                        dbc.Col([html.Div([dcc.Graph(figure=runtime_fig2, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=runtime_fig3, config={'displayModeBar': False})], className="graph-container")], width=12)
                    ])
                ], className="tab-content")
            ], label="⏱️ Runtime", tab_id="runtime"),
            
            # TAB 5: Popularity Analysis
            dbc.Tab([
                html.Div([
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=pop_fig3, config={'displayModeBar': False})], className="graph-container")], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=pop_fig1, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12),
                        dbc.Col([html.Div([dcc.Graph(figure=pop_fig2, config={'displayModeBar': False})], className="graph-container")], width=6, md=6, sm=12)
                    ])
                ], className="tab-content")
            ], label="🔥 Popularity", tab_id="popularity"),
            
            # TAB 6: Advanced Insights
            dbc.Tab([
                html.Div([
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=create_advanced_insights()[0], config={'displayModeBar': False})], className="graph-container")], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([html.Div([dcc.Graph(figure=create_advanced_insights()[1], config={'displayModeBar': False})], className="graph-container")], width=12)
                    ])
                ], className="tab-content")
            ], label="🚀 Insights", tab_id="advanced"),
            
        ], id="tabs", active_tab="temporal")
    ], style={'marginBottom': '32px'}),
    
    # Modern Footer
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Hr(style={'margin': '48px 0 24px 0', 'opacity': '0.2'}),
                html.Div([
                    html.P([
                        "📊 Built with ",
                        html.Strong("Plotly Dash"),
                        " | Data from ",
                        html.A("IMDb Datasets", href="https://datasets.imdbws.com/", 
                              target="_blank",
                              style={'color': COLORS['primary'], 'textDecoration': 'none', 'fontWeight': '500'}),
                    ], style={'textAlign': 'center', 'color': '#64748b', 'fontSize': '0.875rem', 'marginBottom': '8px'}),
                    html.P([
                        "© 2024 IMDb Analytics Dashboard | ",
                        html.Span("Made with ❤️ for Data Visualization", style={'opacity': '0.7'})
                    ], style={'textAlign': 'center', 'color': '#94a3b8', 'fontSize': '0.75rem', 'marginBottom': '24px'})
                ])
            ])
        ])
    ])
    
], fluid=True, style={'maxWidth': '1400px', 'padding': '24px'})

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting IMDb Analytics Dashboard")
    print("="*80)
    print("\n✨ Features:")
    print("   • Modern, clean interface with gradient design")
    print("   • 21+ interactive visualizations across 6 categories")
    print("   • Responsive layout for all screen sizes")
    print("   • Professional color schemes and styling")
    print("\n📊 Dashboard Sections:")
    print("   ✓ Temporal Trends - Historical analysis over time")
    print("   ✓ Rating Analysis - Quality metrics and distributions")
    print("   ✓ Genre Analysis - Genre evolution and patterns")
    print("   ✓ Runtime Analysis - Optimal movie lengths")
    print("   ✓ Popularity Analysis - Audience engagement metrics")
    print("   ✓ Advanced Insights - Deep-dive analytics")
    print("\n🌐 Dashboard running at: http://127.0.0.1:8050/")
    print("   Press CTRL+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=False, port=8050, host='127.0.0.1')
