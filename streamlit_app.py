"""
IdeaRank Streamlit Dashboard

Interactive web interface for exploring IdeaRank scores and insights.
"""

import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="IdeaRank Dashboard",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_databases():
    """Find all IdeaRank databases in the current directory."""
    db_files = list(Path.cwd().glob("*.db"))
    return [str(db) for db in db_files if db.stat().st_size > 0]


@st.cache_data
def load_data(db_path: str):
    """Load all data from the database."""
    conn = sqlite3.connect(db_path)
    
    # Load content sources
    sources_df = pd.read_sql_query("""
        SELECT 
            id,
            name,
            description,
            created_at,
            subscriber_count,
            total_views
        FROM content_sources
    """, conn)
    
    # Load content items with scores
    items_df = pd.read_sql_query("""
        SELECT 
            c.id,
            c.content_source_id,
            c.title,
            c.description,
            c.published_at,
            c.view_count,
            c.impression_count,
            c.content_duration,
            s.score,
            s.uniqueness_score,
            s.cohesion_score,
            s.learning_score,
            s.quality_score,
            s.trust_score,
            s.density_score,
            s.passes_gates,
            s.uniqueness_components,
            s.cohesion_components,
            s.learning_components,
            s.quality_components,
            s.trust_components,
            s.density_components
        FROM content_items c
        LEFT JOIN idearank_scores s ON c.id = s.content_item_id
        WHERE s.score IS NOT NULL
    """, conn)
    
    # Parse JSON components
    for col in ['uniqueness_components', 'cohesion_components', 'learning_components', 
                'quality_components', 'trust_components', 'density_components']:
        if col in items_df.columns:
            items_df[col] = items_df[col].apply(lambda x: json.loads(x) if x else {})
    
    # Convert dates
    items_df['published_at'] = pd.to_datetime(items_df['published_at'])
    sources_df['created_at'] = pd.to_datetime(sources_df['created_at'])
    
    conn.close()
    
    return sources_df, items_df


def main():
    """Main Streamlit app."""
    
    # Title
    st.title("💡 IdeaRank Dashboard")
    st.markdown("**Interactive analysis of your content's IdeaRank scores**")
    
    # Sidebar - Database selection
    st.sidebar.header("📊 Database")
    databases = load_databases()
    
    if not databases:
        st.error("No IdeaRank databases found in current directory.")
        st.info("Run `idearank process-all` first to generate data.")
        return
    
    db_path = st.sidebar.selectbox(
        "Select database:",
        databases,
        format_func=lambda x: Path(x).name
    )
    
    # Load data
    try:
        sources_df, items_df = load_data(db_path)
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return
    
    if len(items_df) == 0:
        st.warning("No scored content found in this database.")
        return
    
    # Sidebar - Source filter
    st.sidebar.header("🎯 Filters")
    
    all_sources = ["All Sources"] + sources_df['name'].tolist()
    selected_source = st.sidebar.selectbox("Content Source:", all_sources)
    
    # Filter data
    if selected_source != "All Sources":
        source_id = sources_df[sources_df['name'] == selected_source]['id'].iloc[0]
        filtered_items = items_df[items_df['content_source_id'] == source_id]
    else:
        filtered_items = items_df
        source_id = None
    
    # Date range filter
    min_date = filtered_items['published_at'].min()
    max_date = filtered_items['published_at'].max()
    
    date_range = st.sidebar.date_input(
        "Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        filtered_items = filtered_items[
            (filtered_items['published_at'] >= pd.Timestamp(date_range[0])) &
            (filtered_items['published_at'] <= pd.Timestamp(date_range[1]))
        ]
    
    # Main content area
    tabs = st.tabs([
        "📈 Overview",
        "🎯 Top Content",
        "📊 Factor Analysis",
        "🔍 Content Explorer",
        "🏆 Source Comparison",
        "📉 Trends"
    ])
    
    # TAB 1: Overview
    with tabs[0]:
        show_overview(sources_df, filtered_items, selected_source)
    
    # TAB 2: Top Content
    with tabs[1]:
        show_top_content(filtered_items)
    
    # TAB 3: Factor Analysis
    with tabs[2]:
        show_factor_analysis(filtered_items)
    
    # TAB 4: Content Explorer
    with tabs[3]:
        show_content_explorer(filtered_items)
    
    # TAB 5: Source Comparison
    with tabs[4]:
        show_source_comparison(sources_df, items_df)
    
    # TAB 6: Trends
    with tabs[5]:
        show_trends(filtered_items, selected_source)


def show_overview(sources_df, items_df, selected_source):
    """Show overview metrics."""
    st.header("📈 Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Content", len(items_df))
    
    with col2:
        st.metric("Avg IdeaRank", f"{items_df['score'].mean():.3f}")
    
    with col3:
        st.metric("Top Score", f"{items_df['score'].max():.3f}")
    
    with col4:
        high_quality = len(items_df[items_df['score'] > 0.7])
        st.metric("High Quality (>0.7)", high_quality)
    
    with col5:
        passed_gates = len(items_df[items_df['passes_gates'] == 1])
        st.metric("Passed Gates", passed_gates)
    
    st.markdown("---")
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        fig = px.histogram(
            items_df,
            x='score',
            nbins=30,
            title="IdeaRank Score Distribution",
            labels={'score': 'IdeaRank Score', 'count': 'Number of Items'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Factor Averages")
        factor_avgs = {
            'Uniqueness': items_df['uniqueness_score'].mean(),
            'Cohesion': items_df['cohesion_score'].mean(),
            'Learning': items_df['learning_score'].mean(),
            'Quality': items_df['quality_score'].mean(),
            'Trust': items_df['trust_score'].mean(),
            'Density': items_df['density_score'].mean(),
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(factor_avgs.keys()),
                y=list(factor_avgs.values()),
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
            )
        ])
        fig.update_layout(
            title="Average Factor Scores",
            yaxis_title="Score",
            xaxis_title="Factor",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)


def show_top_content(items_df):
    """Show top-performing content."""
    st.header("🎯 Top Content")
    
    # Top by overall score
    st.subheader("Top 20 by IdeaRank Score")
    
    top_items = items_df.nlargest(20, 'score')[
        ['title', 'score', 'uniqueness_score', 'cohesion_score', 
         'learning_score', 'quality_score', 'trust_score', 'density_score', 
         'published_at', 'view_count']
    ].copy()
    
    # Format for display
    top_items['published_at'] = top_items['published_at'].dt.strftime('%Y-%m-%d')
    top_items = top_items.round(3)
    
    st.dataframe(
        top_items,
        use_container_width=True,
        hide_index=True,
        column_config={
            'title': st.column_config.TextColumn('Title', width='large'),
            'score': st.column_config.NumberColumn('Score', format="%.3f"),
            'uniqueness_score': st.column_config.NumberColumn('U', format="%.2f"),
            'cohesion_score': st.column_config.NumberColumn('C', format="%.2f"),
            'learning_score': st.column_config.NumberColumn('L', format="%.2f"),
            'quality_score': st.column_config.NumberColumn('Q', format="%.2f"),
            'trust_score': st.column_config.NumberColumn('T', format="%.2f"),
            'density_score': st.column_config.NumberColumn('D', format="%.2f"),
            'published_at': 'Published',
            'view_count': st.column_config.NumberColumn('Views', format="%d"),
        }
    )
    
    # Top by each factor
    st.markdown("---")
    st.subheader("Top Performers by Factor")
    
    factors = {
        'Uniqueness': 'uniqueness_score',
        'Cohesion': 'cohesion_score',
        'Learning': 'learning_score',
        'Quality': 'quality_score',
        'Trust': 'trust_score',
        'Density': 'density_score',
    }
    
    cols = st.columns(3)
    for idx, (factor_name, col_name) in enumerate(factors.items()):
        with cols[idx % 3]:
            st.write(f"**{factor_name}**")
            top_5 = items_df.nlargest(5, col_name)[['title', col_name]]
            for _, row in top_5.iterrows():
                st.write(f"• {row['title'][:40]}... ({row[col_name]:.2f})")


def show_factor_analysis(items_df):
    """Show detailed factor analysis."""
    st.header("📊 Factor Analysis")
    
    # Factor correlation heatmap
    st.subheader("Factor Correlations")
    
    factor_cols = ['uniqueness_score', 'cohesion_score', 'learning_score', 
                   'quality_score', 'trust_score', 'density_score']
    corr_matrix = items_df[factor_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Factor", y="Factor", color="Correlation"),
        x=['U', 'C', 'L', 'Q', 'T', 'D'],
        y=['U', 'C', 'L', 'Q', 'T', 'D'],
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        title="Factor Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Factor component details
    st.markdown("---")
    st.subheader("Component Breakdown")
    
    # Select a content item to inspect
    selected_title = st.selectbox(
        "Select content to inspect:",
        items_df.nlargest(20, 'score')['title'].tolist()
    )
    
    selected_item = items_df[items_df['title'] == selected_title].iloc[0]
    
    # Show factor components
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Uniqueness Components**")
        if selected_item['uniqueness_components']:
            for key, value in selected_item['uniqueness_components'].items():
                st.write(f"• {key}: {value:.4f}" if isinstance(value, (int, float)) else f"• {key}: {value}")
        
        st.write("**Cohesion Components**")
        if selected_item['cohesion_components']:
            for key, value in selected_item['cohesion_components'].items():
                st.write(f"• {key}: {value:.4f}" if isinstance(value, (int, float)) else f"• {key}: {value}")
        
        st.write("**Learning Components**")
        if selected_item['learning_components']:
            for key, value in selected_item['learning_components'].items():
                st.write(f"• {key}: {value:.4f}" if isinstance(value, (int, float)) else f"• {key}: {value}")
    
    with col2:
        st.write("**Quality Components**")
        if selected_item['quality_components']:
            for key, value in selected_item['quality_components'].items():
                st.write(f"• {key}: {value:.4f}" if isinstance(value, (int, float)) else f"• {key}: {value}")
        
        st.write("**Trust Components**")
        if selected_item['trust_components']:
            for key, value in selected_item['trust_components'].items():
                st.write(f"• {key}: {value:.4f}" if isinstance(value, (int, float)) else f"• {key}: {value}")
        
        st.write("**Density Components**")
        if selected_item['density_components']:
            for key, value in selected_item['density_components'].items():
                st.write(f"• {key}: {value:.4f}" if isinstance(value, (int, float)) else f"• {key}: {value}")


def show_content_explorer(items_df):
    """Interactive content explorer."""
    st.header("🔍 Content Explorer")
    
    # Search
    search = st.text_input("🔎 Search content:", placeholder="Enter keywords...")
    
    if search:
        mask = items_df['title'].str.contains(search, case=False, na=False) | \
               items_df['description'].str.contains(search, case=False, na=False)
        display_items = items_df[mask]
    else:
        display_items = items_df
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        ['score', 'uniqueness_score', 'cohesion_score', 'learning_score', 
         'quality_score', 'trust_score', 'density_score', 'published_at', 'view_count']
    )
    
    ascending = st.checkbox("Ascending", value=False)
    
    display_items = display_items.sort_values(by=sort_by, ascending=ascending)
    
    # Display
    st.write(f"Showing {len(display_items)} items")
    
    for idx, row in display_items.head(50).iterrows():
        with st.expander(f"**{row['title']}** (Score: {row['score']:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Published:** {row['published_at'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Description:** {row['description'][:200]}...")
                
                # Factor radar chart
                factors = {
                    'U': row['uniqueness_score'],
                    'C': row['cohesion_score'],
                    'L': row['learning_score'],
                    'Q': row['quality_score'],
                    'T': row['trust_score'],
                    'D': row['density_score'],
                }
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(factors.values()),
                    theta=list(factors.keys()),
                    fill='toself',
                    name='Factors'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Views", f"{row['view_count']:,}")
                st.metric("Impressions", f"{row['impression_count']:,}")
                st.metric("Duration", f"{row['content_duration']/60:.1f} min")
                st.metric("Gates Passed", "✅" if row['passes_gates'] else "❌")


def show_source_comparison(sources_df, items_df):
    """Compare different sources."""
    st.header("🏆 Source Comparison")
    
    # Calculate source-level metrics
    source_metrics = []
    
    for _, source in sources_df.iterrows():
        source_items = items_df[items_df['content_source_id'] == source['id']]
        
        if len(source_items) > 0:
            source_metrics.append({
                'Source': source['name'],
                'Items': len(source_items),
                'Avg Score': source_items['score'].mean(),
                'Avg U': source_items['uniqueness_score'].mean(),
                'Avg C': source_items['cohesion_score'].mean(),
                'Avg L': source_items['learning_score'].mean(),
                'Avg Q': source_items['quality_score'].mean(),
                'Avg T': source_items['trust_score'].mean(),
                'Avg D': source_items['density_score'].mean(),
                'Total Views': source_items['view_count'].sum(),
            })
    
    metrics_df = pd.DataFrame(source_metrics)
    
    if len(metrics_df) > 0:
        # Overall comparison
        st.subheader("Source Performance")
        
        fig = px.bar(
            metrics_df.sort_values('Avg Score', ascending=False),
            x='Source',
            y='Avg Score',
            title="Average IdeaRank Score by Source",
            color='Avg Score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor comparison
        st.subheader("Factor Comparison Across Sources")
        
        factor_cols = ['Avg U', 'Avg C', 'Avg L', 'Avg Q', 'Avg T', 'Avg D']
        
        fig = go.Figure()
        for col in factor_cols:
            fig.add_trace(go.Bar(
                name=col.replace('Avg ', ''),
                x=metrics_df['Source'],
                y=metrics_df[col]
            ))
        
        fig.update_layout(
            barmode='group',
            title="Factor Scores by Source",
            yaxis_title="Average Score",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Detailed Metrics")
        st.dataframe(
            metrics_df.round(3),
            use_container_width=True,
            hide_index=True
        )


def show_trends(items_df, selected_source):
    """Show trends over time."""
    st.header("📉 Trends Over Time")
    
    if len(items_df) < 2:
        st.info("Need at least 2 content items to show trends.")
        return
    
    # Scores over time
    st.subheader("IdeaRank Score Timeline")
    
    fig = px.scatter(
        items_df.sort_values('published_at'),
        x='published_at',
        y='score',
        hover_data=['title', 'view_count'],
        title=f"IdeaRank Scores Over Time - {selected_source}",
        trendline="lowess"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)
    
    # All factors over time
    st.subheader("All Factors Timeline")
    
    # Prepare data for multi-line plot
    timeline_data = items_df.sort_values('published_at').copy()
    
    fig = go.Figure()
    
    factors = {
        'Uniqueness': 'uniqueness_score',
        'Cohesion': 'cohesion_score',
        'Learning': 'learning_score',
        'Quality': 'quality_score',
        'Trust': 'trust_score',
        'Density': 'density_score',
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for idx, (name, col) in enumerate(factors.items()):
        fig.add_trace(go.Scatter(
            x=timeline_data['published_at'],
            y=timeline_data[col],
            mode='lines+markers',
            name=name,
            line=dict(color=colors[idx], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Factor Evolution Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly aggregation
    st.subheader("Monthly Performance")
    
    monthly = items_df.copy()
    monthly['month'] = monthly['published_at'].dt.to_period('M').astype(str)
    
    monthly_agg = monthly.groupby('month').agg({
        'score': 'mean',
        'uniqueness_score': 'mean',
        'cohesion_score': 'mean',
        'learning_score': 'mean',
        'quality_score': 'mean',
        'trust_score': 'mean',
        'density_score': 'mean',
        'title': 'count'
    }).reset_index()
    monthly_agg.columns = ['Month', 'Avg Score', 'Avg U', 'Avg C', 'Avg L', 'Avg Q', 'Avg T', 'Avg D', 'Count']
    
    fig = px.line(
        monthly_agg,
        x='Month',
        y='Avg Score',
        markers=True,
        title="Average IdeaRank Score by Month",
        text='Count'
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

