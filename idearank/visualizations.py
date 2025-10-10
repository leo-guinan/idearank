"""Visualization tools for IdeaRank analysis.

Creates charts and graphs to visualize content evolution over time.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Install with: pip install matplotlib")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not installed. Install with: pip install pandas")


class IdeaRankVisualizer:
    """Creates visualizations from IdeaRank database."""
    
    def __init__(self, db_path: str):
        """Initialize visualizer with database path.
        
        Args:
            db_path: Path to SQLite database
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib required for visualizations. "
                "Install with: pip install matplotlib"
            )
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        logger.info(f"Initialized visualizer for: {db_path}")
    
    def _fetch_data(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dicts."""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def plot_scores_over_time(
        self,
        content_source_id: Optional[str] = None,
        output_path: str = "scores_over_time.png",
        figsize: tuple = (12, 6),
    ) -> str:
        """Plot IdeaRank scores over time.
        
        Args:
            content_source_id: Filter to specific content source (None = all sources)
            output_path: Where to save the plot
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved plot
        """
        # Fetch data (JOIN with scores table)
        if content_source_id:
            query = """
                SELECT c.title, c.published_at, s.score as idearank_score, c.content_source_id
                FROM content_items c
                JOIN idearank_scores s ON c.id = s.content_item_id
                WHERE c.content_source_id = ?
                ORDER BY c.published_at
            """
            data = self._fetch_data(query, (content_source_id,))
        else:
            query = """
                SELECT c.title, c.published_at, s.score as idearank_score, c.content_source_id
                FROM content_items c
                JOIN idearank_scores s ON c.id = s.content_item_id
                ORDER BY c.published_at
            """
            data = self._fetch_data(query)
        
        if not data:
            raise ValueError("No scored content items found in database")
        
        # Group by content source
        sources = {}
        for row in data:
            source_id = row['content_source_id']
            if source_id not in sources:
                sources[source_id] = {'dates': [], 'scores': []}
            
            # Parse date
            try:
                date = datetime.fromisoformat(row['published_at'].replace('Z', '+00:00'))
            except:
                continue
            
            sources[source_id]['dates'].append(date)
            sources[source_id]['scores'].append(row['idearank_score'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for source_id, source_data in sources.items():
            # Get source name (shorten if needed)
            source_name = source_id[:30] + "..." if len(source_id) > 30 else source_id
            ax.plot(
                source_data['dates'],
                source_data['scores'],
                marker='o',
                label=source_name,
                linewidth=2,
                markersize=6,
            )
        
        ax.set_xlabel('Publication Date', fontsize=12)
        ax.set_ylabel('IdeaRank Score', fontsize=12)
        ax.set_title('IdeaRank Scores Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
        return output_path
    
    def plot_factor_breakdown(
        self,
        content_source_id: Optional[str] = None,
        output_path: str = "factor_breakdown.png",
        figsize: tuple = (14, 8),
    ) -> str:
        """Plot all five factors over time.
        
        Args:
            content_source_id: Filter to specific content source
            output_path: Where to save the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        # Fetch data with all factors (JOIN with scores)
        if content_source_id:
            query = """
                SELECT 
                    c.title, c.published_at,
                    s.uniqueness_score, s.cohesion_score, s.learning_score,
                    s.quality_score, s.trust_score
                FROM content_items c
                JOIN idearank_scores s ON c.id = s.content_item_id
                WHERE c.content_source_id = ?
                ORDER BY c.published_at
            """
            data = self._fetch_data(query, (content_source_id,))
        else:
            query = """
                SELECT 
                    c.title, c.published_at,
                    s.uniqueness_score, s.cohesion_score, s.learning_score,
                    s.quality_score, s.trust_score
                FROM content_items c
                JOIN idearank_scores s ON c.id = s.content_item_id
                ORDER BY c.published_at
            """
            data = self._fetch_data(query)
        
        if not data:
            raise ValueError("No scored content items found")
        
        # Parse data
        dates = []
        factors = {
            'Uniqueness (U)': [],
            'Cohesion (C)': [],
            'Learning (L)': [],
            'Quality (Q)': [],
            'Trust (T)': [],
        }
        
        for row in data:
            try:
                date = datetime.fromisoformat(row['published_at'].replace('Z', '+00:00'))
                dates.append(date)
                factors['Uniqueness (U)'].append(row['uniqueness_score'])
                factors['Cohesion (C)'].append(row['cohesion_score'])
                factors['Learning (L)'].append(row['learning_score'])
                factors['Quality (Q)'].append(row['quality_score'])
                factors['Trust (T)'].append(row['trust_score'])
            except:
                continue
        
        # Create plot with subplots
        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
        fig.suptitle('IdeaRank Factor Evolution Over Time', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for (factor_name, scores), ax, color in zip(factors.items(), axes, colors):
            ax.plot(dates, scores, marker='o', color=color, linewidth=2, markersize=5)
            ax.set_ylabel(factor_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
        
        # Format x-axis on bottom plot only
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        axes[-1].set_xlabel('Publication Date', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
        return output_path
    
    def plot_learning_trajectory(
        self,
        content_source_id: str,
        output_path: str = "learning_trajectory.png",
        figsize: tuple = (12, 6),
    ) -> str:
        """Plot learning score trajectory for a content source.
        
        Shows how learning progresses over time (should trend upward).
        
        Args:
            content_source_id: Content source to analyze
            output_path: Where to save the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        query = """
            SELECT c.title, c.published_at, s.learning_score
            FROM content_items c
            JOIN idearank_scores s ON c.id = s.content_item_id
            WHERE c.content_source_id = ?
            ORDER BY c.published_at
        """
        data = self._fetch_data(query, (content_source_id,))
        
        if not data:
            raise ValueError(f"No data found for content source: {content_source_id}")
        
        dates = []
        scores = []
        titles = []
        
        for row in data:
            try:
                date = datetime.fromisoformat(row['published_at'].replace('Z', '+00:00'))
                dates.append(date)
                scores.append(row['learning_score'])
                titles.append(row['title'][:40])
            except:
                continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot learning scores
        ax.plot(dates, scores, marker='o', color='#45B7D1', linewidth=2, markersize=8, label='Learning Score')
        
        # Add trend line
        if PANDAS_AVAILABLE:
            import pandas as pd
            df = pd.DataFrame({'date': dates, 'score': scores})
            df['date_num'] = mdates.date2num(df['date'])
            z = np.polyfit(df['date_num'], df['score'], 1)
            p = np.poly1d(z)
            ax.plot(dates, p(df['date_num']), "--", color='red', alpha=0.7, linewidth=2, label='Trend')
        
        ax.set_xlabel('Publication Date', fontsize=12)
        ax.set_ylabel('Learning Score', fontsize=12)
        ax.set_title(f'Learning Trajectory: {content_source_id[:50]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
        return output_path
    
    def plot_uniqueness_vs_cohesion(
        self,
        content_source_id: Optional[str] = None,
        output_path: str = "uniqueness_vs_cohesion.png",
        figsize: tuple = (10, 10),
    ) -> str:
        """Scatter plot of uniqueness vs cohesion.
        
        Helps identify content that is:
        - High U, High C: Novel but on-brand
        - High U, Low C: Exploratory, off-brand
        - Low U, High C: Repetitive but on-brand
        - Low U, Low C: Repetitive and off-brand
        
        Args:
            content_source_id: Filter to specific content source
            output_path: Where to save the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        if content_source_id:
            query = """
                SELECT c.title, s.uniqueness_score, s.cohesion_score, c.content_source_id
                FROM content_items c
                JOIN idearank_scores s ON c.id = s.content_item_id
                WHERE c.content_source_id = ?
            """
            data = self._fetch_data(query, (content_source_id,))
        else:
            query = """
                SELECT c.title, s.uniqueness_score, s.cohesion_score, c.content_source_id
                FROM content_items c
                JOIN idearank_scores s ON c.id = s.content_item_id
            """
            data = self._fetch_data(query)
        
        if not data:
            raise ValueError("No data found")
        
        # Group by content source
        sources = {}
        for row in data:
            source_id = row['content_source_id']
            if source_id not in sources:
                sources[source_id] = {'u': [], 'c': [], 'titles': []}
            
            sources[source_id]['u'].append(row['uniqueness_score'])
            sources[source_id]['c'].append(row['cohesion_score'])
            sources[source_id]['titles'].append(row['title'][:30])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#96CEB4']
        
        for i, (source_id, source_data) in enumerate(sources.items()):
            source_name = source_id[:25] + "..." if len(source_id) > 25 else source_id
            color = colors[i % len(colors)]
            
            ax.scatter(
                source_data['u'],
                source_data['c'],
                alpha=0.6,
                s=100,
                color=color,
                label=source_name,
                edgecolors='black',
                linewidth=0.5,
            )
        
        # Add quadrant lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add quadrant labels
        ax.text(0.75, 0.75, 'Novel &\nOn-Brand', ha='center', va='center', fontsize=10, alpha=0.5)
        ax.text(0.25, 0.75, 'Repetitive &\nOn-Brand', ha='center', va='center', fontsize=10, alpha=0.5)
        ax.text(0.75, 0.25, 'Novel &\nOff-Brand', ha='center', va='center', fontsize=10, alpha=0.5)
        ax.text(0.25, 0.25, 'Repetitive &\nOff-Brand', ha='center', va='center', fontsize=10, alpha=0.5)
        
        ax.set_xlabel('Uniqueness (U)', fontsize=12)
        ax.set_ylabel('Cohesion (C)', fontsize=12)
        ax.set_title('Uniqueness vs Cohesion Analysis', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
        return output_path
    
    def plot_source_comparison(
        self,
        output_path: str = "source_comparison.png",
        figsize: tuple = (12, 8),
    ) -> str:
        """Compare average scores across content sources.
        
        Args:
            output_path: Where to save the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        query = """
            SELECT 
                c.content_source_id,
                AVG(s.score) as avg_score,
                AVG(s.uniqueness_score) as avg_u,
                AVG(s.cohesion_score) as avg_c,
                AVG(s.learning_score) as avg_l,
                AVG(s.quality_score) as avg_q,
                AVG(s.trust_score) as avg_t,
                COUNT(*) as count
            FROM content_items c
            JOIN idearank_scores s ON c.id = s.content_item_id
            GROUP BY c.content_source_id
            ORDER BY avg_score DESC
        """
        data = self._fetch_data(query)
        
        if not data:
            raise ValueError("No data found")
        
        # Prepare data
        sources = [row['content_source_id'][:20] for row in data]
        factor_data = {
            'U': [row['avg_u'] for row in data],
            'C': [row['avg_c'] for row in data],
            'L': [row['avg_l'] for row in data],
            'Q': [row['avg_q'] for row in data],
            'T': [row['avg_t'] for row in data],
        }
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        x = range(len(sources))
        width = 0.6
        
        colors = {
            'U': '#FF6B6B',
            'C': '#4ECDC4',
            'L': '#45B7D1',
            'Q': '#FFA07A',
            'T': '#98D8C8',
        }
        
        # Create grouped bars
        bar_width = width / 5
        for i, (factor, scores) in enumerate(factor_data.items()):
            offset = (i - 2) * bar_width
            ax.bar(
                [pos + offset for pos in x],
                scores,
                bar_width,
                label=factor,
                color=colors[factor],
                alpha=0.8,
            )
        
        ax.set_xlabel('Content Source', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('Content Source Comparison - Factor Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.legend(title='Factors', loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
        return output_path
    
    def plot_distribution(
        self,
        content_source_id: Optional[str] = None,
        output_path: str = "score_distribution.png",
        figsize: tuple = (12, 6),
    ) -> str:
        """Plot distribution of IdeaRank scores (histogram).
        
        Args:
            content_source_id: Filter to specific content source
            output_path: Where to save the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        if content_source_id:
            query = """
                SELECT s.score as idearank_score
                FROM content_items c
                JOIN idearank_scores s ON c.id = s.content_item_id
                WHERE c.content_source_id = ?
            """
            data = self._fetch_data(query, (content_source_id,))
        else:
            query = """
                SELECT s.score as idearank_score, c.content_source_id
                FROM content_items c
                JOIN idearank_scores s ON c.id = s.content_item_id
            """
            data = self._fetch_data(query)
        
        if not data:
            raise ValueError("No data found")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if content_source_id:
            # Single source histogram
            scores = [row['idearank_score'] for row in data]
            ax.hist(scores, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
            ax.set_title(f'Score Distribution: {content_source_id[:40]}', fontsize=14, fontweight='bold')
        else:
            # Multiple sources
            sources = {}
            for row in data:
                source_id = row['content_source_id']
                if source_id not in sources:
                    sources[source_id] = []
                sources[source_id].append(row['idearank_score'])
            
            colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            for i, (source_id, scores) in enumerate(sources.items()):
                source_name = source_id[:20]
                ax.hist(
                    scores,
                    bins=15,
                    alpha=0.5,
                    label=source_name,
                    color=colors_list[i % len(colors_list)],
                    edgecolor='black',
                    linewidth=0.5,
                )
            
            ax.legend(loc='best')
            ax.set_title('Score Distribution by Content Source', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('IdeaRank Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {output_path}")
        return output_path
    
    def create_dashboard(
        self,
        content_source_id: Optional[str] = None,
        output_dir: str = "idearank_viz",
    ) -> List[str]:
        """Create a complete dashboard with all visualizations.
        
        Args:
            content_source_id: Filter to specific content source (None = all)
            output_dir: Directory to save plots
            
        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        plots = []
        
        logger.info("Creating dashboard...")
        
        try:
            plots.append(self.plot_scores_over_time(
                content_source_id=content_source_id,
                output_path=str(output_path / "scores_over_time.png")
            ))
        except Exception as e:
            logger.error(f"Failed to create scores plot: {e}")
        
        try:
            plots.append(self.plot_factor_breakdown(
                content_source_id=content_source_id,
                output_path=str(output_path / "factor_breakdown.png")
            ))
        except Exception as e:
            logger.error(f"Failed to create factor breakdown: {e}")
        
        try:
            plots.append(self.plot_uniqueness_vs_cohesion(
                content_source_id=content_source_id,
                output_path=str(output_path / "uniqueness_vs_cohesion.png")
            ))
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {e}")
        
        try:
            plots.append(self.plot_distribution(
                content_source_id=content_source_id,
                output_path=str(output_path / "score_distribution.png")
            ))
        except Exception as e:
            logger.error(f"Failed to create distribution plot: {e}")
        
        logger.info(f"Dashboard created: {len(plots)} plots in {output_dir}/")
        return plots
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

