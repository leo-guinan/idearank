"""Command-line interface for IdeaRank.

Provides commands to configure API keys and process YouTube channels.
"""

import click
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

logger = logging.getLogger(__name__)

from idearank.cli_config import CLIConfig
from idearank.content_sources import SourcesConfig, ContentSource
from idearank.config import IdeaRankConfig
from idearank.pipeline import IdeaRankPipeline
from idearank.pipelines.youtube_pipeline import YouTubePipeline
from idearank.pipelines.ghost_pipeline import GhostPipeline
from idearank.integrations.youtube import YouTubeClient
from idearank.integrations.ghost import GhostClient
from idearank.integrations.ghost_export import GhostExportClient
from idearank.integrations.storage import SQLiteStorage
from idearank.providers.chroma import ChromaProvider
from idearank.providers.embeddings import OpenAIEmbeddingProvider
from idearank.providers.topics import DummyTopicModelProvider, LDATopicModelProvider

console = Console()


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


@click.group()
@click.version_option(version="2.0.0", prog_name="idearank")
def main():
    """IdeaRank - Multi-factor ranking for content.
    
    A PageRank replacement for ideas that analyzes uniqueness, cohesion,
    learning trajectory, quality, and trust signals across any content source.
    """
    pass


@main.group()
def config():
    """Manage IdeaRank configuration and API keys."""
    pass


@main.group()
def source():
    """Manage content sources (YouTube channels, Ghost blogs, etc.)."""
    pass


@main.group()
def viz():
    """Visualize IdeaRank data over time."""
    pass


@config.command("set-youtube-key")
@click.argument("api_key")
def set_youtube_key(api_key: str):
    """Set YouTube Data API key.
    
    Get your API key from: https://console.cloud.google.com/apis/credentials
    """
    cfg = CLIConfig()
    cfg.set_youtube_api_key(api_key)
    
    rprint("[green]✓[/green] YouTube API key saved")


@config.command("set-gladia-key")
@click.argument("api_key")
def set_gladia_key(api_key: str):
    """Set Gladia API key for transcription.
    
    Get your API key from: https://gladia.io
    """
    cfg = CLIConfig()
    cfg.set_gladia_api_key(api_key)
    
    rprint("[green]✓[/green] Gladia API key saved")


@config.command("set-chroma-mode")
@click.argument("mode", type=click.Choice(["local", "cloud"]))
def set_chroma_mode(mode: str):
    """Set Chroma storage mode (local or cloud)."""
    cfg = CLIConfig()
    cfg.set_chroma_mode(mode)
    
    rprint(f"[green]✓[/green] Chroma mode set to: {mode}")


@config.command("set-chroma-cloud")
@click.option("--api-key", required=True, help="Chroma Cloud API key")
@click.option("--tenant", default="default_tenant", help="Chroma Cloud tenant")
@click.option("--database", default="default_database", help="Chroma Cloud database")
def set_chroma_cloud(api_key: str, tenant: str, database: str):
    """Configure Chroma Cloud credentials."""
    cfg = CLIConfig()
    cfg.set_chroma_cloud(api_key, tenant, database)
    
    rprint("[green]✓[/green] Chroma Cloud credentials saved")


@config.command("show")
def show_config():
    """Show current configuration."""
    cfg = CLIConfig()
    config_data = cfg.show()
    
    table = Table(title="IdeaRank Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("YouTube API Key", config_data.get("youtube_api_key") or "[red]Not set[/red]")
    table.add_row("Gladia API Key", config_data.get("gladia_api_key") or "[yellow]Not set (optional)[/yellow]")
    table.add_row("Chroma Mode", config_data.get("chroma_mode", "local"))
    
    if config_data.get("chroma_mode") == "cloud":
        table.add_row("Chroma Cloud API Key", config_data.get("chroma_cloud_api_key") or "[red]Not set[/red]")
        table.add_row("Chroma Cloud Tenant", config_data.get("chroma_cloud_tenant") or "[yellow]default_tenant[/yellow]")
        table.add_row("Chroma Cloud Database", config_data.get("chroma_cloud_database") or "[yellow]default_database[/yellow]")
    
    console.print(table)
    
    # Check if properly configured
    is_configured, missing = cfg.is_configured()
    if not is_configured:
        rprint(f"\n[yellow]⚠[/yellow]  Missing configuration: {', '.join(missing)}")
        rprint("\nRun [cyan]idearank config --help[/cyan] to see configuration commands")


@config.command("clear")
@click.confirmation_option(prompt="Are you sure you want to clear all configuration?")
def clear_config():
    """Clear all configuration (requires confirmation)."""
    cfg = CLIConfig()
    cfg.clear()
    
    rprint("[green]✓[/green] Configuration cleared")


@source.command("add")
@click.argument("url_or_file")
@click.option("--name", help="Display name for this source")
@click.option("--max-items", default=50, help="Maximum items to process")
@click.option("--filter", "filter_query", help="Filter query (e.g., 'tag:python')")
@click.option("--type", "content_type", type=click.Choice(["youtube", "ghost_export", "ghost_api", "twitter", "auto"]), default="auto", help="Content type (auto-detect by default)")
def add_source(url_or_file: str, name: Optional[str], max_items: int, filter_query: Optional[str], content_type: str):
    """Add a content source to process.
    
    Examples:
        idearank source add https://youtube.com/@channel
        idearank source add my-blog.ghost.json
        idearank source add https://blog.example.com --type ghost_api
        idearank source add @username
    """
    sources_config = SourcesConfig()
    
    # Special handling for Twitter sources
    if content_type == "twitter" or (content_type == "auto" and ContentSource._is_twitter_handle(url_or_file)):
        # Check if archive is available
        from idearank.integrations.twitter import check_twitter_archive
        
        username = url_or_file.lstrip('@').strip()
        availability = check_twitter_archive(username)
        
        if availability['available'] is True:
            click.echo(f"✓ Twitter archive found for @{username}")
            click.echo(f"  Archive URL: {availability['archive_url']}")
        elif availability['available'] is False:
            click.echo(f"✗ No Twitter archive found for @{username}")
            click.echo(f"  Upload your archive at: {availability.get('upload_url', 'https://community-archive.org/upload')}")
        else:
            click.echo(f"⚠️  Can't verify Twitter archive availability for @{username}")
            click.echo(f"  Check manually at: {availability.get('archive_url', 'https://community-archive.org/search')}")
            click.echo(f"  User directory: {availability.get('user_directory_url', 'https://community-archive.org/user-dir')}")
            click.echo(f"  Upload at: {availability.get('upload_url', 'https://community-archive.org/upload')}")
            click.echo("  Note: Community Archive doesn't provide a public API for checking availability.")
        
        click.echo("  You can still add the source - it will be processed when an archive becomes available.")
    
    # Create source with auto-detection
    source = ContentSource.create(
        url_or_path=url_or_file,
        name=name,
        max_items=max_items,
        filter_query=filter_query,
    )
    
    # Override type if specified
    if content_type != "auto":
        source.type = content_type
    
    # Check if type was detected
    if source.type == "unknown":
        rprint(f"[yellow]⚠[/yellow]  Could not auto-detect content type for: {url_or_file}")
        rprint("    Use --type to specify manually (youtube, ghost_export, ghost_api, twitter)")
        raise click.Abort()
    
    # Add source
    try:
        sources_config.add_source(source)
        rprint(f"[green]✓[/green] Added source: {source.name}")
        rprint(f"    Type: {source.type}")
        rprint(f"    ID: {source.id}")
        rprint(f"    Max items: {source.max_items}")
        if source.filter_query:
            rprint(f"    Filter: {source.filter_query}")
    except ValueError as e:
        rprint(f"[red]✗[/red] {e}")
        raise click.Abort()


@source.command("list")
@click.option("--enabled-only", is_flag=True, help="Show only enabled sources")
def list_sources(enabled_only: bool):
    """List all content sources."""
    sources_config = SourcesConfig()
    sources = sources_config.list_sources(enabled_only=enabled_only)
    
    if not sources:
        rprint("[yellow]No content sources configured.[/yellow]")
        rprint("\nAdd sources with: [cyan]idearank source add URL_OR_FILE[/cyan]")
        return
    
    table = Table(title="Content Sources")
    table.add_column("ID", style="cyan", width=20)
    table.add_column("Name", style="white", width=25)
    table.add_column("Type", style="green", width=12)
    table.add_column("Items", justify="right", width=6)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Last Processed", style="dim", width=20)
    
    for source in sources:
        status = "[green]✓[/green]" if source.enabled else "[dim]✗[/dim]"
        last_proc = source.last_processed[:19] if source.last_processed else "Never"
        
        table.add_row(
            source.id,
            source.name or source.url_or_path[:25],
            source.type,
            str(source.max_items),
            status,
            last_proc,
        )
    
    console.print(table)
    rprint(f"\nTotal sources: {len(sources)}")
    if enabled_only:
        rprint(f"(Showing enabled only. Use without --enabled-only to see all)")


@source.command("remove")
@click.argument("source_id")
def remove_source(source_id: str):
    """Remove a content source."""
    sources_config = SourcesConfig()
    
    if sources_config.remove_source(source_id):
        rprint(f"[green]✓[/green] Removed source: {source_id}")
    else:
        rprint(f"[red]✗[/red] Source not found: {source_id}")


@source.command("enable")
@click.argument("source_id")
def enable_source(source_id: str):
    """Enable a content source for processing."""
    sources_config = SourcesConfig()
    
    if sources_config.enable_source(source_id):
        rprint(f"[green]✓[/green] Enabled source: {source_id}")
    else:
        rprint(f"[red]✗[/red] Source not found: {source_id}")


@source.command("disable")
@click.argument("source_id")
def disable_source(source_id: str):
    """Disable a content source (won't be processed)."""
    sources_config = SourcesConfig()
    
    if sources_config.disable_source(source_id):
        rprint(f"[yellow]✓[/yellow] Disabled source: {source_id}")
    else:
        rprint(f"[red]✗[/red] Source not found: {source_id}")


@source.command("clear")
@click.confirmation_option(prompt="Are you sure you want to remove all sources?")
def clear_sources():
    """Clear all content sources (requires confirmation)."""
    sources_config = SourcesConfig()
    count = sources_config.clear_all()
    rprint(f"[green]✓[/green] Removed {count} sources")


@viz.command("dashboard")
@click.option("--database", default="idearank_all_content.db", help="Database to visualize")
@click.option("--channel", default=None, help="Filter to specific channel (optional)")
@click.option("--output-dir", default="idearank_viz", help="Output directory for plots")
def create_dashboard(database: str, channel: Optional[str], output_dir: str):
    """Create a complete visualization dashboard.
    
    Generates 4 plots:
    - Scores over time
    - Factor breakdown over time
    - Uniqueness vs Cohesion scatter plot
    - Score distribution histogram
    
    Example:
        idearank viz dashboard
        idearank viz dashboard --channel "https://youtube.com/@channel"
    """
    try:
        from idearank.visualizations import IdeaRankVisualizer
    except ImportError:
        rprint("[red]✗[/red] Visualization dependencies not installed")
        rprint("\nInstall with: [cyan]pip install matplotlib pandas[/cyan]")
        return
    
    if not Path(database).exists():
        rprint(f"[red]✗[/red] Database not found: {database}")
        rprint("\nProcess content first with: [cyan]idearank process-all[/cyan]")
        return
    
    rprint(f"[bold]Creating visualization dashboard...[/bold]")
    rprint(f"Database: {database}")
    if channel:
        rprint(f"Channel: {channel}")
    rprint(f"Output: {output_dir}/")
    
    try:
        with IdeaRankVisualizer(database) as viz:
            plots = viz.create_dashboard(
                content_source_id=channel,
                output_dir=output_dir,
            )
        
        rprint(f"\n[green]✓[/green] Created {len(plots)} visualizations:")
        for plot_path in plots:
            rprint(f"  - {plot_path}")
        
        rprint(f"\n[bold]Open {output_dir}/ to view your charts![/bold]")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error creating dashboard: {e}")
        raise


@viz.command("timeline")
@click.option("--database", default="idearank_all_content.db", help="Database to visualize")
@click.option("--channel", default=None, help="Filter to specific channel")
@click.option("--output", default="scores_timeline.png", help="Output file path")
def plot_timeline(database: str, channel: Optional[str], output: str):
    """Plot IdeaRank scores over time.
    
    Example:
        idearank viz timeline
        idearank viz timeline --channel "my-channel-id"
    """
    try:
        from idearank.visualizations import IdeaRankVisualizer
    except ImportError:
        rprint("[red]✗[/red] matplotlib not installed")
        rprint("Install with: [cyan]pip install matplotlib[/cyan]")
        return
    
    if not Path(database).exists():
        rprint(f"[red]✗[/red] Database not found: {database}")
        return
    
    try:
        with IdeaRankVisualizer(database) as viz:
            plot_path = viz.plot_scores_over_time(
                channel_id=channel,
                output_path=output,
            )
        
        rprint(f"[green]✓[/green] Created timeline: {plot_path}")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")


@viz.command("factors")
@click.option("--database", default="idearank_all_content.db", help="Database to visualize")
@click.option("--channel", default=None, help="Filter to specific channel")
@click.option("--output", default="factor_breakdown.png", help="Output file path")
def plot_factors(database: str, channel: Optional[str], output: str):
    """Plot all five factors over time.
    
    Shows evolution of U, C, L, Q, T factors.
    
    Example:
        idearank viz factors --channel "my-channel"
    """
    try:
        from idearank.visualizations import IdeaRankVisualizer
    except ImportError:
        rprint("[red]✗[/red] matplotlib not installed")
        rprint("Install with: [cyan]pip install matplotlib[/cyan]")
        return
    
    if not Path(database).exists():
        rprint(f"[red]✗[/red] Database not found: {database}")
        return
    
    try:
        with IdeaRankVisualizer(database) as viz:
            plot_path = viz.plot_factor_breakdown(
                channel_id=channel,
                output_path=output,
            )
        
        rprint(f"[green]✓[/green] Created factor breakdown: {plot_path}")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")


@viz.command("learning")
@click.argument("channel_id")
@click.option("--database", default="idearank_all_content.db", help="Database to visualize")
@click.option("--output", default="learning_trajectory.png", help="Output file path")
def plot_learning(channel_id: str, database: str, output: str):
    """Plot learning trajectory for a specific channel.
    
    Shows how learning scores evolve (should trend upward for healthy channels).
    
    Example:
        idearank viz learning "https://youtube.com/@channel"
    """
    try:
        from idearank.visualizations import IdeaRankVisualizer
    except ImportError:
        rprint("[red]✗[/red] matplotlib not installed")
        rprint("Install with: [cyan]pip install matplotlib[/cyan]")
        return
    
    if not Path(database).exists():
        rprint(f"[red]✗[/red] Database not found: {database}")
        return
    
    try:
        with IdeaRankVisualizer(database) as viz:
            plot_path = viz.plot_learning_trajectory(
                content_source_id=channel_id,
                output_path=output,
            )
        
        rprint(f"[green]✓[/green] Created learning trajectory: {plot_path}")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")


@viz.command("scatter")
@click.option("--database", default="idearank_all_content.db", help="Database to visualize")
@click.option("--channel", default=None, help="Filter to specific channel")
@click.option("--output", default="uniqueness_vs_cohesion.png", help="Output file path")
def plot_scatter(database: str, channel: Optional[str], output: str):
    """Scatter plot of Uniqueness vs Cohesion.
    
    Helps identify content that is:
    - Novel & On-Brand (high U, high C)
    - Exploratory (high U, low C)
    - Repetitive but On-Brand (low U, high C)
    
    Example:
        idearank viz scatter
    """
    try:
        from idearank.visualizations import IdeaRankVisualizer
    except ImportError:
        rprint("[red]✗[/red] matplotlib not installed")
        rprint("Install with: [cyan]pip install matplotlib[/cyan]")
        return
    
    if not Path(database).exists():
        rprint(f"[red]✗[/red] Database not found: {database}")
        return
    
    try:
        with IdeaRankVisualizer(database) as viz:
            plot_path = viz.plot_uniqueness_vs_cohesion(
                channel_id=channel,
                output_path=output,
            )
        
        rprint(f"[green]✓[/green] Created scatter plot: {plot_path}")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")


@main.command("diagnose")
@click.option("--database", default="idearank_all_content.db", help="Database to diagnose")
def diagnose(database: str):
    """Run diagnostics on IdeaRank database.
    
    Checks for common issues:
    - Flat cohesion scores
    - Flat trust scores
    - Learning score decay
    - Missing timestamps
    - Narrow score spread
    
    Example:
        idearank diagnose
        idearank diagnose --database my_results.db
    """
    if not Path(database).exists():
        rprint(f"[red]✗[/red] Database not found: {database}")
        return
    
    rprint(f"[bold]Running diagnostics on: {database}[/bold]\n")
    
    try:
        from idearank.diagnostics import IdeaRankDiagnostics
        
        with IdeaRankDiagnostics(database) as diag:
            diag.print_report()
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error running diagnostics: {e}")
        raise


@main.command("process-all")
@click.option("--output", default="idearank_all_content.db", help="SQLite database output path")
@click.option("--collection", default="idearank_all_content", help="Chroma collection name")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for embeddings")
def process_all(output: str, collection: str, openai_key: Optional[str]):
    """Process all enabled content sources.
    
    This command processes all sources you've added with 'idearank source add'
    and stores them in a unified database for cross-platform analysis.
    
    Example:
        idearank source add https://youtube.com/@channel1
        idearank source add my-blog.ghost.json
        idearank process-all
    """
    sources_config = SourcesConfig()
    cfg = CLIConfig()
    
    sources = sources_config.list_sources(enabled_only=True)
    
    if not sources:
        rprint("[yellow]No enabled content sources found.[/yellow]")
        rprint("\nAdd sources with: [cyan]idearank source add URL_OR_FILE[/cyan]")
        rprint("View sources with: [cyan]idearank source list[/cyan]")
        return
    
    # Show summary
    console.print(Panel.fit(
        f"[bold]Processing All Content Sources[/bold]\n\n"
        f"Sources: {len(sources)}\n"
        f"Output: {output}\n"
        f"Collection: {collection}",
        border_style="blue"
    ))
    
    rprint("\nSources to process:")
    for i, source in enumerate(sources, 1):
        rprint(f"  {i}. [{source.type}] {source.name} ({source.max_items} items)")
    
    # Process each source
    for i, source in enumerate(sources, 1):
        rprint(f"\n{'='*70}")
        rprint(f"[bold]Processing {i}/{len(sources)}: {source.name}[/bold]")
        rprint(f"{'='*70}")
        
        try:
            if source.type == "youtube":
                _process_youtube_source(source, cfg, output, collection, openai_key)
            elif source.type == "ghost_export":
                _process_ghost_export_source(source, output, collection, openai_key)
            elif source.type == "ghost_api":
                _process_ghost_api_source(source, cfg, output, collection, openai_key)
            elif source.type == "twitter":
                _process_twitter_source(source, output, collection, openai_key)
            else:
                rprint(f"[red]✗[/red] Unknown source type: {source.type}")
                continue
            
            # Mark as processed
            sources_config.mark_processed(source.id)
            rprint(f"[green]✓[/green] Completed: {source.name}")
            
        except Exception as e:
            rprint(f"[red]✗[/red] Failed to process {source.name}: {e}")
            logger.error(f"Error processing {source.id}: {e}", exc_info=True)
            continue
    
    rprint(f"\n{'='*70}")
    rprint(f"[bold green]✓ All sources processed![/bold green]")
    rprint(f"{'='*70}")
    rprint(f"\nResults saved to: {output}")
    rprint(f"Chroma collection: {collection}")


def _process_youtube_source(source: ContentSource, cfg: CLIConfig, output: str, collection: str, openai_key: Optional[str]):
    """Process a YouTube source."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
    
    # Initialize storage
    storage = SQLiteStorage(db_path=output)
    
    # YouTube client
    youtube_client = YouTubeClient(
        youtube_api_key=cfg.get_youtube_api_key(),
        gladia_api_key=cfg.get_gladia_api_key(),
        storage=storage,
    )
    
    # Chroma provider
    persist_dir = Path.home() / ".idearank" / "chroma_db"
    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_provider = ChromaProvider(
        collection_name=collection,
        persist_directory=str(persist_dir),
    )
    
    # Embedding provider
    if openai_key:
        embedding_provider = OpenAIEmbeddingProvider(api_key=openai_key)
    else:
        try:
            embedding_provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        except ImportError:
            embedding_provider = DummyEmbeddingProvider(dimension=384)
    
    # Pipeline
    idearank_config = IdeaRankConfig.default()
    idearank_pipeline = IdeaRankPipeline(
        config=idearank_config,
        embedding_provider=embedding_provider,
        topic_provider=DummyTopicModelProvider(),
        neighborhood_provider=chroma_provider,
    )
    
    youtube_pipeline = YouTubePipeline(
        idearank_pipeline=idearank_pipeline,
        youtube_client=youtube_client,
        storage=storage,
        config=idearank_config,
    )
    
    # Process
    youtube_pipeline.process_channel(
        channel_url=source.url_or_path,
        max_videos=source.max_items,
    )
    
    storage.close()


def _process_ghost_export_source(source: ContentSource, output: str, collection: str, openai_key: Optional[str]):
    """Process a Ghost export source."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
    from idearank.integrations.ghost_export import GhostExportClient
    
    # Initialize storage
    storage = SQLiteStorage(db_path=output)
    
    # Ghost client
    ghost_client = GhostExportClient(
        export_file_path=source.url_or_path,
        storage=storage,
    )
    
    # Chroma provider
    persist_dir = Path.home() / ".idearank" / "chroma_db"
    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_provider = ChromaProvider(
        collection_name=collection,
        persist_directory=str(persist_dir),
    )
    
    # Embedding provider
    if openai_key:
        embedding_provider = OpenAIEmbeddingProvider(api_key=openai_key)
    else:
        try:
            embedding_provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        except ImportError:
            embedding_provider = DummyEmbeddingProvider(dimension=384)
    
    # Pipeline
    idearank_config = IdeaRankConfig.default()
    idearank_pipeline = IdeaRankPipeline(
        config=idearank_config,
        embedding_provider=embedding_provider,
        topic_provider=DummyTopicModelProvider(),
        neighborhood_provider=chroma_provider,
    )
    
    ghost_pipeline = GhostPipeline(
        idearank_pipeline=idearank_pipeline,
        ghost_client=ghost_client,
        storage=storage,
        config=idearank_config,
    )
    
    # Process
    ghost_pipeline.process_blog(
        blog_url=ghost_client.blog_url,
        max_posts=source.max_items,
        filter_query=source.filter_query,
    )
    
    storage.close()


def _process_twitter_source(source: ContentSource, output: str, collection: str, openai_key: Optional[str]):
    """Process a Twitter source."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider, OpenAIEmbeddingProvider
    from idearank.providers.topics import LDATopicModelProvider, DummyTopicModelProvider
    from idearank.pipelines.twitter_pipeline import process_twitter_archive
    from idearank.integrations.twitter import fetch_twitter_archive
    
    # Initialize storage
    storage = SQLiteStorage(db_path=output)
    
    # Chroma provider
    persist_dir = Path.home() / ".idearank" / "chroma_db"
    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_provider = ChromaProvider(
        collection_name=collection,
        persist_directory=str(persist_dir),
    )
    
    # Embedding provider
    if openai_key:
        embedding_provider = OpenAIEmbeddingProvider(api_key=openai_key)
    else:
        try:
            embedding_provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        except ImportError:
            embedding_provider = DummyEmbeddingProvider(dimension=384)
    
    # Topic provider
    try:
        topic_provider = LDATopicModelProvider(num_topics=30)
    except ImportError:
        topic_provider = DummyTopicModelProvider()
    
    # Check if it's a direct archive URL or username
    archive_url = None
    username = source.url_or_path.lstrip('@').strip()
    
    if source.url_or_path.startswith('http'):
        archive_url = source.url_or_path
        # Don't pass username for URL-based fetching - let it extract from URL
        username = None
        rprint(f"[blue]Fetching Twitter archive from direct URL...[/blue]")
    else:
        rprint(f"[blue]Looking up Twitter archive in Community Archive...[/blue]")
    
    # Fetch the archive first
    archive = fetch_twitter_archive(username, limit=source.max_items, archive_url=archive_url)
    
    if not archive:
        if archive_url:
            rprint(f"[red]❌ Could not fetch Twitter archive from URL: {archive_url}[/red]")
        else:
            rprint(f"[red]❌ Could not fetch Twitter archive for @{username}. Archive may not be available on Community Archive.[/red]")
        return
    
    rprint(f"[green]✅ Fetched {len(archive.posts)} tweets from @{archive.username}[/green]")
    
    # Process Twitter archive
    rprint(f"[blue]Processing Twitter archive for @{archive.username}...[/blue]")
    
    results = process_twitter_archive(
        archive=archive,
        storage=storage,
        embedding_provider=embedding_provider,
        topic_provider=topic_provider,
        chroma_provider=chroma_provider,
    )
    
    if results['success']:
        stats = results['statistics']
        rprint(f"[green]✓[/green] Processed {results['processed_count']} tweets")
        rprint(f"  Average score: {stats['average_score']:.4f}")
        rprint(f"  Score range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
        rprint(f"  Total engagement: {stats['total_engagement']:,}")
        rprint(f"  Unique hashtags: {stats['unique_hashtags']}")
        rprint(f"  Unique mentions: {stats['unique_mentions']}")
    else:
        rprint(f"[red]✗[/red] {results['error']}")
    
    storage.close()


def _process_ghost_api_source(source: ContentSource, cfg: CLIConfig, output: str, collection: str, openai_key: Optional[str]):
    """Process a Ghost API source."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
    
    # Initialize storage
    storage = SQLiteStorage(db_path=output)
    
    # Ghost client
    ghost_client = GhostClient(
        blog_url=source.url_or_path,
        content_api_key=cfg.get_gladia_api_key(),  # TODO: Add ghost_api_key to config
        storage=storage,
    )
    
    # Chroma provider
    persist_dir = Path.home() / ".idearank" / "chroma_db"
    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_provider = ChromaProvider(
        collection_name=collection,
        persist_directory=str(persist_dir),
    )
    
    # Embedding provider
    if openai_key:
        embedding_provider = OpenAIEmbeddingProvider(api_key=openai_key)
    else:
        try:
            embedding_provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        except ImportError:
            embedding_provider = DummyEmbeddingProvider(dimension=384)
    
    # Pipeline
    idearank_config = IdeaRankConfig.default()
    idearank_pipeline = IdeaRankPipeline(
        config=idearank_config,
        embedding_provider=embedding_provider,
        topic_provider=DummyTopicModelProvider(),
        neighborhood_provider=chroma_provider,
    )
    
    ghost_pipeline = GhostPipeline(
        idearank_pipeline=idearank_pipeline,
        ghost_client=ghost_client,
        storage=storage,
        config=idearank_config,
    )
    
    # Process
    ghost_pipeline.process_blog(
        blog_url=source.url_or_path,
        max_posts=source.max_items,
        filter_query=source.filter_query,
    )
    
    storage.close()


@main.command()
@click.argument("channel_url")
@click.option("--max-videos", default=50, help="Maximum number of videos to process")
@click.option("--output", default="idearank_results.db", help="SQLite database output path")
@click.option("--collection", default=None, help="Chroma collection name (default: auto-generated)")
@click.option("--use-gladia/--no-gladia", default=False, help="Use Gladia for missing transcripts")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for embeddings")
def process(
    channel_url: str,
    max_videos: int,
    output: str,
    collection: Optional[str],
    use_gladia: bool,
    openai_key: Optional[str],
):
    """Process a YouTube channel and compute IdeaRank scores.
    
    Example:
        idearank process https://youtube.com/@channelname --max-videos 20
    """
    # Load config
    cfg = CLIConfig()
    is_configured, missing = cfg.is_configured()
    
    if not is_configured:
        rprint(f"[red]✗[/red] Missing configuration: {', '.join(missing)}")
        rprint("\nRun [cyan]idearank config show[/cyan] to see current configuration")
        raise click.Abort()
    
    # Show configuration summary
    console.print(Panel.fit(
        f"[bold]Processing YouTube Channel[/bold]\n\n"
        f"Channel: {channel_url}\n"
        f"Max Videos: {max_videos}\n"
        f"Output: {output}\n"
        f"Chroma Mode: {cfg.get_chroma_mode()}\n"
        f"Use Gladia: {'Yes' if use_gladia else 'No (YouTube transcripts only)'}",
        border_style="blue"
    ))
    
    try:
        # Initialize components
        rprint("\n[bold]Initializing components...[/bold]")
        
        # Storage (initialize first for transcript caching)
        storage = SQLiteStorage(db_path=output)
        rprint(f"[green]✓[/green] SQLite storage ready: {output}")
        
        # YouTube client (with storage for transcript caching)
        youtube_api_key = cfg.get_youtube_api_key()
        gladia_api_key = cfg.get_gladia_api_key() if use_gladia else None
        
        youtube_client = YouTubeClient(
            youtube_api_key=youtube_api_key,
            gladia_api_key=gladia_api_key,
            storage=storage,  # Enable transcript caching
        )
        rprint("[green]✓[/green] YouTube client ready (with transcript caching)")
        
        # Chroma provider
        chroma_mode = cfg.get_chroma_mode()
        
        if chroma_mode == "cloud":
            chroma_config = cfg.get_chroma_cloud_config()
            chroma_provider = ChromaProvider(
                collection_name=collection or f"idearank_{Path(output).stem}",
                persist_directory=None,  # Cloud mode
                chroma_cloud_api_key=chroma_config["api_key"],
                chroma_cloud_tenant=chroma_config["tenant"],
                chroma_cloud_database=chroma_config["database"],
            )
            rprint("[green]✓[/green] Chroma Cloud provider ready")
        else:
            # Local mode
            persist_dir = Path.home() / ".idearank" / "chroma_db"
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            chroma_provider = ChromaProvider(
                collection_name=collection or f"idearank_{Path(output).stem}",
                persist_directory=str(persist_dir),
            )
            rprint(f"[green]✓[/green] Chroma local provider ready: {persist_dir}")
        
        # Embedding provider
        if openai_key:
            embedding_provider = OpenAIEmbeddingProvider(api_key=openai_key)
            rprint("[green]✓[/green] OpenAI embedding provider ready")
        else:
            # Use sentence-transformers by default (no API key needed!)
            try:
                from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
                embedding_provider = SentenceTransformerEmbeddingProvider(
                    model_name="all-MiniLM-L6-v2"  # Fast, efficient model
                )
                rprint("[green]✓[/green] Sentence-transformers embedding provider ready (all-MiniLM-L6-v2)")
            except ImportError:
                # Fall back to dummy if sentence-transformers not installed
                from idearank.providers.embeddings import DummyEmbeddingProvider
                embedding_provider = DummyEmbeddingProvider(dimension=384)
                rprint("[yellow]⚠[/yellow]  Using dummy embeddings (install sentence-transformers for better quality)")
        
        # Topic provider (use real LDA!)
        try:
            topic_provider = LDATopicModelProvider(num_topics=30)
            rprint("[green]✓[/green] LDA topic model ready (30 topics)")
        except ImportError:
            topic_provider = DummyTopicModelProvider()
            rprint("[yellow]⚠[/yellow]  Using dummy topic model (install scikit-learn for better cohesion scores)")
        
        # IdeaRank pipeline
        idearank_config = IdeaRankConfig.default()
        idearank_pipeline = IdeaRankPipeline(
            config=idearank_config,
            embedding_provider=embedding_provider,
            topic_provider=topic_provider,
            neighborhood_provider=chroma_provider,
        )
        rprint("[green]✓[/green] IdeaRank pipeline ready")
        
        # YouTube pipeline
        youtube_pipeline = YouTubePipeline(
            idearank_pipeline=idearank_pipeline,
            youtube_client=youtube_client,
            storage=storage,
            config=idearank_config,
        )
        
        # Process channel
        rprint("\n[bold]Processing channel...[/bold]")
        content_source, scores = youtube_pipeline.process_channel(
            channel_url=channel_url,
            max_videos=max_videos,
        )
        
        # Display results
        rprint("\n[bold green]✓ Processing complete![/bold green]")
        
        # Create results table
        table = Table(title=f"IdeaRank Scores - {content_source.name}")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Title", style="white", width=50)
        table.add_column("Score", style="green", justify="right")
        table.add_column("U", justify="right")
        table.add_column("C", justify="right")
        table.add_column("L", justify="right")
        table.add_column("Q", justify="right")
        table.add_column("T", justify="right")
        table.add_column("Gates", justify="center")
        
        # Sort by score
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )
        
        for i, (item_id, score) in enumerate(sorted_scores[:10], 1):
            item = next(item for item in content_source.content_items if item.id == item_id)
            title = item.title[:47] + "..." if len(item.title) > 50 else item.title
            
            table.add_row(
                str(i),
                title,
                f"{score.score:.4f}",
                f"{score.uniqueness.score:.2f}",
                f"{score.cohesion.score:.2f}",
                f"{score.learning.score:.2f}",
                f"{score.quality.score:.2f}",
                f"{score.trust.score:.2f}",
                "✓" if score.passes_gates else "✗",
            )
        
        console.print(table)
        
        rprint(f"\n[green]✓[/green] Results saved to: {output}")
        rprint(f"[green]✓[/green] Total videos processed: {len(scores)}")
        
        # Clean up
        storage.close()
        
    except Exception as e:
        rprint(f"\n[red]✗ Error:[/red] {e}")
        raise


@main.command("process-ghost-export")
@click.argument("export_file", type=click.Path(exists=True))
@click.option("--max-posts", default=50, help="Maximum number of posts to process")
@click.option("--output", default="idearank_results.db", help="SQLite database output path")
@click.option("--collection", default=None, help="Chroma collection name (default: auto-generated)")
@click.option("--filter", "filter_query", default=None, help="Filter: 'tag:NAME' or 'author:NAME'")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for embeddings")
def process_ghost_export(
    export_file: str,
    max_posts: int,
    output: str,
    collection: Optional[str],
    filter_query: Optional[str],
    openai_key: Optional[str],
):
    """Process a Ghost export file and compute IdeaRank scores.
    
    Export your Ghost blog: Settings → Labs → Export
    
    Example:
        idearank process-ghost-export my-blog.ghost.2024-10-09.json --max-posts 20
    """
    # Show configuration summary
    console.print(Panel.fit(
        f"[bold]Processing Ghost Export File[/bold]\n\n"
        f"Export File: {export_file}\n"
        f"Max Posts: {max_posts}\n"
        f"Output: {output}\n"
        f"Filter: {filter_query or 'None'}",
        border_style="blue"
    ))
    
    try:
        # Initialize components
        rprint("\n[bold]Initializing components...[/bold]")
        
        # Storage (initialize first for content caching)
        storage = SQLiteStorage(db_path=output)
        rprint(f"[green]✓[/green] SQLite storage ready: {output}")
        
        # Ghost export client
        ghost_client = GhostExportClient(
            export_file_path=export_file,
            storage=storage,
        )
        
        # Show export stats
        stats = ghost_client.get_export_stats()
        rprint(f"[green]✓[/green] Loaded Ghost export: {stats['blog_title']}")
        rprint(f"    Total posts: {stats['total_posts']} | Published: {stats['published']} | Tags: {stats['total_tags']}")
        
        # Chroma provider (use local for Ghost)
        persist_dir = Path.home() / ".idearank" / "chroma_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        chroma_provider = ChromaProvider(
            collection_name=collection or f"idearank_{Path(output).stem}",
            persist_directory=str(persist_dir),
        )
        rprint(f"[green]✓[/green] Chroma local provider ready: {persist_dir}")
        
        # Embedding provider
        if openai_key:
            embedding_provider = OpenAIEmbeddingProvider(api_key=openai_key)
            rprint("[green]✓[/green] OpenAI embedding provider ready")
        else:
            # Use sentence-transformers by default (no API key needed!)
            try:
                from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
                embedding_provider = SentenceTransformerEmbeddingProvider(
                    model_name="all-MiniLM-L6-v2"  # Fast, efficient model
                )
                rprint("[green]✓[/green] Sentence-transformers embedding provider ready (all-MiniLM-L6-v2)")
            except ImportError:
                # Fall back to dummy if sentence-transformers not installed
                from idearank.providers.embeddings import DummyEmbeddingProvider
                embedding_provider = DummyEmbeddingProvider(dimension=384)
                rprint("[yellow]⚠[/yellow]  Using dummy embeddings (install sentence-transformers for better quality)")
        
        # Topic provider (use real LDA!)
        try:
            topic_provider = LDATopicModelProvider(num_topics=30)
            rprint("[green]✓[/green] LDA topic model ready (30 topics)")
        except ImportError:
            topic_provider = DummyTopicModelProvider()
            rprint("[yellow]⚠[/yellow]  Using dummy topic model (install scikit-learn for better cohesion scores)")
        
        # IdeaRank pipeline
        idearank_config = IdeaRankConfig.default()
        idearank_pipeline = IdeaRankPipeline(
            config=idearank_config,
            embedding_provider=embedding_provider,
            topic_provider=topic_provider,
            neighborhood_provider=chroma_provider,
        )
        rprint("[green]✓[/green] IdeaRank pipeline ready")
        
        # Ghost pipeline
        ghost_pipeline = GhostPipeline(
            idearank_pipeline=idearank_pipeline,
            ghost_client=ghost_client,
            storage=storage,
            config=idearank_config,
        )
        
        # Process blog
        rprint("\n[bold]Processing blog...[/bold]")
        content_source, scores = ghost_pipeline.process_blog(
            blog_url=ghost_client.blog_url,
            max_posts=max_posts,
            filter_query=filter_query,
        )
        
        # Display results
        rprint("\n[bold green]✓ Processing complete![/bold green]")
        
        # Create results table
        table = Table(title=f"IdeaRank Scores - {content_source.name}")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Title", style="white", width=50)
        table.add_column("Score", style="green", justify="right")
        table.add_column("U", justify="right")
        table.add_column("C", justify="right")
        table.add_column("L", justify="right")
        table.add_column("Q", justify="right")
        table.add_column("T", justify="right")
        table.add_column("Gates", justify="center")
        
        # Sort by score
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )
        
        for i, (item_id, score) in enumerate(sorted_scores[:10], 1):
            item = next(item for item in content_source.content_items if item.id == item_id)
            title = item.title[:47] + "..." if len(item.title) > 50 else item.title
            
            table.add_row(
                str(i),
                title,
                f"{score.score:.4f}",
                f"{score.uniqueness.score:.2f}",
                f"{score.cohesion.score:.2f}",
                f"{score.learning.score:.2f}",
                f"{score.quality.score:.2f}",
                f"{score.trust.score:.2f}",
                "✓" if score.passes_gates else "✗",
            )
        
        console.print(table)
        
        rprint(f"\n[green]✓[/green] Results saved to: {output}")
        rprint(f"\n[green]✓[/green] Total posts processed: {len(scores)}")
        
        # Clean up
        storage.close()
        
    except Exception as e:
        rprint(f"\n[red]✗ Error:[/red] {e}")
        raise


@main.command("process-ghost")
@click.argument("blog_url")
@click.option("--max-posts", default=50, help="Maximum number of posts to process")
@click.option("--output", default="idearank_results.db", help="SQLite database output path")
@click.option("--collection", default=None, help="Chroma collection name (default: auto-generated)")
@click.option("--api-key", envvar="GHOST_API_KEY", help="Ghost Content API key (PREMIUM ONLY)")
@click.option("--filter", "filter_query", default=None, help="Ghost filter query (e.g., 'tag:python')")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for embeddings")
def process_ghost(
    blog_url: str,
    max_posts: int,
    output: str,
    collection: Optional[str],
    api_key: Optional[str],
    filter_query: Optional[str],
    openai_key: Optional[str],
):
    """Process a Ghost blog via API (PREMIUM/PAID Ghost plans only).
    
    For free Ghost users, use: idearank process-ghost-export
    
    Example:
        idearank process-ghost https://blog.example.com --api-key YOUR_KEY
    """
    # Show configuration summary
    console.print(Panel.fit(
        f"[bold]Processing Ghost Blog[/bold]\n\n"
        f"Blog URL: {blog_url}\n"
        f"Max Posts: {max_posts}\n"
        f"Output: {output}\n"
        f"Filter: {filter_query or 'None'}",
        border_style="blue"
    ))
    
    try:
        # Initialize components
        rprint("\n[bold]Initializing components...[/bold]")
        
        # Storage (initialize first for content caching)
        storage = SQLiteStorage(db_path=output)
        rprint(f"[green]✓[/green] SQLite storage ready: {output}")
        
        # Ghost client (with storage for content caching)
        ghost_client = GhostClient(
            blog_url=blog_url,
            content_api_key=api_key,
            storage=storage,
        )
        rprint("[green]✓[/green] Ghost client ready (with content caching)")
        
        # Chroma provider (use local for Ghost)
        persist_dir = Path.home() / ".idearank" / "chroma_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        chroma_provider = ChromaProvider(
            collection_name=collection or f"idearank_{Path(output).stem}",
            persist_directory=str(persist_dir),
        )
        rprint(f"[green]✓[/green] Chroma local provider ready: {persist_dir}")
        
        # Embedding provider
        if openai_key:
            embedding_provider = OpenAIEmbeddingProvider(api_key=openai_key)
            rprint("[green]✓[/green] OpenAI embedding provider ready")
        else:
            # Use sentence-transformers by default (no API key needed!)
            try:
                from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
                embedding_provider = SentenceTransformerEmbeddingProvider(
                    model_name="all-MiniLM-L6-v2"  # Fast, efficient model
                )
                rprint("[green]✓[/green] Sentence-transformers embedding provider ready (all-MiniLM-L6-v2)")
            except ImportError:
                # Fall back to dummy if sentence-transformers not installed
                from idearank.providers.embeddings import DummyEmbeddingProvider
                embedding_provider = DummyEmbeddingProvider(dimension=384)
                rprint("[yellow]⚠[/yellow]  Using dummy embeddings (install sentence-transformers for better quality)")
        
        # Topic provider (use real LDA!)
        try:
            topic_provider = LDATopicModelProvider(num_topics=30)
            rprint("[green]✓[/green] LDA topic model ready (30 topics)")
        except ImportError:
            topic_provider = DummyTopicModelProvider()
            rprint("[yellow]⚠[/yellow]  Using dummy topic model (install scikit-learn for better cohesion scores)")
        
        # IdeaRank pipeline
        idearank_config = IdeaRankConfig.default()
        idearank_pipeline = IdeaRankPipeline(
            config=idearank_config,
            embedding_provider=embedding_provider,
            topic_provider=topic_provider,
            neighborhood_provider=chroma_provider,
        )
        rprint("[green]✓[/green] IdeaRank pipeline ready")
        
        # Ghost pipeline
        ghost_pipeline = GhostPipeline(
            idearank_pipeline=idearank_pipeline,
            ghost_client=ghost_client,
            storage=storage,
            config=idearank_config,
        )
        
        # Process blog
        rprint("\n[bold]Processing blog...[/bold]")
        content_source, scores = ghost_pipeline.process_blog(
            blog_url=blog_url,
            max_posts=max_posts,
            filter_query=filter_query,
        )
        
        # Display results
        rprint("\n[bold green]✓ Processing complete![/bold green]")
        
        # Create results table
        table = Table(title=f"IdeaRank Scores - {content_source.name}")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Title", style="white", width=50)
        table.add_column("Score", style="green", justify="right")
        table.add_column("U", justify="right")
        table.add_column("C", justify="right")
        table.add_column("L", justify="right")
        table.add_column("Q", justify="right")
        table.add_column("T", justify="right")
        table.add_column("Gates", justify="center")
        
        # Sort by score
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )
        
        for i, (item_id, score) in enumerate(sorted_scores[:10], 1):
            item = next(item for item in content_source.content_items if item.id == item_id)
            title = item.title[:47] + "..." if len(item.title) > 50 else item.title
            
            table.add_row(
                str(i),
                title,
                f"{score.score:.4f}",
                f"{score.uniqueness.score:.2f}",
                f"{score.cohesion.score:.2f}",
                f"{score.learning.score:.2f}",
                f"{score.quality.score:.2f}",
                f"{score.trust.score:.2f}",
                "✓" if score.passes_gates else "✗",
            )
        
        console.print(table)
        
        rprint(f"\n[green]✓[/green] Results saved to: {output}")
        rprint(f"[green]✓[/green] Total posts processed: {len(scores)}")
        
        # Clean up
        storage.close()
        
    except Exception as e:
        rprint(f"\n[red]✗ Error:[/red] {e}")
        raise


@main.command()
def setup():
    """Interactive setup wizard for first-time configuration."""
    console.print(Panel.fit(
        "[bold]Welcome to IdeaRank![/bold]\n\n"
        "Let's set up your configuration.\n"
        "You can skip optional steps by pressing Enter.",
        border_style="blue"
    ))
    
    cfg = CLIConfig()
    
    # YouTube API Key (required)
    rprint("\n[bold]Step 1: YouTube Data API[/bold]")
    rprint("Get your API key from: https://console.cloud.google.com/apis/credentials")
    youtube_key = click.prompt("YouTube API Key", type=str)
    cfg.set_youtube_api_key(youtube_key)
    rprint("[green]✓[/green] Saved")
    
    # Gladia API Key (optional)
    rprint("\n[bold]Step 2: Gladia Transcription (Optional)[/bold]")
    rprint("Gladia provides high-quality transcription for videos without captions.")
    rprint("Get your API key from: https://gladia.io")
    gladia_key = click.prompt("Gladia API Key (optional, press Enter to skip)", default="", show_default=False)
    if gladia_key:
        cfg.set_gladia_api_key(gladia_key)
        rprint("[green]✓[/green] Saved")
    else:
        rprint("[yellow]⊘[/yellow] Skipped (will use YouTube transcripts only)")
    
    # Chroma mode
    rprint("\n[bold]Step 3: Chroma Vector Storage[/bold]")
    chroma_mode = click.prompt(
        "Use Chroma Cloud or Local storage?",
        type=click.Choice(["local", "cloud"]),
        default="local"
    )
    cfg.set_chroma_mode(chroma_mode)
    
    if chroma_mode == "cloud":
        rprint("\nChroma Cloud configuration:")
        rprint("Note: ChromaDB client connects automatically - no URL needed!")
        cloud_key = click.prompt("Chroma Cloud API Key", type=str)
        cloud_tenant = click.prompt("Tenant", default="default_tenant")
        cloud_db = click.prompt("Database", default="default_database")
        
        cfg.set_chroma_cloud(
            api_key=cloud_key,
            tenant=cloud_tenant,
            database=cloud_db,
        )
        rprint("[green]✓[/green] Chroma Cloud configured")
    else:
        rprint(f"[green]✓[/green] Using local storage at: ~/.idearank/chroma_db")
    
    # Done!
    console.print(Panel.fit(
        "[bold green]Setup Complete![/bold green]\n\n"
        "You're ready to process YouTube channels.\n\n"
        "Try: [cyan]idearank process https://youtube.com/@channelname[/cyan]\n\n"
        "See all commands: [cyan]idearank --help[/cyan]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()

