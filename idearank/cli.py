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
@click.version_option(version="2.1.0", prog_name="idearank")
def main():
    """IdeaRank - Multi-factor ranking for content.
    
    A PageRank replacement for ideas that analyzes uniqueness, cohesion,
    learning trajectory, quality, and trust signals across any content source.
    
    Now includes IdeaRank-Thought competition system for real-time reasoning evaluation!
    """
    pass


@main.command()
def demo_competition():
    """Run the IdeaRank-Thought competition system demo."""
    try:
        import sys
        import os
        
        # Add examples directory to path
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')
        if os.path.exists(examples_dir):
            sys.path.insert(0, examples_dir)
        
        # Import and run the demo
        from competition_demo import run_competition_demo
        console.print("[bold green]Starting IdeaRank-Thought Competition Demo[/bold green]")
        run_competition_demo()
        console.print("[bold green]Demo completed! Check the generated HTML files for visualizations.[/bold green]")
    except ImportError as e:
        console.print(f"[bold red]Error: Competition system not available[/bold red]")
        console.print(f"Details: {e}")
        console.print("Make sure all dependencies are installed and the examples directory exists.")
    except Exception as e:
        console.print(f"[bold red]Error running competition demo: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

@main.command()
def play_game():
    """Play an interactive IdeaRank-Thought competition game."""
    try:
        from idearank.interactive_game import InteractiveGame
        game = InteractiveGame()
        game.run()
    except ImportError as e:
        console.print(f"[bold red]Error: Interactive game not available[/bold red]")
        console.print(f"Details: {e}")
        console.print("Make sure all dependencies are installed.")
    except Exception as e:
        console.print(f"[bold red]Error running interactive game: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


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


@config.command("set-whisper-model")
@click.argument("model", type=click.Choice(["tiny", "base", "small", "medium", "large"]))
def set_whisper_model(model: str):
    """Set Whisper model size for transcription.
    
    Options: tiny (fastest), base, small (default), medium, large (best quality)
    """
    cfg = CLIConfig()
    cfg.set_whisper_model(model)
    
    rprint(f"[green]✓[/green] Whisper model set to: {model}")


@config.command("set-whisper-device")
@click.argument("device", type=click.Choice(["cpu", "cuda", "auto"]))
def set_whisper_device(device: str):
    """Set Whisper device for transcription.
    
    Options: cpu, cuda (GPU), auto (default - uses GPU if available)
    """
    cfg = CLIConfig()
    cfg.set_whisper_device(device)
    
    rprint(f"[green]✓[/green] Whisper device set to: {device}")


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
    table.add_row("Whisper Model", config_data.get("whisper_model", "small"))
    table.add_row("Whisper Device", config_data.get("whisper_device", "auto"))
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
@click.option("--max-items", default=50, type=int, help="Maximum items to process")
@click.option("--all", "process_all_items", is_flag=True, help="Process ALL items (sets max-items to unlimited)")
@click.option("--filter", "filter_query", help="Filter query (e.g., 'tag:python')")
@click.option("--type", "content_type", type=click.Choice(["youtube", "ghost_export", "ghost_api", "medium", "twitter", "auto"]), default="auto", help="Content type (auto-detect by default)")
def add_source(url_or_file: str, name: Optional[str], max_items: int, process_all_items: bool, filter_query: Optional[str], content_type: str):
    """Add a content source to process.
    
    Examples:
        idearank source add https://youtube.com/@channel
        idearank source add my-blog.ghost.json --all
        idearank source add https://blog.example.com --type ghost_api --max-items 100
        idearank source add ~/Downloads/twitter-archive.json --type twitter --all
    """
    sources_config = SourcesConfig()
    
    # Determine actual max_items value
    actual_max_items = -1 if process_all_items else max_items  # -1 signals unlimited
    
    # Create source with auto-detection
    source = ContentSource.create(
        url_or_path=url_or_file,
        name=name,
        max_items=actual_max_items,
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
        rprint(f"    Max items: {'ALL (unlimited)' if source.max_items == -1 else source.max_items}")
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
    table.add_column("ID", style="cyan", width=35)  # Increased width for full IDs
    table.add_column("Name", style="white", width=30)
    table.add_column("Type", style="green", width=12)
    table.add_column("Items", justify="right", width=6)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Last Processed", style="dim", width=20)
    
    for source in sources:
        status = "[green]✓[/green]" if source.enabled else "[dim]✗[/dim]"
        last_proc = source.last_processed[:19] if source.last_processed else "Never"
        max_items_display = "ALL" if source.max_items == -1 else str(source.max_items)
        
        table.add_row(
            source.id,
            source.name or source.url_or_path[:25],
            source.type,
            max_items_display,
            status,
            last_proc,
        )
    
    console.print(table)
    rprint(f"\nTotal sources: {len(sources)}")
    if enabled_only:
        rprint(f"(Showing enabled only. Use without --enabled-only to see all)")
    
    # Show incomplete sources
    incomplete_sources = [s for s in sources if s.max_items != -1 and s.max_items < 1000]
    if incomplete_sources:
        rprint(f"\n[yellow]Incomplete sources (not all items processed):[/yellow]")
        for source in incomplete_sources:
            rprint(f"  [cyan]{source.id}[/cyan] - {source.name} ({source.max_items} items)")
        rprint(f"\nTo remove incomplete sources:")
        for source in incomplete_sources:
            rprint(f"  [dim]idearank source remove {source.id}[/dim]")


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


@source.command("status")
@click.option("--database", default="idearank_all_content.db", help="Database to check status against")
def source_status(database: str):
    """Show processing status of all sources."""
    from idearank.integrations.storage import SQLiteStorage
    
    sources_config = SourcesConfig()
    sources = sources_config.list_sources()
    
    if not sources:
        rprint("[yellow]No content sources configured.[/yellow]")
        return
    
    # Initialize storage to check content counts
    try:
        storage = SQLiteStorage(database)
    except Exception as e:
        rprint(f"[red]Database not found: {database}[/red]")
        rprint("Run [cyan]idearank process-all[/cyan] first to create the database.")
        return
    
    table = Table(title="Source Processing Status")
    table.add_column("Source", style="white", width=30)
    table.add_column("Type", style="green", width=12)
    table.add_column("Items", justify="right", width=6)
    table.add_column("In DB", justify="right", width=6)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Last Processed", style="dim", width=20)
    
    for source in sources:
        # Count items in database
        db_count = _find_existing_content_count(storage, source)
        
        # Determine status
        if not source.enabled:
            status = "[dim]disabled[/dim]"
        elif db_count == 0:
            status = "[yellow]pending[/yellow]"
        elif source.max_items != -1 and db_count >= source.max_items:
            status = "[green]complete[/green]"
        elif source.max_items == -1 and source.last_processed:
            status = "[green]complete[/green]"
        else:
            status = "[blue]partial[/blue]"
        
        max_items_display = "ALL" if source.max_items == -1 else str(source.max_items)
        last_proc = source.last_processed[:19] if source.last_processed else "Never"
        
        table.add_row(
            source.name or source.url_or_path[:25],
            source.type,
            max_items_display,
            str(db_count),
            status,
            last_proc,
        )
    
    storage.close()
    console.print(table)
    rprint(f"\nTotal sources: {len(sources)}")
    rprint(f"Database: {database}")


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
@click.option("--force", is_flag=True, help="Force reprocessing of all sources (ignore cache)")
@click.option("--skip-processed", is_flag=True, default=True, help="Skip sources that have already been processed")
def process_all(output: str, collection: str, openai_key: Optional[str], force: bool, skip_processed: bool):
    """Process all enabled content sources.
    
    This command processes all sources you've added with 'idearank source add'
    and stores them in a unified database for cross-platform analysis.
    
    By default, sources that have already been processed are skipped for efficiency.
    Use --force to reprocess all sources.
    
    Example:
        idearank source add https://youtube.com/@channel1
        idearank source add my-blog.ghost.json
        idearank process-all                    # Skip already processed sources
        idearank process-all --force            # Reprocess everything
    """
    sources_config = SourcesConfig()
    cfg = CLIConfig()
    
    sources = sources_config.list_sources(enabled_only=True)
    
    if not sources:
        rprint("[yellow]No enabled content sources found.[/yellow]")
        rprint("\nAdd sources with: [cyan]idearank source add URL_OR_FILE[/cyan]")
        rprint("View sources with: [cyan]idearank source list[/cyan]")
        return
    
    # Check what needs processing
    sources_to_process = []
    sources_skipped = []
    
    # Initialize storage to check what's already processed
    from idearank.integrations.storage import SQLiteStorage
    storage = SQLiteStorage(output)
    
    for source in sources:
        if force:
            sources_to_process.append(source)
        elif skip_processed and source.last_processed:
            # Check if source was processed recently and has content in database
            existing_count = _find_existing_content_count(storage, source)
            if existing_count > 0:
                sources_skipped.append((source, existing_count))
                continue
        
        # Only add to process list if not already added (force case) and not skipped
        if not force:
            sources_to_process.append(source)
    
    # Show summary
    console.print(Panel.fit(
        f"[bold]Processing All Content Sources[/bold]\n\n"
        f"Total sources: {len(sources)}\n"
        f"To process: {len(sources_to_process)}\n"
        f"Skipped (cached): {len(sources_skipped)}\n"
        f"Output: {output}\n"
        f"Collection: {collection}",
        border_style="blue"
    ))
    
    if sources_skipped:
        rprint(f"\n[yellow]Skipping {len(sources_skipped)} already processed sources:[/yellow]")
        for source, count in sources_skipped:
            rprint(f"  [dim]✓[/dim] [{source.type}] {source.name} ({count} items)")
    
    if sources_to_process:
        rprint(f"\nSources to process:")
        for i, source in enumerate(sources_to_process, 1):
            max_items_display = "ALL" if source.max_items == -1 else str(source.max_items)
            rprint(f"  {i}. [{source.type}] {source.name} ({max_items_display} items)")
    else:
        rprint(f"\n[green]✓ All sources already processed![/green]")
        rprint(f"Use [cyan]--force[/cyan] to reprocess all sources")
        return
    
    # Process each source that needs processing
    for i, source in enumerate(sources_to_process, 1):
        rprint(f"\n{'='*70}")
        rprint(f"[bold]Processing {i}/{len(sources_to_process)}: {source.name}[/bold]")
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
            elif source.type == "medium":
                _process_medium_source(source, output, collection, openai_key)
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
    
    storage.close()
    
    rprint(f"\n{'='*70}")
    if sources_skipped:
        rprint(f"[bold green]✓ Processing complete![/bold green]")
        rprint(f"Processed: {len(sources_to_process)} sources")
        rprint(f"Skipped (cached): {len(sources_skipped)} sources")
    else:
        rprint(f"[bold green]✓ All sources processed![/bold green]")
    rprint(f"{'='*70}")
    rprint(f"\nResults saved to: {output}")
    rprint(f"Chroma collection: {collection}")


def _count_source_content(storage, source_id: str) -> int:
    """Count how many content items exist for a source."""
    try:
        from idearank.integrations.storage import ContentItemRecord
        count = storage.session.query(ContentItemRecord).filter_by(content_source_id=source_id).count()
        return count
    except Exception:
        return 0


def _find_existing_content_count(storage, source) -> int:
    """Find existing content count for a source by matching name or URL patterns."""
    try:
        from idearank.integrations.storage import ContentItemRecord, ContentSourceRecord
        
        # First try exact source ID match
        count = storage.session.query(ContentItemRecord).filter_by(content_source_id=source.id).count()
        if count > 0:
            return count
        
        # Try to find by source name patterns
        if source.type == "medium":
            # Look for medium_* patterns or "Medium User" name
            medium_count = storage.session.query(ContentItemRecord).join(
                ContentSourceRecord, 
                ContentItemRecord.content_source_id == ContentSourceRecord.id
            ).filter(
                ContentSourceRecord.name.like('medium_%') | 
                ContentSourceRecord.name.like('%Medium%')
            ).count()
            if medium_count > 0:
                return medium_count
        
        elif source.type == "youtube":
            # Look for YouTube channel IDs (UC_*)
            youtube_count = storage.session.query(ContentItemRecord).filter(
                ContentItemRecord.content_source_id.like('UC_%')
            ).count()
            if youtube_count > 0:
                return youtube_count
        
        elif source.type == "ghost_export":
            # Look for ghost patterns or "Idea Nexus Ventures" name
            ghost_count = storage.session.query(ContentItemRecord).join(
                ContentSourceRecord,
                ContentItemRecord.content_source_id == ContentSourceRecord.id
            ).filter(
                ContentSourceRecord.name.like('%ghost%') |
                ContentSourceRecord.name.like('%Idea Nexus%') |
                ContentSourceRecord.name.like('%Ventures%')
            ).count()
            if ghost_count > 0:
                return ghost_count
        
        return 0
    except Exception:
        return 0


def _process_youtube_source(source: ContentSource, cfg: CLIConfig, output: str, collection: str, openai_key: Optional[str]):
    """Process a YouTube source."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
    
    # Convert -1 to None for unlimited
    max_videos = None if source.max_items == -1 else source.max_items
    
    # Initialize storage
    storage = SQLiteStorage(db_path=output)
    
    # YouTube client
    youtube_client = YouTubeClient(
        youtube_api_key=cfg.get_youtube_api_key(),
        whisper_model=cfg.get_whisper_model(),
        whisper_device=cfg.get_whisper_device(),
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
    
    # Semantic extractor (for decomposing long content)
    semantic_extractor = None
    if openai_key:
        from idearank.semantic_extractor import SemanticExtractor
        semantic_extractor = SemanticExtractor(api_key=openai_key, model="gpt-5-nano")
    
    # Pipeline
    idearank_config = IdeaRankConfig.default()
    idearank_pipeline = IdeaRankPipeline(
        config=idearank_config,
        embedding_provider=embedding_provider,
        topic_provider=DummyTopicModelProvider(),
        neighborhood_provider=chroma_provider,
        storage=storage,  # Pass storage for chunk/semantic persistence
        semantic_extractor=semantic_extractor,  # For decomposing long content
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
        max_videos=max_videos,
    )
    
    storage.close()


def _process_ghost_export_source(source: ContentSource, output: str, collection: str, openai_key: Optional[str]):
    """Process a Ghost export source."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
    from idearank.integrations.ghost_export import GhostExportClient
    
    # Convert -1 to None for unlimited
    max_posts = None if source.max_items == -1 else source.max_items
    
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
    
    # Semantic extractor (for decomposing long content)
    semantic_extractor = None
    if openai_key:
        from idearank.semantic_extractor import SemanticExtractor
        semantic_extractor = SemanticExtractor(api_key=openai_key, model="gpt-5-nano")
    
    # Pipeline
    idearank_config = IdeaRankConfig.default()
    idearank_pipeline = IdeaRankPipeline(
        config=idearank_config,
        embedding_provider=embedding_provider,
        topic_provider=DummyTopicModelProvider(),
        neighborhood_provider=chroma_provider,
        storage=storage,  # Pass storage for chunk/semantic persistence
        semantic_extractor=semantic_extractor,  # For decomposing long content
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
        max_posts=max_posts,
        filter_query=source.filter_query,
    )
    
    storage.close()


def _process_twitter_source(source: ContentSource, output: str, collection: str, openai_key: Optional[str]):
    """Process a Twitter source from local JSON file."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider, OpenAIEmbeddingProvider
    from idearank.providers.topics import LDATopicModelProvider, DummyTopicModelProvider
    from idearank.pipelines.twitter_pipeline import process_twitter_archive
    from idearank.integrations.twitter import load_twitter_archive
    
    # Convert -1 to None for unlimited
    limit = None if source.max_items == -1 else source.max_items
    
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
    
    # Semantic extractor (for decomposing content)
    semantic_extractor = None
    if openai_key:
        from idearank.semantic_extractor import SemanticExtractor
        semantic_extractor = SemanticExtractor(api_key=openai_key, model="gpt-5-nano")
    
    # Load the Twitter archive from JSON file
    file_path = source.url_or_path
    rprint(f"[blue]Loading Twitter archive from: {file_path}...[/blue]")
    
    try:
        archive = load_twitter_archive(file_path)
    except FileNotFoundError:
        rprint(f"[red]❌ File not found: {file_path}[/red]")
        return
    except ValueError as e:
        rprint(f"[red]❌ Invalid archive format: {e}[/red]")
        return
    except Exception as e:
        rprint(f"[red]❌ Error loading archive: {e}[/red]")
        return
    
    rprint(f"[green]✅ Loaded {len(archive.posts)} tweets from @{archive.username}[/green]")
    
    # Process Twitter archive
    rprint(f"[blue]Processing Twitter archive for @{archive.username}...[/blue]")
    
    results = process_twitter_archive(
        archive=archive,
        storage=storage,
        embedding_provider=embedding_provider,
        topic_provider=topic_provider,
        chroma_provider=chroma_provider,
        limit=limit,
        semantic_extractor=semantic_extractor,
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
        rprint(f"[red]✗[/red] {results.get('error', 'Unknown error')}")
    
    storage.close()


def _process_medium_source(source: ContentSource, output: str, collection: str, openai_key: Optional[str]):
    """Process a Medium archive source."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider, OpenAIEmbeddingProvider
    from idearank.providers.topics import LDATopicModelProvider, DummyTopicModelProvider
    from idearank.pipelines.medium_pipeline import process_medium_archive
    
    # Convert -1 to None for unlimited
    limit = None if source.max_items == -1 else source.max_items
    
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
    
    # Semantic extractor (for decomposing content)
    semantic_extractor = None
    if openai_key:
        from idearank.semantic_extractor import SemanticExtractor
        semantic_extractor = SemanticExtractor(api_key=openai_key, model="gpt-5-nano")
    
    # Process archive
    content_source, stats = process_medium_archive(
        archive_path=source.url_or_path,
        storage=storage,
        embedding_provider=embedding_provider,
        topic_provider=topic_provider,
        neighborhood_provider=chroma_provider,
        limit=limit,
        skip_drafts=True,
        semantic_extractor=semantic_extractor,
    )
    
    rprint(f"[green]✓[/green] Processed {stats['total_posts']} posts")
    rprint(f"  Average score: {stats['avg_score']:.4f}")
    rprint(f"  Score range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
    rprint(f"  Total claps: {stats['total_claps']:,}")
    
    storage.close()


def _process_ghost_api_source(source: ContentSource, cfg: CLIConfig, output: str, collection: str, openai_key: Optional[str]):
    """Process a Ghost API source."""
    from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
    
    # Convert -1 to None for unlimited
    max_posts = None if source.max_items == -1 else source.max_items
    
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
    
    # Semantic extractor (for decomposing long content)
    semantic_extractor = None
    if openai_key:
        from idearank.semantic_extractor import SemanticExtractor
        semantic_extractor = SemanticExtractor(api_key=openai_key, model="gpt-5-nano")
    
    # Pipeline
    idearank_config = IdeaRankConfig.default()
    idearank_pipeline = IdeaRankPipeline(
        config=idearank_config,
        embedding_provider=embedding_provider,
        topic_provider=DummyTopicModelProvider(),
        neighborhood_provider=chroma_provider,
        storage=storage,  # Pass storage for chunk/semantic persistence
        semantic_extractor=semantic_extractor,  # For decomposing long content
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
        max_posts=max_posts,
        filter_query=source.filter_query,
    )
    
    storage.close()


@main.command()
@click.argument("channel_url")
@click.option("--max-videos", default=50, type=int, help="Maximum number of videos to process")
@click.option("--all", "process_all_videos", is_flag=True, help="Process ALL videos (ignores --max-videos)")
@click.option("--output", default="idearank_results.db", help="SQLite database output path")
@click.option("--collection", default=None, help="Chroma collection name (default: auto-generated)")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for embeddings")
def process(
    channel_url: str,
    max_videos: int,
    process_all_videos: bool,
    output: str,
    collection: Optional[str],
    openai_key: Optional[str],
):
    """Process a YouTube channel and compute IdeaRank scores.
    
    Example:
        idearank process https://youtube.com/@channelname --max-videos 20
        idearank process https://youtube.com/@channelname --all
    """
    # Load config
    cfg = CLIConfig()
    is_configured, missing = cfg.is_configured()
    
    if not is_configured:
        rprint(f"[red]✗[/red] Missing configuration: {', '.join(missing)}")
        rprint("\nRun [cyan]idearank config show[/cyan] to see current configuration")
        raise click.Abort()
    
    # Determine actual max_videos value
    actual_max_videos = None if process_all_videos else max_videos
    
    # Show configuration summary
    console.print(Panel.fit(
        f"[bold]Processing YouTube Channel[/bold]\n\n"
        f"Channel: {channel_url}\n"
        f"Max Videos: {'ALL (pagination enabled)' if process_all_videos else max_videos}\n"
        f"Output: {output}\n"
        f"Chroma Mode: {cfg.get_chroma_mode()}\n"
        f"Whisper Model: {cfg.get_whisper_model()}",
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
        
        youtube_client = YouTubeClient(
            youtube_api_key=youtube_api_key,
            whisper_model=cfg.get_whisper_model(),
            whisper_device=cfg.get_whisper_device(),
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
        if process_all_videos:
            rprint("[yellow]⚠[/yellow]  Processing ALL videos - this may take a while!")
        content_source, scores = youtube_pipeline.process_channel(
            channel_url=channel_url,
            max_videos=actual_max_videos,
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
@click.option("--max-posts", default=50, type=int, help="Maximum number of posts to process")
@click.option("--all", "process_all_posts", is_flag=True, help="Process ALL posts (ignores --max-posts)")
@click.option("--output", default="idearank_results.db", help="SQLite database output path")
@click.option("--collection", default=None, help="Chroma collection name (default: auto-generated)")
@click.option("--filter", "filter_query", default=None, help="Filter: 'tag:NAME' or 'author:NAME'")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for embeddings")
def process_ghost_export(
    export_file: str,
    max_posts: int,
    process_all_posts: bool,
    output: str,
    collection: Optional[str],
    filter_query: Optional[str],
    openai_key: Optional[str],
):
    """Process a Ghost export file and compute IdeaRank scores.
    
    Export your Ghost blog: Settings → Labs → Export
    
    Example:
        idearank process-ghost-export my-blog.ghost.2024-10-09.json --max-posts 20
        idearank process-ghost-export my-blog.ghost.2024-10-09.json --all
    """
    # Determine actual max_posts value
    actual_max_posts = None if process_all_posts else max_posts
    
    # Show configuration summary
    console.print(Panel.fit(
        f"[bold]Processing Ghost Export File[/bold]\n\n"
        f"Export File: {export_file}\n"
        f"Max Posts: {'ALL' if process_all_posts else max_posts}\n"
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
        if process_all_posts:
            rprint("[yellow]⚠[/yellow]  Processing ALL posts - this may take a while!")
        content_source, scores = ghost_pipeline.process_blog(
            blog_url=ghost_client.blog_url,
            max_posts=actual_max_posts,
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


@main.command("process-medium")
@click.argument("archive_file", type=click.Path(exists=True))
@click.option("--max-posts", default=50, type=int, help="Maximum number of posts to process")
@click.option("--all", "process_all_posts", is_flag=True, help="Process ALL posts (ignores --max-posts)")
@click.option("--skip-drafts/--include-drafts", default=True, help="Skip draft posts")
@click.option("--output", default="idearank_results.db", help="SQLite database output path")
@click.option("--collection", default=None, help="Chroma collection name (default: auto-generated)")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for embeddings")
def process_medium(
    archive_file: str,
    max_posts: int,
    process_all_posts: bool,
    skip_drafts: bool,
    output: str,
    collection: Optional[str],
    openai_key: Optional[str],
):
    """Process a Medium archive ZIP file and compute IdeaRank scores.
    
    Export your Medium archive: Settings → Security and apps → Download your information
    
    Note: Comments and short responses are automatically filtered out to focus on
    substantive content. Use --include-drafts to process draft posts.
    
    Example:
        idearank process-medium medium-export.zip --max-posts 20
        idearank process-medium medium-export.zip --all
    """
    # Determine actual max_posts value
    actual_max_posts = None if process_all_posts else max_posts
    
    # Show configuration summary
    console.print(Panel.fit(
        f"[bold]Processing Medium Archive[/bold]\n\n"
        f"Archive File: {archive_file}\n"
        f"Max Posts: {'ALL' if process_all_posts else max_posts}\n"
        f"Skip Drafts: {skip_drafts}\n"
        f"Output: {output}",
        border_style="blue"
    ))
    
    try:
        # Initialize components
        rprint("\n[bold]Initializing components...[/bold]")
        
        # Storage
        storage = SQLiteStorage(db_path=output)
        rprint(f"[green]✓[/green] SQLite storage ready: {output}")
        
        # Chroma provider (use local)
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
            try:
                from idearank.providers.embeddings import SentenceTransformerEmbeddingProvider
                embedding_provider = SentenceTransformerEmbeddingProvider(
                    model_name="all-MiniLM-L6-v2"
                )
                rprint("[green]✓[/green] Sentence-transformers embedding provider ready (all-MiniLM-L6-v2)")
            except ImportError:
                from idearank.providers.embeddings import DummyEmbeddingProvider
                embedding_provider = DummyEmbeddingProvider(dimension=384)
                rprint("[yellow]⚠[/yellow]  Using dummy embeddings (install sentence-transformers for better quality)")
        
        # Topic provider
        try:
            topic_provider = LDATopicModelProvider(num_topics=30)
            rprint("[green]✓[/green] LDA topic model ready (30 topics)")
        except ImportError:
            topic_provider = DummyTopicModelProvider()
            rprint("[yellow]⚠[/yellow]  Using dummy topic model (install scikit-learn for better cohesion scores)")
        
        # Process Medium archive
        from idearank.pipelines.medium_pipeline import process_medium_archive
        
        rprint("\n[bold]Processing Medium archive...[/bold]")
        if process_all_posts:
            rprint("[yellow]⚠[/yellow]  Processing ALL posts - this may take a while!")
        content_source, stats = process_medium_archive(
            archive_path=archive_file,
            storage=storage,
            embedding_provider=embedding_provider,
            topic_provider=topic_provider,
            neighborhood_provider=chroma_provider,
            limit=actual_max_posts,
            skip_drafts=skip_drafts,
        )
        
        # Display results
        rprint("\n[bold green]✓ Processing complete![/bold green]")
        
        # Show statistics
        rprint(f"\n[bold]Medium Blog:[/bold] {content_source.name}")
        rprint(f"  Total posts processed: {stats['total_posts']}")
        rprint(f"  Average IdeaRank score: {stats['avg_score']:.4f}")
        rprint(f"  Score range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
        rprint(f"  Total claps: {stats['total_claps']:,}")
        rprint(f"  Average word count: {stats['avg_word_count']:.0f}")
        
        if stats['date_range']['earliest']:
            rprint(f"  Date range: {stats['date_range']['earliest'].strftime('%Y-%m-%d')} to {stats['date_range']['latest'].strftime('%Y-%m-%d')}")
        
        rprint(f"\n[bold]Results saved to:[/bold] {output}")
        rprint(f"[bold]Chroma collection:[/bold] {collection or f'idearank_{Path(output).stem}'}")
        
        # Show top posts
        rprint("\n[bold]Top 10 Posts by IdeaRank:[/bold]")
        
        # Query database for top posts
        import sqlite3
        conn = sqlite3.connect(output)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.title, s.score, s.uniqueness_score, s.cohesion_score, 
                   s.learning_score, s.quality_score, s.trust_score
            FROM content_items c
            JOIN idearank_scores s ON c.id = s.content_item_id
            WHERE c.content_source_id = ?
            ORDER BY s.score DESC
            LIMIT 10
        """, (content_source.id,))
        
        table = Table(title=f"Top Posts - {content_source.name}")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Title", style="white", width=50)
        table.add_column("Score", style="green", width=8)
        table.add_column("U", style="blue", width=6)
        table.add_column("C", style="blue", width=6)
        table.add_column("L", style="blue", width=6)
        table.add_column("Q", style="blue", width=6)
        table.add_column("T", style="blue", width=6)
        
        for idx, row in enumerate(cursor.fetchall(), 1):
            title, score, u, c, l, q, t = row
            table.add_row(
                str(idx),
                title[:47] + "..." if len(title) > 50 else title,
                f"{score:.4f}",
                f"{u:.3f}",
                f"{c:.3f}",
                f"{l:.3f}",
                f"{q:.3f}",
                f"{t:.3f}",
            )
        
        conn.close()
        console.print(table)
        
    except Exception as e:
        rprint(f"\n[red]✗ Error:[/red] {e}")
        raise


@main.command("process-ghost")
@click.argument("blog_url")
@click.option("--max-posts", default=50, type=int, help="Maximum number of posts to process")
@click.option("--all", "process_all_posts", is_flag=True, help="Process ALL posts (ignores --max-posts)")
@click.option("--output", default="idearank_results.db", help="SQLite database output path")
@click.option("--collection", default=None, help="Chroma collection name (default: auto-generated)")
@click.option("--api-key", envvar="GHOST_API_KEY", help="Ghost Content API key (PREMIUM ONLY)")
@click.option("--filter", "filter_query", default=None, help="Ghost filter query (e.g., 'tag:python')")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key for embeddings")
def process_ghost(
    blog_url: str,
    max_posts: int,
    process_all_posts: bool,
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
        idearank process-ghost https://blog.example.com --api-key YOUR_KEY --all
    """
    # Determine actual max_posts value
    actual_max_posts = None if process_all_posts else max_posts
    
    # Show configuration summary
    console.print(Panel.fit(
        f"[bold]Processing Ghost Blog[/bold]\n\n"
        f"Blog URL: {blog_url}\n"
        f"Max Posts: {'ALL' if process_all_posts else max_posts}\n"
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
        if process_all_posts:
            rprint("[yellow]⚠[/yellow]  Processing ALL posts - this may take a while!")
        content_source, scores = ghost_pipeline.process_blog(
            blog_url=blog_url,
            max_posts=actual_max_posts,
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
    
    # Whisper Configuration (optional)
    rprint("\n[bold]Step 2: Whisper Transcription (Optional)[/bold]")
    rprint("Configure local Whisper transcription for videos without captions.")
    rprint("Models: tiny (fastest) < base < small (default) < medium < large (best)")
    
    whisper_model = click.prompt(
        "Whisper Model", 
        type=click.Choice(["tiny", "base", "small", "medium", "large"]),
        default="small",
        show_default=True
    )
    cfg.set_whisper_model(whisper_model)
    rprint(f"[green]✓[/green] Whisper model set to: {whisper_model}")
    
    whisper_device = click.prompt(
        "Whisper Device",
        type=click.Choice(["cpu", "cuda", "auto"]),
        default="auto",
        show_default=True
    )
    cfg.set_whisper_device(whisper_device)
    rprint(f"[green]✓[/green] Whisper device set to: {whisper_device}")
    
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

