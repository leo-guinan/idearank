#!/usr/bin/env python3
"""
Quick script to help clean up IdeaRank sources.
Shows incomplete sources and provides removal commands.
"""

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    # Load sources config
    config_path = Path.home() / ".idearank" / "sources.json"
    
    if not config_path.exists():
        console.print("[red]No sources config found at ~/.idearank/sources.json[/red]")
        return
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    sources = data.get('sources', [])
    
    if not sources:
        console.print("[yellow]No sources configured[/yellow]")
        return
    
    # Create table
    table = Table(title="All Sources")
    table.add_column("ID", style="cyan", width=35)
    table.add_column("Name", style="white", width=30)
    table.add_column("Type", style="green", width=12)
    table.add_column("Items", justify="right", width=6)
    table.add_column("Status", justify="center", width=8)
    
    for source in sources:
        status = "✓" if source.get('enabled', True) else "✗"
        max_items = source.get('max_items', 50)
        max_items_display = "ALL" if max_items == -1 else str(max_items)
        
        table.add_row(
            source['id'],
            source.get('name', source['url_or_path'][:25]),
            source['type'],
            max_items_display,
            status,
        )
    
    console.print(table)
    
    # Show incomplete sources
    incomplete_sources = [s for s in sources if s.get('max_items', 50) != -1 and s.get('max_items', 50) < 1000]
    
    if incomplete_sources:
        console.print(f"\n[yellow]Found {len(incomplete_sources)} incomplete sources:[/yellow]")
        console.print("\n[bold]Removal commands:[/bold]")
        for source in incomplete_sources:
            console.print(f"  [dim]idearank source remove {source['id']}[/dim]")
        
        console.print(f"\n[bold]Or run all at once:[/bold]")
        commands = [f"idearank source remove {s['id']}" for s in incomplete_sources]
        console.print(f"  [dim]{' && '.join(commands)}[/dim]")
    else:
        console.print("\n[green]All sources are complete (processing all items)![/green]")

if __name__ == "__main__":
    main()
