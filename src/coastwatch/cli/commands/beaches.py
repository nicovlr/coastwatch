"""List configured beaches."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table


@click.command()
@click.option("--region", "-r", default=None, help="Filter by region")
@click.pass_context
def beaches(ctx: click.Context, region: str | None) -> None:
    """List all configured beaches."""
    beach_list = ctx.obj["beaches"]

    if region:
        beach_list = [b for b in beach_list if region.lower() in b.region.lower()]

    console = Console()
    table = Table(title="BAYWATCH — Configured Beaches")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Region", style="green")
    table.add_column("Surf", justify="center")
    table.add_column("Lat/Lon", style="dim")

    for b in beach_list:
        surf = "Yes" if b.metadata.surf_spot else "No"
        coords = f"{b.coordinates.latitude:.4f}, {b.coordinates.longitude:.4f}"
        table.add_row(b.id, b.name, b.region, surf, coords)

    console.print(table)
    console.print(f"\n[dim]{len(beach_list)} beach(es) configured[/dim]")
