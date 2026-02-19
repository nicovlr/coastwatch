"""Show historical conditions for a beach."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table


@click.command()
@click.argument("beach_id")
@click.option("--hours", "-h", default=24, help="Hours of history to show")
@click.option("--format", "fmt", type=click.Choice(["table", "csv", "json"]), default="table")
@click.pass_context
def history(ctx: click.Context, beach_id: str, hours: int, fmt: str) -> None:
    """Show historical conditions for a beach."""
    repo = ctx.obj["repo"]
    beaches = ctx.obj["beaches"]
    console = Console()

    beach = next((b for b in beaches if b.id == beach_id), None)
    if not beach:
        console.print(f"[red]Unknown beach: {beach_id}[/red]")
        raise SystemExit(1)

    observations = repo.get_history(beach_id, hours=hours)
    if not observations:
        console.print(f"[yellow]No data for {beach.name} in the last {hours}h.[/yellow]")
        raise SystemExit(0)

    if fmt == "json":
        data = []
        for obs in observations:
            data.append({
                "captured_at": obs.captured_at,
                "crowd": obs.ai_crowd_level or obs.cv_crowd_level,
                "waves": obs.ai_wave_size or obs.cv_wave_level,
                "weather": obs.ai_weather_condition or obs.cv_weather_condition,
                "beach_score": obs.ai_beach_score,
                "surf_score": obs.ai_surf_score,
            })
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if fmt == "csv":
        console.print("captured_at,crowd,waves,weather,beach_score,surf_score")
        for obs in observations:
            crowd = obs.ai_crowd_level or obs.cv_crowd_level or ""
            waves = obs.ai_wave_size or obs.cv_wave_level or ""
            weather = obs.ai_weather_condition or obs.cv_weather_condition or ""
            score = obs.ai_beach_score or ""
            surf = obs.ai_surf_score or ""
            console.print(f"{obs.captured_at},{crowd},{waves},{weather},{score},{surf}")
        return

    # Table format
    table = Table(title=f"{beach.name} - Last {hours}h ({len(observations)} observations)")
    table.add_column("Time", style="dim")
    table.add_column("Crowd")
    table.add_column("Waves")
    table.add_column("Weather")
    table.add_column("Score", justify="right")

    for obs in observations:
        time_str = obs.captured_at[11:16] if len(obs.captured_at) > 16 else obs.captured_at
        crowd = obs.ai_crowd_level or obs.cv_crowd_level or "?"
        waves = obs.ai_wave_size or obs.cv_wave_level or "?"
        weather = obs.ai_weather_condition or obs.cv_weather_condition or "?"
        score = f"{obs.ai_beach_score:.1f}" if obs.ai_beach_score else "-"
        table.add_row(time_str, crowd, waves, weather, score)

    console.print(table)
