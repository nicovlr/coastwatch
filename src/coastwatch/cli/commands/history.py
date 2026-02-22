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
                "camera_status": obs.camera_status,
                "people": obs.person_count,
                "waves": obs.ai_wave_size or obs.cv_wave_level,
                "temp_c": obs.weather_temperature_c,
                "weather": obs.weather_condition or obs.ai_weather_condition,
                "current_danger": obs.ai_current_danger_level,
                "beach_score": obs.ai_beach_score,
                "surf_score": obs.ai_surf_score,
            })
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if fmt == "csv":
        console.print("captured_at,camera,people,waves,temp_c,weather,currents,beach_score,surf_score")
        for obs in observations:
            cam = obs.camera_status or ""
            people = obs.person_count if obs.person_count is not None else ""
            waves = obs.ai_wave_size or obs.cv_wave_level or ""
            temp = f"{obs.weather_temperature_c:.1f}" if obs.weather_temperature_c is not None else ""
            weather = obs.weather_condition or obs.ai_weather_condition or ""
            currents = obs.ai_current_danger_level or ""
            score = obs.ai_beach_score or ""
            surf = obs.ai_surf_score or ""
            console.print(f"{obs.captured_at},{cam},{people},{waves},{temp},{weather},{currents},{score},{surf}")
        return

    # Table format
    table = Table(title=f"BAYWATCH — {beach.name} — Last {hours}h ({len(observations)} observations)")
    table.add_column("Time", style="dim")
    table.add_column("Cam", style="dim")
    table.add_column("People", justify="right")
    table.add_column("Waves")
    table.add_column("Temp", justify="right")
    table.add_column("Weather")
    table.add_column("Currents")
    table.add_column("Score", justify="right")

    for obs in observations:
        time_str = obs.captured_at[11:16] if len(obs.captured_at) > 16 else obs.captured_at
        cam = obs.camera_status or "?"
        people = str(obs.person_count) if obs.person_count is not None else "-"
        waves = obs.ai_wave_size or obs.cv_wave_level or "?"
        temp = f"{obs.weather_temperature_c:.0f}°C" if obs.weather_temperature_c is not None else "-"
        weather = obs.weather_condition or obs.ai_weather_condition or "?"
        currents = obs.ai_current_danger_level or "-"
        score = f"{obs.ai_beach_score:.1f}" if obs.ai_beach_score else "-"
        table.add_row(time_str, cam, people, waves, temp, weather, currents, score)

    console.print(table)
