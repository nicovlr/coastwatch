"""Rank beaches by current conditions."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table


@click.command()
@click.option("--activity", "-a",
              type=click.Choice(["surfing", "swimming", "walking", "bodyboarding", "any"]),
              default="any", help="Filter by activity")
@click.option("--max-age", default=60, help="Max age of data in minutes")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def best(ctx: click.Context, activity: str, max_age: int, as_json: bool) -> None:
    """Rank beaches by current conditions."""
    repo = ctx.obj["repo"]
    beaches = ctx.obj["beaches"]
    console = Console()

    observations = repo.get_best_beaches(max_age_minutes=max_age)

    if activity != "any":
        observations = [
            o for o in observations
            if activity in (o.ai_best_for or [])
        ]

    if not observations:
        console.print(
            f"[yellow]No recent data (max {max_age} min). Run 'baywatch capture --once' first.[/yellow]"
        )
        raise SystemExit(0)

    beach_map = {b.id: b for b in beaches}

    if as_json:
        data = []
        for obs in observations:
            b = beach_map.get(obs.beach_id)
            data.append({
                "beach_id": obs.beach_id,
                "name": b.name if b else obs.beach_id,
                "beach_score": obs.ai_beach_score,
                "surf_score": obs.ai_surf_score,
                "people": obs.person_count,
                "crowd": obs.ai_crowd_level,
                "waves": obs.ai_wave_size or obs.cv_wave_level,
                "weather_temp_c": obs.weather_temperature_c,
                "weather_condition": obs.weather_condition,
                "current_danger": obs.ai_current_danger_level,
                "best_for": obs.ai_best_for,
                "summary": obs.ai_summary,
            })
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    filter_str = f" for {activity}" if activity != "any" else ""
    table = Table(title=f"BAYWATCH — Best Beaches Right Now{filter_str}")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Beach", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("People", justify="right")
    table.add_column("Waves")
    table.add_column("Weather")
    table.add_column("Safety", style="dim")
    table.add_column("Best for", style="dim")

    for i, obs in enumerate(observations, 1):
        b = beach_map.get(obs.beach_id)
        name = b.name if b else obs.beach_id
        score = f"{obs.ai_beach_score:.1f}" if obs.ai_beach_score else "-"
        people = str(obs.person_count) if obs.person_count is not None else "-"
        waves = obs.ai_wave_size or obs.cv_wave_level or "?"

        # Weather: show temperature if available
        if obs.weather_temperature_c is not None:
            weather = f"{obs.weather_temperature_c:.0f}°C {obs.weather_condition or ''}"
        else:
            weather = obs.ai_weather_condition or "?"

        # Safety column (currents)
        danger = obs.ai_current_danger_level or "-"
        if obs.ai_current_rip_detected:
            danger = f"[red bold]{danger} RIP![/red bold]"
        elif danger in ("high", "extreme"):
            danger = f"[red]{danger}[/red]"
        elif danger == "moderate":
            danger = f"[yellow]{danger}[/yellow]"

        best_for = ", ".join(obs.ai_best_for) if obs.ai_best_for else "-"

        style = "bold green" if i == 1 else None
        table.add_row(str(i), name, score, people, waves, weather, danger, best_for, style=style)

    console.print(table)

    if observations and observations[0].ai_summary:
        console.print(f"\n[bold green]#1[/bold green] {observations[0].ai_summary}")
