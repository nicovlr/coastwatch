"""Show current conditions for a beach."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@click.command()
@click.argument("beach_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx: click.Context, beach_id: str, as_json: bool) -> None:
    """Show current conditions for a specific beach."""
    repo = ctx.obj["repo"]
    beaches = ctx.obj["beaches"]
    console = Console()

    # Find beach config
    beach = next((b for b in beaches if b.id == beach_id), None)
    if not beach:
        console.print(f"[red]Unknown beach: {beach_id}[/red]")
        console.print("Run 'coastwatch beaches' to see available beaches.")
        raise SystemExit(1)

    obs = repo.get_latest(beach_id)
    if not obs:
        console.print(f"[yellow]No data yet for {beach.name}. Run 'coastwatch capture --once' first.[/yellow]")
        raise SystemExit(0)

    if as_json:
        data = {
            "beach": beach_id,
            "name": beach.name,
            "captured_at": obs.captured_at,
            "crowd": {"cv_level": obs.cv_crowd_level, "ai_level": obs.ai_crowd_level,
                       "ai_count": obs.ai_crowd_count, "ai_notes": obs.ai_crowd_notes},
            "waves": {"cv_level": obs.cv_wave_level, "ai_size": obs.ai_wave_size,
                       "ai_quality": obs.ai_wave_quality, "ai_notes": obs.ai_wave_notes},
            "weather": {"cv_condition": obs.cv_weather_condition, "ai_condition": obs.ai_weather_condition,
                         "ai_wind": obs.ai_wind_estimate, "ai_notes": obs.ai_weather_notes},
            "scores": {"beach": obs.ai_beach_score, "surf": obs.ai_surf_score},
            "summary": obs.ai_summary,
            "best_for": obs.ai_best_for,
        }
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # Rich formatted output
    # Calculate age
    try:
        captured = datetime.fromisoformat(obs.captured_at.replace("Z", "+00:00"))
        age_min = int((datetime.now(timezone.utc) - captured).total_seconds() / 60)
        age_str = f"{age_min} min ago" if age_min < 60 else f"{age_min // 60}h{age_min % 60:02d} ago"
    except Exception:
        age_str = "unknown"

    # Use AI data if available, fall back to OpenCV
    crowd = obs.ai_crowd_level or obs.cv_crowd_level or "?"
    crowd_detail = ""
    if obs.ai_crowd_count is not None:
        crowd_detail = f" (est. ~{obs.ai_crowd_count} people"
        if obs.ai_crowd_distribution:
            crowd_detail += f", {obs.ai_crowd_distribution}"
        crowd_detail += ")"

    waves = obs.ai_wave_size or obs.cv_wave_level or "?"
    wave_detail = ""
    if obs.ai_wave_quality:
        wave_detail = f", {obs.ai_wave_quality} quality"
    if obs.ai_wave_type:
        wave_detail += f", {obs.ai_wave_type}"

    weather = obs.ai_weather_condition or obs.cv_weather_condition or "?"
    weather_detail = ""
    if obs.ai_wind_estimate:
        weather_detail = f", {obs.ai_wind_estimate} wind"
    if obs.ai_visibility:
        weather_detail += f", {obs.ai_visibility} visibility"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold", min_width=10)
    table.add_column("Value")

    table.add_row("Crowd", f"{crowd}{crowd_detail}")
    table.add_row("Waves", f"{waves}{wave_detail}")
    table.add_row("Weather", f"{weather}{weather_detail}")

    if obs.ai_beach_score is not None:
        table.add_row("Beach", f"{obs.ai_beach_score:.1f}/10")
    if obs.ai_surf_score is not None:
        table.add_row("Surf", f"{obs.ai_surf_score:.1f}/10")
    if obs.ai_best_for:
        table.add_row("Best for", ", ".join(obs.ai_best_for))

    title = f"{beach.name}"
    subtitle = f"Last updated: {obs.captured_at[:19]} ({age_str})"

    panel = Panel(table, title=title, subtitle=subtitle, border_style="blue")
    console.print(panel)

    if obs.ai_summary:
        console.print(f"\n[italic]{obs.ai_summary}[/italic]")

    if obs.error_message:
        console.print(f"\n[dim yellow]Note: {obs.error_message}[/dim yellow]")
