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

    beach = next((b for b in beaches if b.id == beach_id), None)
    if not beach:
        console.print(f"[red]Unknown beach: {beach_id}[/red]")
        console.print("Run 'baywatch beaches' to see available beaches.")
        raise SystemExit(1)

    obs = repo.get_latest(beach_id)
    if not obs:
        console.print(f"[yellow]No data yet for {beach.name}. Run 'baywatch capture --once' first.[/yellow]")
        raise SystemExit(0)

    if as_json:
        data = {
            "beach": beach_id,
            "name": beach.name,
            "captured_at": obs.captured_at,
            "camera_status": obs.camera_status,
            "people": {"count": obs.person_count, "method": obs.detection_method},
            "crowd": {"ai_level": obs.ai_crowd_level, "ai_count": obs.ai_crowd_count,
                       "ai_notes": obs.ai_crowd_notes},
            "waves": {"cv_level": obs.cv_wave_level, "ai_size": obs.ai_wave_size,
                       "ai_quality": obs.ai_wave_quality, "ai_notes": obs.ai_wave_notes},
            "weather": {
                "temperature_c": obs.weather_temperature_c,
                "condition": obs.weather_condition,
                "wind_speed_kmh": obs.weather_wind_speed_kmh,
                "wind_direction": obs.weather_wind_direction,
                "humidity_pct": obs.weather_humidity_pct,
                "ai_condition": obs.ai_weather_condition,
            },
            "currents": {
                "danger_level": obs.ai_current_danger_level,
                "rip_detected": obs.ai_current_rip_detected,
                "indicators": obs.ai_current_indicators,
                "notes": obs.ai_current_notes,
            },
            "scores": {"beach": obs.ai_beach_score, "surf": obs.ai_surf_score},
            "summary": obs.ai_summary,
            "best_for": obs.ai_best_for,
        }
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # Calculate age
    try:
        captured = datetime.fromisoformat(obs.captured_at.replace("Z", "+00:00"))
        age_min = int((datetime.now(timezone.utc) - captured).total_seconds() / 60)
        age_str = f"{age_min} min ago" if age_min < 60 else f"{age_min // 60}h{age_min % 60:02d} ago"
    except Exception:
        age_str = "unknown"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold", min_width=12)
    table.add_column("Value")

    # Camera status
    cam_status = obs.camera_status or "unknown"
    cam_style = {"working": "green", "night": "dim", "offline": "red", "obstructed": "yellow"}.get(cam_status, "")
    table.add_row("Camera", f"[{cam_style}]{cam_status}[/{cam_style}]"
                  + (f" — {obs.camera_status_reason}" if obs.camera_status_reason else ""))

    # People count (YOLO)
    if obs.person_count is not None:
        table.add_row("People", f"{obs.person_count} detected ({obs.detection_method or 'yolo'})")

    # Crowd (AI)
    crowd = obs.ai_crowd_level
    if crowd:
        crowd_detail = crowd
        if obs.ai_crowd_count is not None:
            crowd_detail += f" (est. ~{obs.ai_crowd_count}"
            if obs.ai_crowd_distribution:
                crowd_detail += f", {obs.ai_crowd_distribution}"
            crowd_detail += ")"
        table.add_row("Crowd", crowd_detail)

    # Waves
    waves = obs.ai_wave_size or obs.cv_wave_level or "?"
    wave_detail = waves
    if obs.ai_wave_quality:
        wave_detail += f", {obs.ai_wave_quality} quality"
    if obs.ai_wave_type:
        wave_detail += f", {obs.ai_wave_type}"
    table.add_row("Waves", wave_detail)

    # Weather (API data)
    if obs.weather_temperature_c is not None:
        weather_str = f"{obs.weather_temperature_c:.1f}°C"
        if obs.weather_condition:
            weather_str += f", {obs.weather_condition}"
        if obs.weather_description:
            weather_str += f" ({obs.weather_description})"
        table.add_row("Weather", weather_str)

        wind_str = ""
        if obs.weather_wind_speed_kmh is not None:
            wind_str = f"{obs.weather_wind_speed_kmh:.0f} km/h"
            if obs.weather_wind_direction:
                wind_str += f" {obs.weather_wind_direction}"
            if obs.weather_wind_gust_kmh:
                wind_str += f" (gusts {obs.weather_wind_gust_kmh:.0f})"
        if wind_str:
            table.add_row("Wind", wind_str)

        if obs.weather_humidity_pct is not None:
            table.add_row("Humidity", f"{obs.weather_humidity_pct}%")
    elif obs.ai_weather_condition:
        weather_str = obs.ai_weather_condition
        if obs.ai_wind_estimate:
            weather_str += f", {obs.ai_wind_estimate} wind"
        table.add_row("Weather", weather_str)

    # Currents safety
    if obs.ai_current_danger_level and obs.ai_current_danger_level != "unknown":
        danger = obs.ai_current_danger_level
        danger_style = {
            "safe": "green", "low": "green", "moderate": "yellow",
            "high": "red bold", "extreme": "red bold blink",
        }.get(danger, "")
        current_str = f"[{danger_style}]{danger}[/{danger_style}]"
        if obs.ai_current_rip_detected:
            current_str += " [red bold]RIP CURRENT DETECTED[/red bold]"
        if obs.ai_current_indicators:
            current_str += f"\n  Indicators: {', '.join(obs.ai_current_indicators)}"
        table.add_row("Currents", current_str)

    # Scores
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

    if obs.ai_current_notes:
        console.print(f"\n[dim]{obs.ai_current_notes}[/dim]")

    if obs.error_message:
        console.print(f"\n[dim yellow]Note: {obs.error_message}[/dim yellow]")
