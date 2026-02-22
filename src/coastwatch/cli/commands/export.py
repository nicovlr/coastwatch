"""Export training data for model fine-tuning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import click
from rich.console import Console

FRAMES_DIR = Path("~/.coastwatch/frames").expanduser()


@click.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["yolo", "csv", "json"]), default="csv",
              help="Export format: yolo (YOLO labels), csv (flat table), json (full records)")
@click.option("--output", "-o", default="./training_data", help="Output directory")
@click.option("--beach", "-b", multiple=True, help="Filter by beach ID (default: all)")
@click.option("--hours", "-h", default=168, help="Hours of history to export (default: 7 days)")
@click.pass_context
def export(ctx: click.Context, fmt: str, output: str, beach: tuple[str, ...], hours: int) -> None:
    """Export captured data for model training and fine-tuning.

    Exports images + labels in various formats:
    - csv: flat table with all observations (good for pandas/analysis)
    - json: full observation records
    - yolo: YOLO-compatible labels (image path + person count + conditions)
    """
    repo = ctx.obj["repo"]
    beaches = ctx.obj["beaches"]
    console = Console()

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    beach_ids = list(beach) if beach else [b.id for b in beaches]
    all_obs = []
    for bid in beach_ids:
        obs_list = repo.get_history(bid, hours=hours, limit=10000)
        all_obs.extend(obs_list)

    if not all_obs:
        console.print(f"[yellow]No data found in last {hours}h.[/yellow]")
        raise SystemExit(0)

    if fmt == "csv":
        csv_path = out_dir / "observations.csv"
        fields = [
            "beach_id", "captured_at", "camera_status",
            "person_count", "detection_method",
            "cv_wave_level", "cv_whitecap_ratio",
            "weather_temperature_c", "weather_wind_speed_kmh", "weather_condition",
            "ai_crowd_level", "ai_crowd_count", "ai_wave_size", "ai_wave_quality",
            "ai_current_danger_level", "ai_current_rip_detected",
            "ai_beach_score", "ai_surf_score",
            "frame_path",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for obs in all_obs:
                frame_path = _find_frame(obs.beach_id, obs.captured_at)
                writer.writerow({
                    "beach_id": obs.beach_id,
                    "captured_at": obs.captured_at,
                    "camera_status": obs.camera_status,
                    "person_count": obs.person_count,
                    "detection_method": obs.detection_method,
                    "cv_wave_level": obs.cv_wave_level,
                    "cv_whitecap_ratio": obs.cv_whitecap_ratio,
                    "weather_temperature_c": obs.weather_temperature_c,
                    "weather_wind_speed_kmh": obs.weather_wind_speed_kmh,
                    "weather_condition": obs.weather_condition,
                    "ai_crowd_level": obs.ai_crowd_level,
                    "ai_crowd_count": obs.ai_crowd_count,
                    "ai_wave_size": obs.ai_wave_size,
                    "ai_wave_quality": obs.ai_wave_quality,
                    "ai_current_danger_level": obs.ai_current_danger_level,
                    "ai_current_rip_detected": obs.ai_current_rip_detected,
                    "ai_beach_score": obs.ai_beach_score,
                    "ai_surf_score": obs.ai_surf_score,
                    "frame_path": str(frame_path) if frame_path else "",
                })
        console.print(f"[green]Exported {len(all_obs)} observations to {csv_path}[/green]")

    elif fmt == "json":
        json_path = out_dir / "observations.json"
        records = []
        for obs in all_obs:
            frame_path = _find_frame(obs.beach_id, obs.captured_at)
            records.append({
                "beach_id": obs.beach_id,
                "captured_at": obs.captured_at,
                "camera_status": obs.camera_status,
                "person_count": obs.person_count,
                "detection_method": obs.detection_method,
                "waves": {
                    "cv_level": obs.cv_wave_level,
                    "ai_size": obs.ai_wave_size,
                    "ai_quality": obs.ai_wave_quality,
                },
                "weather": {
                    "temperature_c": obs.weather_temperature_c,
                    "wind_speed_kmh": obs.weather_wind_speed_kmh,
                    "condition": obs.weather_condition,
                },
                "currents": {
                    "danger_level": obs.ai_current_danger_level,
                    "rip_detected": obs.ai_current_rip_detected,
                    "indicators": obs.ai_current_indicators,
                },
                "scores": {
                    "beach": obs.ai_beach_score,
                    "surf": obs.ai_surf_score,
                },
                "frame_path": str(frame_path) if frame_path else None,
            })
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Exported {len(records)} observations to {json_path}[/green]")

    elif fmt == "yolo":
        # Export in a format ready for YOLO fine-tuning:
        # images/ folder with symlinks, labels/ with person count metadata
        images_dir = out_dir / "images"
        labels_dir = out_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        count = 0
        for obs in all_obs:
            frame_path = _find_frame(obs.beach_id, obs.captured_at)
            if not frame_path:
                continue

            ts = obs.captured_at[:19].replace(":", "-").replace("T", "_")
            name = f"{obs.beach_id}_{ts}"

            # Symlink image
            img_link = images_dir / f"{name}.jpg"
            if not img_link.exists():
                img_link.symlink_to(frame_path)

            # Label file: metadata for training
            label_path = labels_dir / f"{name}.txt"
            label_path.write_text(
                f"# beach={obs.beach_id} time={obs.captured_at}\n"
                f"# person_count={obs.person_count} camera={obs.camera_status}\n"
                f"# waves={obs.cv_wave_level} weather={obs.weather_condition}\n"
                f"# danger={obs.ai_current_danger_level} rip={obs.ai_current_rip_detected}\n"
                f"# beach_score={obs.ai_beach_score} surf_score={obs.ai_surf_score}\n"
            )
            count += 1

        # Dataset YAML for YOLO
        dataset_yaml = out_dir / "dataset.yaml"
        dataset_yaml.write_text(
            f"path: {out_dir.resolve()}\n"
            f"train: images\n"
            f"val: images\n"
            f"names:\n"
            f"  0: person\n"
        )
        console.print(f"[green]Exported {count} frames to {out_dir}/ (images/ + labels/ + dataset.yaml)[/green]")

    # Summary
    frame_count = sum(1 for obs in all_obs if _find_frame(obs.beach_id, obs.captured_at))
    console.print(f"\n[dim]{len(all_obs)} observations, {frame_count} with saved frames[/dim]")
    if frame_count == 0:
        console.print("[yellow]No frames found. Run 'baywatch capture --once' to start collecting images.[/yellow]")
    console.print(f"\n[bold]Training tips:[/bold]")
    console.print("  1. Collect data over several days/weeks with 'baywatch capture' (daemon mode)")
    console.print("  2. Export with 'baywatch export --format yolo -o ./dataset'")
    console.print("  3. Fine-tune: yolo detect train data=./dataset/dataset.yaml model=yolov8n.pt epochs=50")


def _find_frame(beach_id: str, captured_at: str) -> Path | None:
    """Find saved frame file for an observation."""
    ts = captured_at[:19].replace(":", "-").replace("T", "_")
    path = FRAMES_DIR / beach_id / f"{ts}.jpg"
    return path if path.exists() else None
