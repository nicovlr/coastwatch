"""Fine-tune YOLO model on collected beach data."""

from __future__ import annotations

import shutil
from pathlib import Path

import click
from rich.console import Console

FRAMES_DIR = Path("~/.coastwatch/frames").expanduser()


@click.command()
@click.option("--epochs", default=50, help="Number of training epochs")
@click.option("--model", default="yolov8n.pt", help="Base model to fine-tune")
@click.option("--data-dir", default="~/.coastwatch/training", help="Training data directory")
@click.pass_context
def train(ctx: click.Context, epochs: int, model: str, data_dir: str) -> None:
    """Fine-tune YOLO person detector on collected beach frames.

    Workflow:
    1. Collect frames over time with 'baywatch capture' (daemon mode)
    2. Run 'baywatch train' to fine-tune the person detector
    3. The improved model is saved and used automatically

    The training uses YOLO's built-in pseudo-labeling: the current model
    generates labels on your collected frames, then trains on them.
    This improves detection on your specific beach webcams over time.
    """
    console = Console()

    data_path = Path(data_dir).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)

    images_dir = data_path / "images" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Collect all saved frames
    frame_count = 0
    if FRAMES_DIR.exists():
        for beach_dir in FRAMES_DIR.iterdir():
            if not beach_dir.is_dir():
                continue
            for frame in beach_dir.glob("*.jpg"):
                dest = images_dir / f"{beach_dir.name}_{frame.name}"
                if not dest.exists():
                    shutil.copy2(frame, dest)
                    frame_count += 1

    if frame_count == 0 and not any(images_dir.iterdir()):
        console.print("[yellow]No frames collected yet.[/yellow]")
        console.print("Run 'baywatch capture' for a while to collect training data first.")
        raise SystemExit(0)

    total_images = len(list(images_dir.glob("*.jpg")))
    console.print(f"[bold]Training data:[/bold] {total_images} frames ({frame_count} new)")

    # Step 1: Auto-label with current model
    console.print("\n[bold]Step 1/2:[/bold] Generating labels with current model...")
    labels_dir = data_path / "labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        console.print("[red]ultralytics not installed. Run: pip install ultralytics[/red]")
        raise SystemExit(1)

    base_model = YOLO(model)

    labeled = 0
    for img_path in images_dir.glob("*.jpg"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            labeled += 1
            continue

        results = base_model(str(img_path), conf=0.2, verbose=False)
        lines = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person class
                    # YOLO format: class x_center y_center width height (normalized)
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines))
        labeled += 1

    console.print(f"  Labeled {labeled} images")

    # Step 2: Write dataset.yaml
    dataset_yaml = data_path / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {data_path.resolve()}\n"
        f"train: images/train\n"
        f"val: images/train\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
    )

    # Step 3: Fine-tune
    console.print(f"\n[bold]Step 2/2:[/bold] Fine-tuning {model} for {epochs} epochs...")
    output_dir = Path("~/.coastwatch/models").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    fine_model = YOLO(model)
    fine_model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=640,
        batch=-1,  # auto batch size
        project=str(output_dir),
        name="baywatch",
        exist_ok=True,
        verbose=True,
    )

    # Copy best model
    best_pt = output_dir / "baywatch" / "weights" / "best.pt"
    if best_pt.exists():
        final_path = output_dir / "baywatch-person.pt"
        shutil.copy2(best_pt, final_path)
        console.print(f"\n[green bold]Model saved: {final_path}[/green bold]")
        console.print(f"\nTo use it, update config/settings.yaml:")
        console.print(f'  yolo:')
        console.print(f'    model: "{final_path}"')
        console.print(f"\nOr run: baywatch capture --once --no-ai")
    else:
        console.print("[red]Training completed but no best.pt found.[/red]")
