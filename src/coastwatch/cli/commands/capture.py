"""Capture and analyze beach conditions."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console

from coastwatch.analysis.opencv_analyzer import ImageAnalyzer
from coastwatch.analysis.person_detector import PersonDetector
from coastwatch.analysis.pipeline import AnalysisPipeline
from coastwatch.analysis.vision_client import VisionClient
from coastwatch.analysis.weather_client import WeatherClient
from coastwatch.capture.grabber import FrameGrabber
from coastwatch.capture.scheduler import CaptureScheduler
from coastwatch.common.rate_limiter import TokenBucketRateLimiter


@click.command()
@click.option("--once", is_flag=True, help="Run a single capture cycle and exit")
@click.option("--beach", "-b", multiple=True, help="Specific beach IDs (default: all)")
@click.option("--no-ai", is_flag=True, help="Skip Claude Vision, only run local analysis + YOLO + weather")
@click.pass_context
def capture(ctx: click.Context, once: bool, beach: tuple[str, ...], no_ai: bool) -> None:
    """Capture and analyze beach conditions."""
    settings = ctx.obj["settings"]
    beaches = ctx.obj["beaches"]
    repo = ctx.obj["repo"]
    console = Console()

    # Build components
    grabber = FrameGrabber(
        timeout=settings.capture.request_timeout_sec,
        max_retries=settings.capture.max_retries,
        backoff=settings.capture.retry_backoff_sec,
    )
    image_analyzer = ImageAnalyzer(settings.opencv, settings.camera)

    # Person detector (YOLO)
    person_detector = None
    if settings.yolo.enabled:
        person_detector = PersonDetector(settings.yolo)

    # Weather client
    weather_client = None
    if settings.weather_api.enabled:
        weather_client = WeatherClient(settings.weather_api)

    # Vision client (Claude)
    vision_client = None
    if not no_ai and settings.claude.enabled:
        rate_limiter = TokenBucketRateLimiter(
            rpm=settings.claude.rate_limit_rpm,
            daily=settings.claude.rate_limit_daily,
        )
        vision_client = VisionClient(settings.claude, rate_limiter)

    pipeline = AnalysisPipeline(
        image_analyzer=image_analyzer,
        person_detector=person_detector,
        weather_client=weather_client,
        vision_client=vision_client,
    )
    scheduler = CaptureScheduler(
        beaches=beaches,
        grabber=grabber,
        pipeline=pipeline,
        repository=repo,
        default_interval=settings.capture.default_interval_sec,
    )

    beach_ids = list(beach) if beach else None
    components = ["YOLO" if person_detector else None,
                  "Weather API" if weather_client else None,
                  "Claude Vision" if vision_client else None]
    active = [c for c in components if c]
    mode_str = " + ".join(active) if active else "local analysis only"

    if once:
        console.print(f"[bold]Capturing[/bold] ({mode_str})...")
        successful = asyncio.run(scheduler.run_once(beach_ids=beach_ids, use_ai=not no_ai))
        console.print(f"\n[green]Done![/green] {len(successful)} beach(es) processed successfully.")
        if successful:
            console.print(f"  Processed: {', '.join(successful)}")
        failed = set(b.id for b in beaches) - set(successful)
        if beach_ids:
            failed = set(beach_ids) - set(successful)
        if failed:
            console.print(f"  [red]Failed: {', '.join(failed)}[/red]")
    else:
        console.print(f"[bold]Starting daemon[/bold] ({mode_str})...")
        console.print(f"Monitoring {len(beaches)} beaches every {settings.capture.default_interval_sec}s")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        asyncio.run(scheduler.run_daemon(use_ai=not no_ai))
