"""BAYWATCH CLI entrypoint."""

from __future__ import annotations

import logging

import click

# Suppress noisy httpx request logs in normal mode
logging.getLogger("httpx").setLevel(logging.WARNING)

from coastwatch.config.loader import load_beaches, load_settings, resolve_path
from coastwatch.storage.database import Database
from coastwatch.storage.repository import ObservationRepository


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_context(config_path: str | None, settings_path: str | None) -> dict:
    """Load config and initialize all dependencies."""
    settings = load_settings(settings_path)
    beaches = load_beaches(config_path)
    db_path = resolve_path(settings.storage.database_path)
    db = Database(db_path)
    db.ensure_schema()
    repo = ObservationRepository(db)
    repo.sync_beaches(beaches)
    return {
        "settings": settings,
        "beaches": beaches,
        "db": db,
        "repo": repo,
    }


@click.group()
@click.option("--config", "-c", default=None, help="Path to beaches.yaml config")
@click.option("--settings", "-s", default=None, help="Path to settings.yaml config")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config: str | None, settings: str | None, verbose: bool) -> None:
    """BAYWATCH — Beach conditions monitoring for the French Atlantic coast."""
    setup_logging("DEBUG" if verbose else "INFO")
    ctx.ensure_object(dict)
    ctx.obj.update(build_context(config, settings))


# Import and register commands
from coastwatch.cli.commands.beaches import beaches  # noqa: E402
from coastwatch.cli.commands.best import best  # noqa: E402
from coastwatch.cli.commands.capture import capture  # noqa: E402
from coastwatch.cli.commands.history import history  # noqa: E402
from coastwatch.cli.commands.export import export  # noqa: E402
from coastwatch.cli.commands.status import status  # noqa: E402
from coastwatch.cli.commands.train import train  # noqa: E402

cli.add_command(beaches)
cli.add_command(best)
cli.add_command(capture)
cli.add_command(export)
cli.add_command(history)
cli.add_command(status)
cli.add_command(train)
