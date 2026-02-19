"""Load and validate YAML configuration files."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from coastwatch.config.models import AppSettings, BeachConfig

DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"


def load_beaches(config_path: str | Path | None = None) -> list[BeachConfig]:
    """Load beach configurations from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_DIR / "beaches.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return [BeachConfig(**beach) for beach in data.get("beaches", [])]


def load_settings(settings_path: str | Path | None = None) -> AppSettings:
    """Load application settings from YAML file."""
    path = Path(settings_path) if settings_path else DEFAULT_CONFIG_DIR / "settings.yaml"
    if not path.exists():
        return AppSettings()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AppSettings(**data)


def resolve_path(path_str: str) -> Path:
    """Expand ~ and env vars in a path string."""
    return Path(os.path.expandvars(os.path.expanduser(path_str)))
