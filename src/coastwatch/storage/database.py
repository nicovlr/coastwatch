"""SQLite database management and schema migrations."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    beach_id        TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    source_url      TEXT NOT NULL,

    -- Camera status
    camera_status       TEXT,
    camera_status_reason TEXT,

    -- Person detection (YOLO)
    person_count        INTEGER,
    person_confidence   REAL,
    detection_method    TEXT,

    -- Local image analysis (waves only)
    cv_wave_level       TEXT,
    cv_whitecap_ratio   REAL,
    cv_edge_density     REAL,
    cv_wave_confidence  REAL,
    cv_image_quality    REAL,

    -- Weather API data
    weather_temperature_c   REAL,
    weather_feels_like_c    REAL,
    weather_humidity_pct    INTEGER,
    weather_wind_speed_kmh  REAL,
    weather_wind_direction  TEXT,
    weather_wind_gust_kmh   REAL,
    weather_condition       TEXT,
    weather_description     TEXT,
    weather_precipitation_mm REAL,
    weather_visibility_km   REAL,
    weather_uv_index        REAL,

    -- Claude Vision analysis (nullable)
    ai_crowd_level          TEXT,
    ai_crowd_count          INTEGER,
    ai_crowd_distribution   TEXT,
    ai_crowd_notes          TEXT,
    ai_wave_size            TEXT,
    ai_wave_quality         TEXT,
    ai_wave_type            TEXT,
    ai_wave_period          TEXT,
    ai_wave_notes           TEXT,
    ai_weather_condition    TEXT,
    ai_wind_estimate        TEXT,
    ai_wind_direction       TEXT,
    ai_visibility           TEXT,
    ai_weather_notes        TEXT,
    ai_current_danger_level TEXT,
    ai_current_rip_detected INTEGER,
    ai_current_indicators   TEXT,
    ai_current_notes        TEXT,
    ai_beach_score          REAL,
    ai_surf_score           REAL,
    ai_summary              TEXT,
    ai_best_for             TEXT,

    -- Metadata
    analysis_model      TEXT,
    processing_time_ms  INTEGER,
    error_message       TEXT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_obs_beach_time
    ON observations (beach_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_obs_captured_at
    ON observations (captured_at DESC);

CREATE TABLE IF NOT EXISTS beaches (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    region      TEXT NOT NULL,
    latitude    REAL NOT NULL,
    longitude   REAL NOT NULL,
    orientation TEXT,
    surf_spot   INTEGER NOT NULL DEFAULT 1,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

# ALTER TABLE statements for migrating from v0.1.0 schema
MIGRATION_COLUMNS = [
    ("camera_status", "TEXT"),
    ("camera_status_reason", "TEXT"),
    ("person_count", "INTEGER"),
    ("person_confidence", "REAL"),
    ("detection_method", "TEXT"),
    ("weather_temperature_c", "REAL"),
    ("weather_feels_like_c", "REAL"),
    ("weather_humidity_pct", "INTEGER"),
    ("weather_wind_speed_kmh", "REAL"),
    ("weather_wind_direction", "TEXT"),
    ("weather_wind_gust_kmh", "REAL"),
    ("weather_condition", "TEXT"),
    ("weather_description", "TEXT"),
    ("weather_precipitation_mm", "REAL"),
    ("weather_visibility_km", "REAL"),
    ("weather_uv_index", "REAL"),
    ("ai_current_danger_level", "TEXT"),
    ("ai_current_rip_detected", "INTEGER"),
    ("ai_current_indicators", "TEXT"),
    ("ai_current_notes", "TEXT"),
]


class Database:
    """SQLite connection manager."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def ensure_schema(self) -> None:
        """Create tables if they don't exist, then run migrations."""
        self.conn.executescript(SCHEMA_SQL)
        self._migrate()

    def _migrate(self) -> None:
        """Add new columns to existing observations table if missing."""
        existing = {
            row[1] for row in self.conn.execute("PRAGMA table_info(observations)").fetchall()
        }
        for col_name, col_type in MIGRATION_COLUMNS:
            if col_name not in existing:
                self.conn.execute(f"ALTER TABLE observations ADD COLUMN {col_name} {col_type}")
                logger.info("Migration: added column observations.%s", col_name)
        self.conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
