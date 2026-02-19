"""SQLite database management and schema migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    beach_id        TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    source_url      TEXT NOT NULL,

    -- OpenCV numeric features
    cv_crowd_count      INTEGER,
    cv_crowd_level      TEXT,
    cv_crowd_confidence REAL,
    cv_wave_level       TEXT,
    cv_whitecap_ratio   REAL,
    cv_edge_density     REAL,
    cv_wave_confidence  REAL,
    cv_weather_condition TEXT,
    cv_brightness       REAL,
    cv_blue_ratio       REAL,
    cv_visibility       TEXT,
    cv_weather_confidence REAL,
    cv_image_quality    REAL,

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
        """Create tables if they don't exist."""
        self.conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
