"""Data access layer for observation records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from coastwatch.storage.database import Database


@dataclass
class Observation:
    """A single observation record for a beach."""
    beach_id: str
    captured_at: str
    source_url: str = ""
    # Camera status
    camera_status: str | None = None
    camera_status_reason: str | None = None
    # Person detection (YOLO)
    person_count: int | None = None
    person_confidence: float | None = None
    detection_method: str | None = None
    # Local image analysis (waves)
    cv_wave_level: str | None = None
    cv_whitecap_ratio: float | None = None
    cv_edge_density: float | None = None
    cv_wave_confidence: float | None = None
    cv_image_quality: float | None = None
    # Weather API
    weather_temperature_c: float | None = None
    weather_feels_like_c: float | None = None
    weather_humidity_pct: int | None = None
    weather_wind_speed_kmh: float | None = None
    weather_wind_direction: str | None = None
    weather_wind_gust_kmh: float | None = None
    weather_condition: str | None = None
    weather_description: str | None = None
    weather_precipitation_mm: float | None = None
    weather_visibility_km: float | None = None
    weather_uv_index: float | None = None
    # Claude Vision
    ai_crowd_level: str | None = None
    ai_crowd_count: int | None = None
    ai_crowd_distribution: str | None = None
    ai_crowd_notes: str | None = None
    ai_wave_size: str | None = None
    ai_wave_quality: str | None = None
    ai_wave_type: str | None = None
    ai_wave_period: str | None = None
    ai_wave_notes: str | None = None
    ai_weather_condition: str | None = None
    ai_wind_estimate: str | None = None
    ai_wind_direction: str | None = None
    ai_visibility: str | None = None
    ai_weather_notes: str | None = None
    ai_current_danger_level: str | None = None
    ai_current_rip_detected: bool | None = None
    ai_current_indicators: list[str] = field(default_factory=list)
    ai_current_notes: str | None = None
    ai_beach_score: float | None = None
    ai_surf_score: float | None = None
    ai_summary: str | None = None
    ai_best_for: list[str] = field(default_factory=list)
    # Meta
    analysis_model: str | None = None
    processing_time_ms: int | None = None
    error_message: str | None = None
    id: int | None = None


class ObservationRepository:
    """CRUD operations for observation records."""

    def __init__(self, db: Database):
        self._db = db

    def save(self, obs: Observation) -> int:
        """Insert a new observation. Returns row id."""
        best_for_json = json.dumps(obs.ai_best_for) if obs.ai_best_for else None
        indicators_json = json.dumps(obs.ai_current_indicators) if obs.ai_current_indicators else None
        rip_int = int(obs.ai_current_rip_detected) if obs.ai_current_rip_detected is not None else None
        cursor = self._db.conn.execute(
            """INSERT INTO observations (
                beach_id, captured_at, source_url,
                camera_status, camera_status_reason,
                person_count, person_confidence, detection_method,
                cv_wave_level, cv_whitecap_ratio, cv_edge_density, cv_wave_confidence,
                cv_image_quality,
                weather_temperature_c, weather_feels_like_c, weather_humidity_pct,
                weather_wind_speed_kmh, weather_wind_direction, weather_wind_gust_kmh,
                weather_condition, weather_description, weather_precipitation_mm,
                weather_visibility_km, weather_uv_index,
                ai_crowd_level, ai_crowd_count, ai_crowd_distribution, ai_crowd_notes,
                ai_wave_size, ai_wave_quality, ai_wave_type, ai_wave_period, ai_wave_notes,
                ai_weather_condition, ai_wind_estimate, ai_wind_direction, ai_visibility, ai_weather_notes,
                ai_current_danger_level, ai_current_rip_detected, ai_current_indicators, ai_current_notes,
                ai_beach_score, ai_surf_score, ai_summary, ai_best_for,
                analysis_model, processing_time_ms, error_message
            ) VALUES (
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?
            )""",
            (
                obs.beach_id, obs.captured_at, obs.source_url,
                obs.camera_status, obs.camera_status_reason,
                obs.person_count, obs.person_confidence, obs.detection_method,
                obs.cv_wave_level, obs.cv_whitecap_ratio, obs.cv_edge_density, obs.cv_wave_confidence,
                obs.cv_image_quality,
                obs.weather_temperature_c, obs.weather_feels_like_c, obs.weather_humidity_pct,
                obs.weather_wind_speed_kmh, obs.weather_wind_direction, obs.weather_wind_gust_kmh,
                obs.weather_condition, obs.weather_description, obs.weather_precipitation_mm,
                obs.weather_visibility_km, obs.weather_uv_index,
                obs.ai_crowd_level, obs.ai_crowd_count, obs.ai_crowd_distribution, obs.ai_crowd_notes,
                obs.ai_wave_size, obs.ai_wave_quality, obs.ai_wave_type, obs.ai_wave_period, obs.ai_wave_notes,
                obs.ai_weather_condition, obs.ai_wind_estimate, obs.ai_wind_direction, obs.ai_visibility,
                obs.ai_weather_notes,
                obs.ai_current_danger_level, rip_int, indicators_json, obs.ai_current_notes,
                obs.ai_beach_score, obs.ai_surf_score, obs.ai_summary, best_for_json,
                obs.analysis_model, obs.processing_time_ms, obs.error_message,
            ),
        )
        self._db.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def _row_to_obs(self, row: dict) -> Observation:
        d = dict(row)
        best_for_raw = d.pop("ai_best_for", None)
        d["ai_best_for"] = json.loads(best_for_raw) if best_for_raw else []
        indicators_raw = d.pop("ai_current_indicators", None)
        d["ai_current_indicators"] = json.loads(indicators_raw) if indicators_raw else []
        rip_raw = d.pop("ai_current_rip_detected", None)
        d["ai_current_rip_detected"] = bool(rip_raw) if rip_raw is not None else None
        d.pop("created_at", None)
        # Handle legacy columns from v0.1.0 that no longer exist in Observation
        for legacy_col in (
            "cv_crowd_count", "cv_crowd_level", "cv_crowd_confidence",
            "cv_weather_condition", "cv_brightness", "cv_blue_ratio",
            "cv_visibility", "cv_weather_confidence",
        ):
            d.pop(legacy_col, None)
        return Observation(**d)

    def get_latest(self, beach_id: str) -> Observation | None:
        """Get most recent observation for a beach."""
        row = self._db.conn.execute(
            "SELECT * FROM observations WHERE beach_id = ? ORDER BY captured_at DESC LIMIT 1",
            (beach_id,),
        ).fetchone()
        return self._row_to_obs(row) if row else None

    def get_history(self, beach_id: str, hours: int = 24, limit: int = 100) -> list[Observation]:
        """Get observations for a beach within the last N hours."""
        rows = self._db.conn.execute(
            """SELECT * FROM observations
               WHERE beach_id = ? AND captured_at > datetime('now', ?)
               ORDER BY captured_at DESC LIMIT ?""",
            (beach_id, f"-{hours} hours", limit),
        ).fetchall()
        return [self._row_to_obs(r) for r in rows]

    def get_best_beaches(self, max_age_minutes: int = 30) -> list[Observation]:
        """Rank beaches by score from most recent observations."""
        rows = self._db.conn.execute(
            """SELECT o.* FROM observations o
               INNER JOIN (
                   SELECT beach_id, MAX(captured_at) as latest
                   FROM observations
                   WHERE captured_at > datetime('now', ?)
                   GROUP BY beach_id
               ) t ON o.beach_id = t.beach_id AND o.captured_at = t.latest
               ORDER BY COALESCE(o.ai_beach_score, 5.0) DESC""",
            (f"-{max_age_minutes} minutes",),
        ).fetchall()
        return [self._row_to_obs(r) for r in rows]

    def sync_beaches(self, beaches: list) -> None:
        """Sync beach registry from config into the beaches table."""
        for b in beaches:
            self._db.conn.execute(
                """INSERT OR REPLACE INTO beaches (id, name, region, latitude, longitude, orientation, surf_spot)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (b.id, b.name, b.region, b.coordinates.latitude, b.coordinates.longitude,
                 b.metadata.orientation, 1 if b.metadata.surf_spot else 0),
            )
        self._db.conn.commit()
