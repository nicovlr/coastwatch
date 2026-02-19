"""Pydantic models for configuration validation."""

from __future__ import annotations

from pydantic import BaseModel, HttpUrl


class WebcamConfig(BaseModel):
    snapshot_url: str
    type: str = "snapshot"  # snapshot | mjpeg | hls
    refresh_interval_sec: int = 300
    headers: dict[str, str] = {}
    fallback_urls: list[str] = []


class Coordinates(BaseModel):
    latitude: float
    longitude: float


class BeachMetadata(BaseModel):
    orientation: str = "west"
    elevation_m: float = 0
    fov_degrees: float = 90
    timezone: str = "Europe/Paris"
    surf_spot: bool = True


class BeachConfig(BaseModel):
    id: str
    name: str
    region: str
    coordinates: Coordinates
    webcam: WebcamConfig
    metadata: BeachMetadata = BeachMetadata()


class CaptureSettings(BaseModel):
    default_interval_sec: int = 300
    max_concurrent_captures: int = 4
    request_timeout_sec: float = 15.0
    max_retries: int = 3
    retry_backoff_sec: float = 5.0


class OpenCVSettings(BaseModel):
    enabled: bool = True
    crowd_blob_min_area: int = 200
    crowd_blob_max_area: int = 5000
    crowd_min_circularity: float = 0.3
    wave_canny_threshold_low: int = 50
    wave_canny_threshold_high: int = 150
    wave_min_contour_length: int = 100
    sky_region_ratio: float = 0.35
    brightness_sunny_threshold: int = 170
    brightness_overcast_threshold: int = 100
    blue_channel_clear_sky_min: int = 140


class ClaudeSettings(BaseModel):
    enabled: bool = True
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.1
    rate_limit_rpm: int = 30
    rate_limit_daily: int = 500


class StorageSettings(BaseModel):
    database_path: str = "~/.coastwatch/coastwatch.db"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str = "~/.coastwatch/coastwatch.log"


class AppSettings(BaseModel):
    capture: CaptureSettings = CaptureSettings()
    opencv: OpenCVSettings = OpenCVSettings()
    claude: ClaudeSettings = ClaudeSettings()
    storage: StorageSettings = StorageSettings()
    logging: LoggingSettings = LoggingSettings()
