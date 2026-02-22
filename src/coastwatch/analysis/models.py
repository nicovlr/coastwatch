"""Pydantic models for analysis results."""

from __future__ import annotations

from pydantic import BaseModel


# --- Person detection (YOLO) ---

class PersonDetectionResult(BaseModel):
    person_count: int = 0
    confidence_avg: float = 0.0
    detection_method: str = "yolo"  # yolo | vision


# --- Local image analysis (waves + camera status) ---

class WaveEstimate(BaseModel):
    wave_level: str = "unknown"  # flat|small|medium|large|heavy
    whitecap_ratio: float = 0.0
    edge_density: float = 0.0
    confidence: float = 0.0


class ImageQuality(BaseModel):
    is_usable: bool = True
    quality_score: float = 1.0
    issues: list[str] = []


class CameraStatus(BaseModel):
    status: str = "working"  # working|night|offline|obstructed
    reason: str = ""


class LocalAnalysisResult(BaseModel):
    waves: WaveEstimate = WaveEstimate()
    image_quality: ImageQuality = ImageQuality()
    camera_status: CameraStatus = CameraStatus()


# --- Weather API data ---

class WeatherAPIData(BaseModel):
    temperature_c: float | None = None
    feels_like_c: float | None = None
    humidity_pct: int | None = None
    wind_speed_kmh: float | None = None
    wind_direction: str | None = None
    wind_gust_kmh: float | None = None
    condition: str = "unknown"  # clear|partly_cloudy|overcast|rain|storm|fog|snow
    description: str = ""
    precipitation_mm: float = 0.0
    visibility_km: float | None = None
    uv_index: float | None = None
    source: str = "openweathermap"


# --- Claude Vision results ---

class CrowdAnalysis(BaseModel):
    level: str = "unknown"
    estimated_count: int | None = None
    distribution: str = "none"
    notes: str = ""


class WaveAnalysis(BaseModel):
    size: str = "unknown"
    quality: str = "unknown"
    type: str = "unknown"
    period_estimate: str = ""
    notes: str = ""


class WeatherAnalysis(BaseModel):
    condition: str = "unknown"
    wind_estimate: str = "unknown"
    wind_direction_visual: str | None = None
    visibility: str = "unknown"
    notes: str = ""


class CurrentAnalysis(BaseModel):
    danger_level: str = "unknown"  # safe|low|moderate|high|extreme
    rip_current_detected: bool = False
    indicators: list[str] = []
    notes: str = ""


class OverallAnalysis(BaseModel):
    beach_score: float = 5.0
    surf_score: float | None = None
    summary: str = ""
    best_for: list[str] = []


class VisionAnalysis(BaseModel):
    crowd: CrowdAnalysis = CrowdAnalysis()
    waves: WaveAnalysis = WaveAnalysis()
    weather: WeatherAnalysis = WeatherAnalysis()
    currents: CurrentAnalysis = CurrentAnalysis()
    overall: OverallAnalysis = OverallAnalysis()
