"""Pydantic models for analysis results."""

from __future__ import annotations

from pydantic import BaseModel


# --- OpenCV results ---

class CrowdEstimate(BaseModel):
    blob_count: int = 0
    crowd_level: str = "unknown"  # empty|light|moderate|crowded|packed
    confidence: float = 0.0


class WaveEstimate(BaseModel):
    wave_level: str = "unknown"  # flat|small|medium|large|heavy
    whitecap_ratio: float = 0.0
    edge_density: float = 0.0
    confidence: float = 0.0


class WeatherEstimate(BaseModel):
    condition: str = "unknown"  # sunny|partly_cloudy|overcast|rainy|stormy|foggy
    brightness: float = 0.0
    blue_ratio: float = 0.0
    visibility: str = "unknown"  # high|medium|low
    confidence: float = 0.0


class ImageQuality(BaseModel):
    is_usable: bool = True
    quality_score: float = 1.0
    issues: list[str] = []


class OpenCVResult(BaseModel):
    crowd: CrowdEstimate = CrowdEstimate()
    waves: WaveEstimate = WaveEstimate()
    weather: WeatherEstimate = WeatherEstimate()
    image_quality: ImageQuality = ImageQuality()


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


class OverallAnalysis(BaseModel):
    beach_score: float = 5.0
    surf_score: float | None = None
    summary: str = ""
    best_for: list[str] = []


class VisionAnalysis(BaseModel):
    crowd: CrowdAnalysis = CrowdAnalysis()
    waves: WaveAnalysis = WaveAnalysis()
    weather: WeatherAnalysis = WeatherAnalysis()
    overall: OverallAnalysis = OverallAnalysis()
