"""Custom exception hierarchy for CoastWatch."""

from __future__ import annotations


class CoastwatchError(Exception):
    """Base exception for all coastwatch errors."""


class ConfigError(CoastwatchError):
    """Invalid or missing configuration."""


class WebcamUnavailableError(CoastwatchError):
    """All URLs for a webcam failed after retries."""

    def __init__(self, beach_id: str, urls_tried: list[str], last_error: Exception | None = None):
        self.beach_id = beach_id
        self.urls_tried = urls_tried
        self.last_error = last_error
        super().__init__(f"Webcam unavailable for {beach_id}: tried {len(urls_tried)} URL(s)")


class ImageQualityError(CoastwatchError):
    """Image too poor for analysis."""


class VisionAnalysisError(CoastwatchError):
    """Claude API call failed after retries."""

    def __init__(self, beach_id: str, message: str, status_code: int | None = None):
        self.beach_id = beach_id
        self.status_code = status_code
        super().__init__(f"Vision analysis failed for {beach_id}: {message}")


class VisionParseError(CoastwatchError):
    """Claude returned non-JSON or invalid schema."""


class RateLimitError(CoastwatchError):
    """Daily API budget exhausted."""


class DatabaseError(CoastwatchError):
    """SQLite operation failed."""
