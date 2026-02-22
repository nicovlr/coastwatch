"""Image analysis for beach webcam frames: waves and camera status."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from coastwatch.analysis.models import (
    CameraStatus,
    ImageQuality,
    LocalAnalysisResult,
    WaveEstimate,
)
from coastwatch.common.solar import is_daylight
from coastwatch.config.models import CameraSettings, OpenCVSettings

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Analyzes beach webcam frames: wave detection and camera status."""

    def __init__(
        self,
        opencv_settings: OpenCVSettings | None = None,
        camera_settings: CameraSettings | None = None,
    ):
        self._s = opencv_settings or OpenCVSettings()
        self._cam = camera_settings or CameraSettings()

    def analyze(
        self,
        image_bytes: bytes,
        latitude: float = 0.0,
        longitude: float = 0.0,
        timezone: str = "Europe/Paris",
    ) -> LocalAnalysisResult:
        """Run wave analysis and camera status detection on a single frame."""
        img = self._decode_image(image_bytes)
        if img is None:
            return LocalAnalysisResult(
                image_quality=ImageQuality(is_usable=False, quality_score=0.0, issues=["decode_failed"]),
                camera_status=CameraStatus(status="offline", reason="Image decode failed"),
            )

        quality = self._assess_quality(img)
        camera_status = self._detect_camera_status(img, quality, latitude, longitude, timezone)

        if camera_status.status != "working":
            return LocalAnalysisResult(
                image_quality=quality,
                camera_status=camera_status,
            )

        return LocalAnalysisResult(
            waves=self._analyze_waves(img),
            image_quality=quality,
            camera_status=camera_status,
        )

    def _decode_image(self, image_bytes: bytes) -> np.ndarray | None:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _detect_camera_status(
        self,
        img: np.ndarray,
        quality: ImageQuality,
        latitude: float,
        longitude: float,
        timezone: str,
    ) -> CameraStatus:
        """Determine camera status: working, night, offline, or obstructed."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        std_dev = float(np.std(gray))

        # Very dark image
        if mean_brightness < self._cam.brightness_night_threshold:
            if self._cam.use_solar and latitude != 0.0:
                if not is_daylight(latitude, longitude, timezone):
                    return CameraStatus(status="night", reason="Dark image during nighttime")
                else:
                    return CameraStatus(status="offline", reason="Dark image during daytime — camera likely offline")
            return CameraStatus(status="night", reason="Very dark image")

        # Solid color (lens cap, frozen frame, etc.)
        if std_dev < self._cam.solid_color_std_threshold:
            return CameraStatus(status="obstructed", reason=f"Uniform image (std={std_dev:.1f})")

        # Extremely low brightness but not pitch black
        if mean_brightness < self._cam.brightness_offline_threshold:
            return CameraStatus(status="offline", reason=f"Very low brightness ({mean_brightness:.0f})")

        return CameraStatus(status="working", reason="")

    def _analyze_waves(self, img: np.ndarray) -> WaveEstimate:
        """Wave analysis via edge detection and whitecap segmentation."""
        h, w = img.shape[:2]
        water_region = img[int(h * 0.4):, :]

        # Whitecap detection (white foam on water)
        hsv = cv2.cvtColor(water_region, cv2.COLOR_BGR2HSV)
        whitecap_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
        total_pixels = whitecap_mask.size
        whitecap_pixels = cv2.countNonZero(whitecap_mask)
        whitecap_ratio = whitecap_pixels / total_pixels if total_pixels > 0 else 0

        # Edge detection for wave structure
        gray_water = cv2.cvtColor(water_region, cv2.COLOR_BGR2GRAY)
        gray_water = cv2.GaussianBlur(gray_water, (5, 5), 0)
        edges = cv2.Canny(
            gray_water,
            self._s.wave_canny_threshold_low,
            self._s.wave_canny_threshold_high,
        )
        edge_pixels = cv2.countNonZero(edges)
        edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0

        wave_metric = whitecap_ratio * 0.6 + edge_density * 0.4

        if wave_metric < 0.01:
            level = "flat"
        elif wave_metric < 0.03:
            level = "small"
        elif wave_metric < 0.06:
            level = "medium"
        elif wave_metric < 0.10:
            level = "large"
        else:
            level = "heavy"

        confidence = min(0.8, 0.4 + wave_metric * 5)

        return WaveEstimate(
            wave_level=level,
            whitecap_ratio=round(whitecap_ratio, 4),
            edge_density=round(edge_density, 4),
            confidence=round(confidence, 2),
        )

    def _assess_quality(self, img: np.ndarray) -> ImageQuality:
        """Assess if the image is usable for analysis."""
        issues: list[str] = []
        h, w = img.shape[:2]

        if h < 100 or w < 100:
            issues.append("too_small")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 20:
            issues.append("too_dark")

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 10:
            issues.append("too_blurry")

        if float(np.std(gray)) < 5:
            issues.append("solid_color")

        is_usable = len(issues) == 0
        score = 1.0
        if "too_dark" in issues:
            score -= 0.4
        if "too_blurry" in issues:
            score -= 0.3
        if "solid_color" in issues:
            score -= 0.5
        if "too_small" in issues:
            score -= 0.3
        score = max(0.0, score)

        return ImageQuality(is_usable=is_usable, quality_score=round(score, 2), issues=issues)
