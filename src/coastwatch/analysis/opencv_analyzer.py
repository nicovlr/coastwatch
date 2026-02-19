"""OpenCV-based pre-processing for beach webcam frames."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from coastwatch.analysis.models import (
    CrowdEstimate,
    ImageQuality,
    OpenCVResult,
    WaveEstimate,
    WeatherEstimate,
)
from coastwatch.config.models import OpenCVSettings

logger = logging.getLogger(__name__)


class OpenCVAnalyzer:
    """Pre-processes beach webcam frames using OpenCV algorithms."""

    def __init__(self, settings: OpenCVSettings | None = None):
        self._s = settings or OpenCVSettings()

    def analyze(self, image_bytes: bytes) -> OpenCVResult:
        """Run all analyses on a single frame."""
        img = self._decode_image(image_bytes)
        if img is None:
            return OpenCVResult(
                image_quality=ImageQuality(is_usable=False, quality_score=0.0, issues=["decode_failed"])
            )

        quality = self._assess_quality(img)
        return OpenCVResult(
            crowd=self._estimate_crowd(img),
            waves=self._analyze_waves(img),
            weather=self._estimate_weather(img),
            image_quality=quality,
        )

    def _decode_image(self, image_bytes: bytes) -> np.ndarray | None:
        """Decode JPEG/PNG bytes to OpenCV BGR array."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def _estimate_crowd(self, img: np.ndarray) -> CrowdEstimate:
        """Crowd estimation via blob detection on the beach area."""
        h, w = img.shape[:2]
        # Focus on the middle-lower part of the image (beach/sand area)
        beach_region = img[int(h * 0.35):int(h * 0.85), :]

        gray = cv2.cvtColor(beach_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold to segment foreground from sand
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self._s.crowd_blob_min_area
        params.maxArea = self._s.crowd_blob_max_area
        params.filterByCircularity = True
        params.minCircularity = self._s.crowd_min_circularity
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh)
        count = len(keypoints)

        if count == 0:
            level = "empty"
        elif count <= 10:
            level = "light"
        elif count <= 30:
            level = "moderate"
        elif count <= 70:
            level = "crowded"
        else:
            level = "packed"

        # Confidence is lower for extreme counts (less reliable)
        confidence = min(0.7, 0.3 + count * 0.02) if count > 0 else 0.5

        return CrowdEstimate(blob_count=count, crowd_level=level, confidence=round(confidence, 2))

    def _analyze_waves(self, img: np.ndarray) -> WaveEstimate:
        """Wave analysis via edge detection and whitecap segmentation."""
        h, w = img.shape[:2]
        # Lower portion = water/wave area
        water_region = img[int(h * 0.4):, :]

        # Whitecap detection (white foam on water)
        hsv = cv2.cvtColor(water_region, cv2.COLOR_BGR2HSV)
        # White foam: low saturation, high value
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

        # Combined metric
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

    def _estimate_weather(self, img: np.ndarray) -> WeatherEstimate:
        """Weather estimation via sky region color and brightness."""
        h, w = img.shape[:2]
        sky_region = img[:int(h * self._s.sky_region_ratio), :]

        # Mean brightness
        gray_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray_sky))

        # Mean blue channel
        blue_mean = float(np.mean(sky_region[:, :, 0]))  # BGR -> B channel
        overall_mean = float(np.mean(sky_region))
        blue_ratio = blue_mean / overall_mean if overall_mean > 0 else 0

        # Visibility via Laplacian variance on full image
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())

        # Color variance in sky (uniformity)
        sky_hsv = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        sat_std = float(np.std(sky_hsv[:, :, 1]))

        # Classification
        if brightness > self._s.brightness_sunny_threshold and blue_mean > self._s.blue_channel_clear_sky_min:
            condition = "sunny"
        elif brightness > self._s.brightness_overcast_threshold:
            if sat_std > 30:
                condition = "partly_cloudy"
            else:
                condition = "overcast"
        elif brightness > 60:
            condition = "rainy"
        else:
            condition = "stormy"

        # Visibility
        if laplacian_var > 500:
            visibility = "high"
        elif laplacian_var > 100:
            visibility = "medium"
        else:
            visibility = "low"

        confidence = 0.6 if condition in ("sunny", "overcast") else 0.4

        return WeatherEstimate(
            condition=condition,
            brightness=round(brightness, 1),
            blue_ratio=round(blue_ratio, 3),
            visibility=visibility,
            confidence=confidence,
        )

    def _assess_quality(self, img: np.ndarray) -> ImageQuality:
        """Assess if the image is usable for analysis."""
        issues: list[str] = []
        h, w = img.shape[:2]

        # Resolution check
        if h < 100 or w < 100:
            issues.append("too_small")

        # Darkness check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 20:
            issues.append("too_dark")

        # Blur check (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 10:
            issues.append("too_blurry")

        # Solid color check (very low std dev)
        if float(np.std(gray)) < 5:
            issues.append("solid_color")

        is_usable = len(issues) == 0
        # Quality score based on metrics
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
