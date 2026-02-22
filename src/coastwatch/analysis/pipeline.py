"""Analysis pipeline orchestrator: YOLO + ImageAnalyzer + WeatherClient + VisionClient."""

from __future__ import annotations

import logging
import time

from coastwatch.analysis.models import (
    LocalAnalysisResult,
    PersonDetectionResult,
    VisionAnalysis,
    WeatherAPIData,
)
from coastwatch.analysis.opencv_analyzer import ImageAnalyzer
from coastwatch.analysis.person_detector import PersonDetector
from coastwatch.analysis.vision_client import VisionClient
from coastwatch.analysis.weather_client import WeatherClient
from coastwatch.capture.grabber import GrabbedFrame
from coastwatch.common.exceptions import RateLimitError, WeatherAPIError
from coastwatch.config.models import BeachConfig
from coastwatch.storage.repository import Observation

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates person detection, image analysis, weather, and Claude Vision."""

    def __init__(
        self,
        image_analyzer: ImageAnalyzer,
        person_detector: PersonDetector | None = None,
        weather_client: WeatherClient | None = None,
        vision_client: VisionClient | None = None,
    ):
        self._image_analyzer = image_analyzer
        self._person_detector = person_detector
        self._weather_client = weather_client
        self._vision = vision_client

    async def process_frame(
        self,
        frame: GrabbedFrame,
        beach: BeachConfig,
        use_ai: bool = True,
    ) -> Observation:
        """Full analysis pipeline for a single frame."""
        start = time.monotonic()

        # Step 1: Image analysis (waves + camera status)
        local_result = self._image_analyzer.analyze(
            frame.image_bytes,
            latitude=beach.coordinates.latitude,
            longitude=beach.coordinates.longitude,
            timezone=beach.metadata.timezone,
        )
        logger.debug(
            "%s Image: camera=%s waves=%s quality=%.1f",
            beach.id,
            local_result.camera_status.status,
            local_result.waves.wave_level,
            local_result.image_quality.quality_score,
        )

        # Step 2: Person detection (YOLO) — only if camera is working
        person_result: PersonDetectionResult | None = None
        if self._person_detector and local_result.camera_status.status == "working":
            try:
                person_result = self._person_detector.detect(frame.image_bytes)
                logger.info("%s YOLO: %d person(s)", beach.id, person_result.person_count)
            except Exception as e:
                logger.error("%s YOLO failed: %s", beach.id, e)

        # Step 3: Weather API (always, independent of image)
        weather_data: WeatherAPIData | None = None
        if self._weather_client:
            try:
                weather_data = self._weather_client.get_weather(
                    beach.coordinates.latitude,
                    beach.coordinates.longitude,
                    beach_id=beach.id,
                )
            except WeatherAPIError as e:
                logger.warning("%s weather: %s", beach.id, e)
            except Exception as e:
                logger.error("%s weather failed: %s", beach.id, e)

        # Step 4: Claude Vision (if enabled, camera working, image usable)
        vision_result: VisionAnalysis | None = None
        error_message: str | None = None
        model_used: str | None = None

        if (use_ai and self._vision
                and local_result.camera_status.status == "working"
                and local_result.image_quality.is_usable):
            try:
                vision_result = await self._vision.analyze_frame(
                    frame.image_bytes, beach,
                    person_result=person_result,
                    local_result=local_result,
                    weather_data=weather_data,
                )
                model_used = self._vision._settings.model
                logger.info(
                    "%s AI: score=%.1f currents=%s summary=%s",
                    beach.id,
                    vision_result.overall.beach_score,
                    vision_result.currents.danger_level,
                    vision_result.overall.summary[:80],
                )
            except RateLimitError as e:
                error_message = str(e)
                logger.warning("%s: %s", beach.id, e)
            except Exception as e:
                error_message = str(e)
                logger.error("%s AI failed: %s", beach.id, e)
        elif (use_ai and self._vision
              and local_result.camera_status.status != "working"):
            error_message = f"Camera {local_result.camera_status.status}: {local_result.camera_status.reason}"
            logger.info("%s: skipping AI — %s", beach.id, error_message)
        elif (use_ai and self._vision
              and not local_result.image_quality.is_usable):
            error_message = f"Image unusable: {local_result.image_quality.issues}"
            logger.warning("%s: skipping AI — %s", beach.id, error_message)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return self._merge_results(
            beach, frame, local_result, person_result, weather_data,
            vision_result, model_used, elapsed_ms, error_message,
        )

    def _merge_results(
        self,
        beach: BeachConfig,
        frame: GrabbedFrame,
        local_result: LocalAnalysisResult,
        person_result: PersonDetectionResult | None,
        weather_data: WeatherAPIData | None,
        vision_result: VisionAnalysis | None,
        model_used: str | None,
        elapsed_ms: int,
        error_message: str | None,
    ) -> Observation:
        """Merge all results into a single Observation."""
        obs = Observation(
            beach_id=beach.id,
            captured_at=frame.captured_at,
            source_url=frame.source_url,
            # Camera status
            camera_status=local_result.camera_status.status,
            camera_status_reason=local_result.camera_status.reason,
            # Waves
            cv_wave_level=local_result.waves.wave_level,
            cv_whitecap_ratio=local_result.waves.whitecap_ratio,
            cv_edge_density=local_result.waves.edge_density,
            cv_wave_confidence=local_result.waves.confidence,
            cv_image_quality=local_result.image_quality.quality_score,
            # Meta
            analysis_model=model_used,
            processing_time_ms=elapsed_ms,
            error_message=error_message,
        )

        # Person detection
        if person_result:
            obs.person_count = person_result.person_count
            obs.person_confidence = person_result.confidence_avg
            obs.detection_method = person_result.detection_method

        # Weather API
        if weather_data and weather_data.temperature_c is not None:
            obs.weather_temperature_c = weather_data.temperature_c
            obs.weather_feels_like_c = weather_data.feels_like_c
            obs.weather_humidity_pct = weather_data.humidity_pct
            obs.weather_wind_speed_kmh = weather_data.wind_speed_kmh
            obs.weather_wind_direction = weather_data.wind_direction
            obs.weather_wind_gust_kmh = weather_data.wind_gust_kmh
            obs.weather_condition = weather_data.condition
            obs.weather_description = weather_data.description
            obs.weather_precipitation_mm = weather_data.precipitation_mm
            obs.weather_visibility_km = weather_data.visibility_km
            obs.weather_uv_index = weather_data.uv_index

        # Claude Vision
        if vision_result:
            obs.ai_crowd_level = vision_result.crowd.level
            obs.ai_crowd_count = vision_result.crowd.estimated_count
            obs.ai_crowd_distribution = vision_result.crowd.distribution
            obs.ai_crowd_notes = vision_result.crowd.notes
            obs.ai_wave_size = vision_result.waves.size
            obs.ai_wave_quality = vision_result.waves.quality
            obs.ai_wave_type = vision_result.waves.type
            obs.ai_wave_period = vision_result.waves.period_estimate
            obs.ai_wave_notes = vision_result.waves.notes
            obs.ai_weather_condition = vision_result.weather.condition
            obs.ai_wind_estimate = vision_result.weather.wind_estimate
            obs.ai_wind_direction = vision_result.weather.wind_direction_visual
            obs.ai_visibility = vision_result.weather.visibility
            obs.ai_weather_notes = vision_result.weather.notes
            obs.ai_current_danger_level = vision_result.currents.danger_level
            obs.ai_current_rip_detected = vision_result.currents.rip_current_detected
            obs.ai_current_indicators = vision_result.currents.indicators
            obs.ai_current_notes = vision_result.currents.notes
            obs.ai_beach_score = vision_result.overall.beach_score
            obs.ai_surf_score = vision_result.overall.surf_score
            obs.ai_summary = vision_result.overall.summary
            obs.ai_best_for = vision_result.overall.best_for

        return obs
