"""Analysis pipeline orchestrator: OpenCV -> Claude Vision -> merge."""

from __future__ import annotations

import logging
import time

from coastwatch.analysis.models import OpenCVResult, VisionAnalysis
from coastwatch.analysis.opencv_analyzer import OpenCVAnalyzer
from coastwatch.analysis.vision_client import VisionClient
from coastwatch.capture.grabber import GrabbedFrame
from coastwatch.common.exceptions import RateLimitError
from coastwatch.config.models import BeachConfig
from coastwatch.storage.repository import Observation

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates OpenCV pre-processing and Claude Vision analysis."""

    def __init__(
        self,
        opencv_analyzer: OpenCVAnalyzer,
        vision_client: VisionClient | None = None,
    ):
        self._opencv = opencv_analyzer
        self._vision = vision_client

    async def process_frame(
        self,
        frame: GrabbedFrame,
        beach: BeachConfig,
        use_ai: bool = True,
    ) -> Observation:
        """Full analysis pipeline for a single frame."""
        start = time.monotonic()

        # Step 1: OpenCV analysis (always runs)
        opencv_result = self._opencv.analyze(frame.image_bytes)
        logger.debug(
            "%s OpenCV: crowd=%s waves=%s weather=%s quality=%.1f",
            beach.id,
            opencv_result.crowd.crowd_level,
            opencv_result.waves.wave_level,
            opencv_result.weather.condition,
            opencv_result.image_quality.quality_score,
        )

        # Step 2: Claude Vision (if enabled and image is usable)
        vision_result: VisionAnalysis | None = None
        error_message: str | None = None
        model_used: str | None = None

        if use_ai and self._vision and opencv_result.image_quality.is_usable:
            try:
                vision_result = await self._vision.analyze_frame(
                    frame.image_bytes, beach, opencv_result
                )
                model_used = self._vision._settings.model
                logger.info(
                    "%s AI: beach_score=%.1f surf_score=%s summary=%s",
                    beach.id,
                    vision_result.overall.beach_score,
                    vision_result.overall.surf_score,
                    vision_result.overall.summary[:80],
                )
            except RateLimitError as e:
                error_message = str(e)
                logger.warning("%s: %s", beach.id, e)
            except Exception as e:
                error_message = str(e)
                logger.error("%s AI analysis failed: %s", beach.id, e)
        elif use_ai and self._vision and not opencv_result.image_quality.is_usable:
            error_message = f"Image unusable: {opencv_result.image_quality.issues}"
            logger.warning("%s: skipping AI - %s", beach.id, error_message)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return self._merge_results(
            beach, frame, opencv_result, vision_result, model_used, elapsed_ms, error_message
        )

    def _merge_results(
        self,
        beach: BeachConfig,
        frame: GrabbedFrame,
        opencv_result: OpenCVResult,
        vision_result: VisionAnalysis | None,
        model_used: str | None,
        elapsed_ms: int,
        error_message: str | None,
    ) -> Observation:
        """Merge OpenCV and Claude results into a single Observation."""
        obs = Observation(
            beach_id=beach.id,
            captured_at=frame.captured_at,
            source_url=frame.source_url,
            # OpenCV fields
            cv_crowd_count=opencv_result.crowd.blob_count,
            cv_crowd_level=opencv_result.crowd.crowd_level,
            cv_crowd_confidence=opencv_result.crowd.confidence,
            cv_wave_level=opencv_result.waves.wave_level,
            cv_whitecap_ratio=opencv_result.waves.whitecap_ratio,
            cv_edge_density=opencv_result.waves.edge_density,
            cv_wave_confidence=opencv_result.waves.confidence,
            cv_weather_condition=opencv_result.weather.condition,
            cv_brightness=opencv_result.weather.brightness,
            cv_blue_ratio=opencv_result.weather.blue_ratio,
            cv_visibility=opencv_result.weather.visibility,
            cv_weather_confidence=opencv_result.weather.confidence,
            cv_image_quality=opencv_result.image_quality.quality_score,
            # Meta
            analysis_model=model_used,
            processing_time_ms=elapsed_ms,
            error_message=error_message,
        )

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
            obs.ai_beach_score = vision_result.overall.beach_score
            obs.ai_surf_score = vision_result.overall.surf_score
            obs.ai_summary = vision_result.overall.summary
            obs.ai_best_for = vision_result.overall.best_for

        return obs
