"""Claude Vision API integration for beach analysis."""

from __future__ import annotations

import base64
import json
import logging
import re

import anthropic

from coastwatch.analysis.models import OpenCVResult, VisionAnalysis
from coastwatch.common.exceptions import VisionAnalysisError, VisionParseError
from coastwatch.common.rate_limiter import TokenBucketRateLimiter
from coastwatch.config.models import BeachConfig, ClaudeSettings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a coastal conditions analyst specializing in French Atlantic beaches. You analyze webcam images to provide structured reports on beach conditions.

You will receive:
1. A beach webcam image
2. Pre-computed OpenCV analysis results for context

Your task: Analyze the image and return a JSON object with this exact schema:

{
  "crowd": {
    "level": "empty|light|moderate|crowded|packed",
    "estimated_count": <int or null>,
    "distribution": "even|clustered|shoreline|none",
    "notes": "<string, 1-2 sentences>"
  },
  "waves": {
    "size": "flat|ankle|knee|waist|chest|overhead|double_overhead",
    "quality": "poor|fair|good|excellent",
    "type": "closeout|reform|peeling_left|peeling_right|mixed",
    "period_estimate": "<string, e.g. 'short period' or 'long period'>",
    "notes": "<string, 1-2 sentences>"
  },
  "weather": {
    "condition": "sunny|partly_cloudy|overcast|rainy|stormy|foggy",
    "wind_estimate": "calm|light|moderate|strong|gale",
    "wind_direction_visual": "<string or null, if visible from flags/trees>",
    "visibility": "excellent|good|moderate|poor",
    "notes": "<string, 1-2 sentences>"
  },
  "overall": {
    "beach_score": <float 1.0-10.0>,
    "surf_score": <float 1.0-10.0 or null if not a surf spot>,
    "summary": "<string, 2-3 sentences describing overall conditions>",
    "best_for": ["<activity1>", "<activity2>"]
  }
}

Respond ONLY with valid JSON. No markdown, no commentary outside the JSON."""


class VisionClient:
    """Sends beach webcam frames to Claude Vision API for analysis."""

    def __init__(self, settings: ClaudeSettings, rate_limiter: TokenBucketRateLimiter):
        self._client = anthropic.Anthropic()
        self._settings = settings
        self._rate_limiter = rate_limiter

    async def analyze_frame(
        self,
        image_bytes: bytes,
        beach: BeachConfig,
        opencv_result: OpenCVResult,
    ) -> VisionAnalysis:
        """Send frame to Claude Vision API with structured prompt."""
        await self._rate_limiter.acquire()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        context_text = (
            f"Beach: {beach.name} ({beach.region})\n"
            f"Camera facing: {beach.metadata.orientation}\n"
            f"Surf spot: {'yes' if beach.metadata.surf_spot else 'no'}\n\n"
            f"OpenCV pre-analysis:\n"
            f"- Crowd: {opencv_result.crowd.crowd_level} ({opencv_result.crowd.blob_count} detections)\n"
            f"- Waves: {opencv_result.waves.wave_level} (whitecap ratio: {opencv_result.waves.whitecap_ratio:.3f})\n"
            f"- Weather: {opencv_result.weather.condition} (brightness: {opencv_result.weather.brightness:.0f})\n"
            f"- Image quality: {opencv_result.image_quality.quality_score:.1f}/1.0\n\n"
            f"Analyze this image and return the JSON report."
        )

        try:
            response = self._client.messages.create(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": context_text,
                            },
                        ],
                    }
                ],
                temperature=self._settings.temperature,
            )
        except anthropic.APIError as e:
            raise VisionAnalysisError(beach.id, str(e), getattr(e, "status_code", None))

        raw_text = response.content[0].text
        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> VisionAnalysis:
        """Parse Claude's JSON response into VisionAnalysis."""
        # Strip markdown code fences if present
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise VisionParseError(f"Invalid JSON from Claude: {e}\nRaw: {text[:500]}")

        try:
            return VisionAnalysis(**data)
        except Exception as e:
            raise VisionParseError(f"Invalid schema from Claude: {e}")
