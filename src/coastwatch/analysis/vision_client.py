"""Claude Vision API integration for beach analysis."""

from __future__ import annotations

import base64
import json
import logging
import re

import anthropic

from coastwatch.analysis.models import (
    LocalAnalysisResult,
    PersonDetectionResult,
    VisionAnalysis,
    WeatherAPIData,
)
from coastwatch.common.exceptions import VisionAnalysisError, VisionParseError
from coastwatch.common.rate_limiter import TokenBucketRateLimiter
from coastwatch.config.models import BeachConfig, ClaudeSettings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a coastal conditions analyst specializing in French Atlantic beaches. You analyze webcam images to provide structured reports on beach conditions, including dangerous rip current detection (courants de baïne).

You will receive:
1. A beach webcam image
2. Context: person count (YOLO detection), wave analysis, real weather data

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
  "currents": {
    "danger_level": "safe|low|moderate|high|extreme",
    "rip_current_detected": <boolean>,
    "indicators": ["<list of observed indicators>"],
    "notes": "<string, 1-3 sentences about current safety>"
  },
  "overall": {
    "beach_score": <float 1.0-10.0>,
    "surf_score": <float 1.0-10.0 or null if not a surf spot>,
    "summary": "<string, 2-3 sentences describing overall conditions>",
    "best_for": ["<activity1>", "<activity2>"]
  }
}

For rip current analysis, look for these visual indicators:
- Channels of darker, calmer water cutting through breaking waves
- Discolored or muddy water flowing seaward
- Foam, seaweed, or debris moving steadily out to sea
- A gap in the line of breaking waves
- Choppy, turbulent water in a narrow band going offshore

If the image shows low tide with exposed sandbars or a receding tide, increase vigilance for baïnes (rip currents common on the French Atlantic coast).

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
        person_result: PersonDetectionResult | None = None,
        local_result: LocalAnalysisResult | None = None,
        weather_data: WeatherAPIData | None = None,
    ) -> VisionAnalysis:
        """Send frame to Claude Vision API with structured prompt."""
        await self._rate_limiter.acquire()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Build context with all available data
        lines = [
            f"Beach: {beach.name} ({beach.region})",
            f"Camera facing: {beach.metadata.orientation}",
            f"Surf spot: {'yes' if beach.metadata.surf_spot else 'no'}",
            "",
        ]

        if person_result:
            lines.append(f"Person detection (YOLO): {person_result.person_count} person(s) detected "
                         f"(avg confidence: {person_result.confidence_avg:.2f})")

        if local_result:
            lines.append(f"Wave analysis: {local_result.waves.wave_level} "
                         f"(whitecap ratio: {local_result.waves.whitecap_ratio:.3f})")
            lines.append(f"Image quality: {local_result.image_quality.quality_score:.1f}/1.0")

        if weather_data and weather_data.temperature_c is not None:
            lines.append("")
            lines.append("Real weather data (OpenWeatherMap):")
            lines.append(f"  Temperature: {weather_data.temperature_c:.1f}°C "
                         f"(feels like {weather_data.feels_like_c:.1f}°C)" if weather_data.feels_like_c else
                         f"  Temperature: {weather_data.temperature_c:.1f}°C")
            lines.append(f"  Condition: {weather_data.condition} — {weather_data.description}")
            lines.append(f"  Wind: {weather_data.wind_speed_kmh:.0f} km/h {weather_data.wind_direction or '?'}"
                         + (f" (gusts {weather_data.wind_gust_kmh:.0f} km/h)" if weather_data.wind_gust_kmh else ""))
            lines.append(f"  Humidity: {weather_data.humidity_pct}%")
            if weather_data.precipitation_mm > 0:
                lines.append(f"  Precipitation: {weather_data.precipitation_mm:.1f} mm/h")
            if weather_data.visibility_km is not None:
                lines.append(f"  Visibility: {weather_data.visibility_km:.0f} km")

        lines.append("")
        lines.append("Analyze this image and return the JSON report. "
                      "Pay special attention to rip current indicators.")

        context_text = "\n".join(lines)

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
