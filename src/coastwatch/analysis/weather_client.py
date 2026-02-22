"""OpenWeatherMap API client for real weather data."""

from __future__ import annotations

import logging
import os
import time

import httpx

from coastwatch.analysis.models import WeatherAPIData
from coastwatch.common.exceptions import WeatherAPIError
from coastwatch.config.models import WeatherAPISettings

logger = logging.getLogger(__name__)

OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Map OWM condition IDs to our condition strings
_CONDITION_MAP = {
    range(200, 300): "storm",
    range(300, 400): "rain",
    range(500, 600): "rain",
    range(600, 700): "snow",
    range(700, 800): "fog",
}


def _owm_id_to_condition(weather_id: int) -> str:
    if weather_id == 800:
        return "clear"
    if 801 <= weather_id <= 802:
        return "partly_cloudy"
    if 803 <= weather_id <= 804:
        return "overcast"
    for id_range, condition in _CONDITION_MAP.items():
        if weather_id in id_range:
            return condition
    return "unknown"


def _degrees_to_direction(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


class WeatherClient:
    """Fetches real weather data from OpenWeatherMap free tier."""

    def __init__(self, settings: WeatherAPISettings | None = None):
        self._s = settings or WeatherAPISettings()
        self._api_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")
        self._cache: dict[str, tuple[float, WeatherAPIData]] = {}

    def get_weather(self, latitude: float, longitude: float, beach_id: str = "") -> WeatherAPIData:
        """Fetch current weather for coordinates. Uses cache with TTL."""
        cache_key = f"{latitude:.4f},{longitude:.4f}"

        # Check cache
        if cache_key in self._cache:
            ts, cached_data = self._cache[cache_key]
            if time.monotonic() - ts < self._s.cache_ttl_sec:
                logger.debug("Weather cache hit for %s", beach_id or cache_key)
                return cached_data

        if not self._api_key:
            logger.warning("OPENWEATHERMAP_API_KEY not set, returning empty weather data")
            return WeatherAPIData()

        try:
            resp = httpx.get(
                OWM_BASE_URL,
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "appid": self._api_key,
                    "units": "metric",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise WeatherAPIError(str(e), status_code=e.response.status_code)
        except httpx.RequestError as e:
            raise WeatherAPIError(str(e))

        data = resp.json()
        weather_main = data.get("weather", [{}])[0]
        main = data.get("main", {})
        wind = data.get("wind", {})

        weather_id = weather_main.get("id", 0)
        wind_deg = wind.get("deg", 0)

        result = WeatherAPIData(
            temperature_c=main.get("temp"),
            feels_like_c=main.get("feels_like"),
            humidity_pct=main.get("humidity"),
            wind_speed_kmh=round(wind.get("speed", 0) * 3.6, 1),  # m/s to km/h
            wind_direction=_degrees_to_direction(wind_deg) if wind_deg else None,
            wind_gust_kmh=round(wind.get("gust", 0) * 3.6, 1) if wind.get("gust") else None,
            condition=_owm_id_to_condition(weather_id),
            description=weather_main.get("description", ""),
            precipitation_mm=data.get("rain", {}).get("1h", 0.0) + data.get("snow", {}).get("1h", 0.0),
            visibility_km=round(data.get("visibility", 10000) / 1000, 1),
            uv_index=None,  # not available in free tier
        )

        # Cache the result
        self._cache[cache_key] = (time.monotonic(), result)
        logger.debug("Weather for %s: %s %.1f°C wind %.0fkm/h %s",
                      beach_id or cache_key, result.condition,
                      result.temperature_c or 0, result.wind_speed_kmh or 0,
                      result.wind_direction or "?")

        return result
