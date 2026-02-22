"""Sunrise/sunset calculation for camera status detection."""

from __future__ import annotations

from datetime import datetime, timezone

from astral import LocationInfo
from astral.sun import sun


def is_daylight(latitude: float, longitude: float, tz_name: str = "Europe/Paris") -> bool:
    """Return True if the sun is currently up at the given coordinates."""
    loc = LocationInfo(latitude=latitude, longitude=longitude, timezone=tz_name)
    now = datetime.now(timezone.utc)
    try:
        s = sun(loc.observer, date=now.date(), tzinfo=timezone.utc)
        return s["sunrise"] <= now <= s["sunset"]
    except Exception:
        # In polar regions or edge cases, assume daylight
        return True


def get_sun_times(latitude: float, longitude: float, tz_name: str = "Europe/Paris") -> dict[str, datetime]:
    """Return sunrise/sunset times for today at the given coordinates."""
    loc = LocationInfo(latitude=latitude, longitude=longitude, timezone=tz_name)
    now = datetime.now(timezone.utc)
    s = sun(loc.observer, date=now.date(), tzinfo=timezone.utc)
    return {"sunrise": s["sunrise"], "sunset": s["sunset"]}
