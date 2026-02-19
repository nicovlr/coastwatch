"""Token-bucket rate limiter for API calls."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from coastwatch.common.exceptions import RateLimitError


class TokenBucketRateLimiter:
    """Dual rate limiter: per-minute (burst) and per-day (budget)."""

    def __init__(self, rpm: int = 30, daily: int = 500):
        self._rpm = rpm
        self._daily = daily
        self._tokens = float(rpm)
        self._last_refill = time.monotonic()
        self._daily_used = 0
        self._daily_reset_date = datetime.now(timezone.utc).date()

    async def acquire(self) -> None:
        """Wait until a token is available. Raises RateLimitError if daily cap reached."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_reset_date:
            self._daily_used = 0
            self._daily_reset_date = today

        if self._daily_used >= self._daily:
            raise RateLimitError(
                f"Daily API limit reached ({self._daily}). Resets at midnight UTC."
            )

        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._daily_used += 1
                return
            wait_time = (1.0 - self._tokens) / (self._rpm / 60.0)
            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(float(self._rpm), self._tokens + elapsed * (self._rpm / 60.0))
        self._last_refill = now

    @property
    def remaining_today(self) -> int:
        today = datetime.now(timezone.utc).date()
        if today != self._daily_reset_date:
            return self._daily
        return max(0, self._daily - self._daily_used)
