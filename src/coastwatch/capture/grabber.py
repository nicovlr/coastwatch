"""HTTP frame grabber for webcam snapshots."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from coastwatch.common.exceptions import WebcamUnavailableError
from coastwatch.config.models import BeachConfig

logger = logging.getLogger(__name__)

WINDY_API_URL = "https://api.windy.com/webcams/api/v3/webcams"


@dataclass
class GrabbedFrame:
    beach_id: str
    image_bytes: bytes
    captured_at: str  # ISO 8601 UTC
    source_url: str
    content_type: str = "image/jpeg"


@dataclass
class GrabResult:
    beach_id: str
    frame: GrabbedFrame | None = None
    error: Exception | None = None

    @property
    def success(self) -> bool:
        return self.frame is not None


class FrameGrabber:
    """Fetches single frames from webcam snapshot URLs."""

    def __init__(self, timeout: float = 15.0, max_retries: int = 3, backoff: float = 5.0):
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff = backoff
        self._windy_api_key = os.environ.get("WINDY_API_KEY", "")

    async def grab_frame(self, beach: BeachConfig) -> GrabbedFrame:
        """Fetch a single frame from the beach webcam."""
        url = beach.webcam.snapshot_url
        urls_tried: list[str] = []

        all_urls = [url] + beach.webcam.fallback_urls
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt_url in all_urls:
                urls_tried.append(attempt_url)
                try:
                    image_bytes = await self._fetch_url(client, attempt_url, beach)
                    return GrabbedFrame(
                        beach_id=beach.id,
                        image_bytes=image_bytes,
                        captured_at=datetime.now(timezone.utc).isoformat(),
                        source_url=attempt_url,
                    )
                except Exception as e:
                    last_error = e
                    logger.warning("Failed to grab %s from %s: %s", beach.id, attempt_url, e)

        raise WebcamUnavailableError(beach.id, urls_tried, last_error)

    async def _fetch_url(self, client: httpx.AsyncClient, url: str, beach: BeachConfig) -> bytes:
        """Fetch image bytes from a URL, handling Windy API protocol."""
        if url.startswith("windy://"):
            return await self._fetch_windy(client, url[8:])

        headers = dict(beach.webcam.headers)
        for attempt in range(self._max_retries):
            try:
                resp = await client.get(url, headers=headers, follow_redirects=True)
                resp.raise_for_status()
                return resp.content
            except httpx.HTTPError as e:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._backoff * (attempt + 1))
                else:
                    raise

        raise RuntimeError(f"Unreachable: all retries failed for {url}")

    async def _fetch_windy(self, client: httpx.AsyncClient, webcam_id: str) -> bytes:
        """Fetch snapshot via Windy Webcams API v3."""
        if not self._windy_api_key:
            raise RuntimeError("WINDY_API_KEY not set. Get a free key at https://api.windy.com/webcams")

        api_url = f"{WINDY_API_URL}/{webcam_id}"
        headers = {"X-WINDY-API-KEY": self._windy_api_key}
        params = {"include": "images"}

        for attempt in range(self._max_retries):
            try:
                resp = await client.get(api_url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()

                # Get the best available image URL
                images = data.get("images", {})
                current = images.get("current", {})
                # Try preview first (larger), fall back to icon
                image_url = current.get("preview") or current.get("icon") or current.get("thumbnail")
                if not image_url:
                    raise RuntimeError(f"No image URL in Windy response for webcam {webcam_id}")

                img_resp = await client.get(image_url, follow_redirects=True)
                img_resp.raise_for_status()
                return img_resp.content

            except httpx.HTTPError as e:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._backoff * (attempt + 1))
                else:
                    raise

        raise RuntimeError(f"Unreachable: all retries failed for Windy webcam {webcam_id}")

    async def grab_all(
        self, beaches: list[BeachConfig], concurrency: int = 4
    ) -> list[GrabResult]:
        """Fetch frames from multiple beaches concurrently."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _grab_one(beach: BeachConfig) -> GrabResult:
            async with semaphore:
                try:
                    frame = await self.grab_frame(beach)
                    return GrabResult(beach_id=beach.id, frame=frame)
                except Exception as e:
                    logger.error("Capture failed for %s: %s", beach.id, e)
                    return GrabResult(beach_id=beach.id, error=e)

        results = await asyncio.gather(*[_grab_one(b) for b in beaches])
        return list(results)
