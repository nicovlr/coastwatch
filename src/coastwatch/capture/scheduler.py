"""Periodic capture scheduler."""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from coastwatch.analysis.pipeline import AnalysisPipeline
from coastwatch.capture.grabber import FrameGrabber
from coastwatch.config.models import BeachConfig
from coastwatch.storage.repository import ObservationRepository

logger = logging.getLogger(__name__)


class CaptureScheduler:
    """Manages periodic capture-and-analyze cycles."""

    def __init__(
        self,
        beaches: list[BeachConfig],
        grabber: FrameGrabber,
        pipeline: AnalysisPipeline,
        repository: ObservationRepository,
        default_interval: int = 300,
    ):
        self._beaches = beaches
        self._grabber = grabber
        self._pipeline = pipeline
        self._repo = repository
        self._default_interval = default_interval
        self._running = False

    async def run_once(self, beach_ids: list[str] | None = None, use_ai: bool = True) -> list[str]:
        """Execute a single capture-and-analyze cycle. Returns successful beach IDs."""
        beaches = self._beaches
        if beach_ids:
            beaches = [b for b in beaches if b.id in beach_ids]

        results = await self._grabber.grab_all(beaches)
        successful: list[str] = []

        for result in results:
            if not result.success:
                logger.warning("Skipping %s: capture failed", result.beach_id)
                continue

            beach = next(b for b in beaches if b.id == result.beach_id)
            try:
                obs = await self._pipeline.process_frame(result.frame, beach, use_ai=use_ai)
                self._repo.save(obs)
                successful.append(result.beach_id)
                logger.info("Processed %s: score=%s", result.beach_id,
                            obs.ai_beach_score or "N/A (OpenCV only)")
            except Exception as e:
                logger.error("Analysis failed for %s: %s", result.beach_id, e)

        return successful

    async def run_daemon(self, use_ai: bool = True) -> None:
        """Run continuous capture loop until interrupted."""
        self._running = True
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop)

        logger.info("Daemon started. Monitoring %d beaches every %ds",
                     len(self._beaches), self._default_interval)

        while self._running:
            start = time.monotonic()
            try:
                successful = await self.run_once(use_ai=use_ai)
                logger.info("Cycle complete: %d/%d beaches OK",
                            len(successful), len(self._beaches))
            except Exception as e:
                logger.error("Capture cycle error: %s", e)

            elapsed = time.monotonic() - start
            sleep_time = max(0, self._default_interval - elapsed)
            if self._running and sleep_time > 0:
                logger.debug("Sleeping %.0fs until next cycle", sleep_time)
                await asyncio.sleep(sleep_time)

        logger.info("Daemon stopped.")

    def _stop(self) -> None:
        logger.info("Shutdown signal received")
        self._running = False
