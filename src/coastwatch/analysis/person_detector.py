"""YOLOv8-based person detection for beach crowd counting."""

from __future__ import annotations

import logging

import numpy as np

from coastwatch.analysis.models import PersonDetectionResult
from coastwatch.config.models import YOLOSettings

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton to avoid reloading on every frame
_model = None


def _get_model(model_name: str):
    """Load YOLOv8 model once (lazy singleton)."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        logger.info("Loading YOLO model: %s", model_name)
        _model = YOLO(model_name)
    return _model


class PersonDetector:
    """Counts people in beach webcam frames using YOLOv8-nano."""

    def __init__(self, settings: YOLOSettings | None = None):
        self._s = settings or YOLOSettings()

    def detect(self, image_bytes: bytes) -> PersonDetectionResult:
        """Run person detection on a JPEG/PNG image."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        import cv2
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return PersonDetectionResult(person_count=0, confidence_avg=0.0)

        model = _get_model(self._s.model)
        results = model(img, conf=self._s.confidence_threshold, verbose=False)

        person_count = 0
        confidence_sum = 0.0

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self._s.person_class_id:
                    person_count += 1
                    confidence_sum += float(box.conf[0])

        confidence_avg = (confidence_sum / person_count) if person_count > 0 else 0.0

        logger.debug("YOLO detected %d person(s), avg confidence %.2f", person_count, confidence_avg)

        return PersonDetectionResult(
            person_count=person_count,
            confidence_avg=round(confidence_avg, 3),
            detection_method="yolo",
        )
