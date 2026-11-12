"""
Label Detector — YOLOv8-based garment label detection.
Detects size labels, care labels, hang tags, and unknown labels.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from src.core.models import BBox, LabelDetection, LabelType

logger = logging.getLogger(__name__)


class LabelDetector:
    """YOLOv8-based detector that locates garment labels in an image frame."""

    LABEL_CLASSES: dict[int, str] = {
        0: "size_label",
        1: "care_label",
        2: "hang_tag",
        3: "unknown_label",
    }

    _LABEL_TYPE_MAP: dict[str, LabelType] = {
        "size_label": LabelType.SIZE_LABEL,
        "care_label": LabelType.CARE_LABEL,
        "hang_tag": LabelType.HANG_TAG,
        "unknown_label": LabelType.UNKNOWN,
    }

    def __init__(self, config: dict) -> None:
        self.conf: float = float(config.get("conf_threshold", 0.5))
        self.iou: float = float(config.get("iou_threshold", 0.45))
        self.img_size: int = int(config.get("img_size", 640))
        self.device: str = str(config.get("device", "cpu"))
        self.weights_path: str = str(
            config.get("weights", "models/label_detector/weights/best.pt")
        )
        self.model = None  # Lazy-loaded on first call

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the YOLO model from disk (or fall back to pretrained yolov8n)."""
        from ultralytics import YOLO  # type: ignore

        if not os.path.exists(self.weights_path):
            logger.warning(
                "Custom weights not found at '%s', using pretrained YOLOv8n as placeholder.",
                self.weights_path,
            )
            self.model = YOLO("yolov8n.pt")
        else:
            logger.info("Loading YOLOv8 weights from '%s'.", self.weights_path)
            self.model = YOLO(self.weights_path)

        self.model.to(self.device)
        logger.info("LabelDetector model loaded on device '%s'.", self.device)

    def _crop(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        """Return the sub-image corresponding to *bbox*."""
        h, w = image.shape[:2]
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))
        return image[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[LabelDetection]:
        """
        Run label detection on *image* (BGR numpy array).

        Returns
        -------
        List[LabelDetection]
            Detected labels sorted by confidence descending.
        """
        if self.model is None:
            self._load_model()

        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        detections: List[LabelDetection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # xyxy coordinates (absolute pixel values)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val: float = float(box.conf[0])
                cls_id: int = int(box.cls[0])

                class_name = self.LABEL_CLASSES.get(cls_id, "unknown_label")
                label_type = self._LABEL_TYPE_MAP.get(class_name, LabelType.UNKNOWN)
                needs_teaching = label_type == LabelType.UNKNOWN

                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf_val)
                cropped = self._crop(image, bbox)

                detection = LabelDetection(
                    label_type=label_type,
                    bbox=bbox,
                    confidence=conf_val,
                    cropped_image=cropped,
                    needs_teaching=needs_teaching,
                )
                detections.append(detection)

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("Detected %d label(s).", len(detections))
        return detections

    def warmup(self, rounds: int = 1) -> None:
        """
        Run dummy inference to pre-load GPU kernels / JIT caches.
        Call once at startup before the first real inspection.
        """
        if self.model is None:
            self._load_model()

        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for _ in range(rounds):
            self.model.predict(
                source=dummy,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )
        logger.info("LabelDetector warmup complete (%d round(s)).", rounds)
