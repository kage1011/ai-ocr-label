"""
src/core/pipeline.py
InspectionPipeline — orchestrates the full garment-label inspection sequence:
  camera frame → label detection → template matching → OCR → consistency judge
  → alarm → snapshot save → InspectionResult.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from .judge import ConsistencyJudge
from .models import ErrorCode, InspectionResult, LabelDetection, LabelType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.debug("Config loaded from %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InspectionPipeline:
    """
    End-to-end inspection orchestrator.

    Instantiate once at start-up, call :meth:`warm_up` to load heavy models,
    then call :meth:`run_sync` for every incoming frame.

    Attributes:
        cfg:            Full configuration dictionary from ``config.yaml``.
        detector:       Label-detection model (YOLO / placeholder).
        knowledge_base: Vector / SQL knowledge base (placeholder).
        size_extractor: OCR + embedding size extractor (placeholder).
        judge:          :class:`ConsistencyJudge` instance.
    """

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG_PATH) -> None:
        self.cfg: Dict[str, Any] = load_config(config_path)
        self._warmed_up: bool = False

        # Sub-components — replaced with real objects in warm_up()
        self.detector: Optional[Any] = None
        self.knowledge_base: Optional[Any] = None
        self.size_extractor: Optional[Any] = None
        self.judge: Optional[ConsistencyJudge] = None

        logger.info("InspectionPipeline initialised (not yet warmed up)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """
        Load all heavy models into memory / GPU.

        Call once after construction.  Subsequent calls are no-ops.
        """
        if self._warmed_up:
            return

        t0 = time.perf_counter()
        logger.info("Warming up pipeline …")

        # --- Label detector (YOLO via ultralytics) ---------------------
        try:
            from ultralytics import YOLO  # type: ignore
            weights = self.cfg["models"]["label_detector"]["weights"]
            if Path(weights).exists():
                self.detector = YOLO(weights)
                device = self.cfg["models"]["label_detector"].get("device", "cpu")
                self.detector.to(device)
                logger.info("Label detector loaded: %s on %s", weights, device)
            else:
                logger.warning(
                    "Detector weights not found at '%s' — using stub detector", weights
                )
                self.detector = _StubDetector()
        except ImportError:
            logger.warning("ultralytics not installed — using stub detector")
            self.detector = _StubDetector()

        # --- Feature extractor (ONNX placeholder) ----------------------
        # TODO: load MobileNetV3 ONNX embedder via onnxruntime
        logger.info("Feature extractor: placeholder (not yet implemented)")
        self.knowledge_base = _StubKnowledgeBase()

        # --- OCR / size extractor (PaddleOCR placeholder) --------------
        # TODO: instantiate PaddleOCR with config settings
        logger.info("Size extractor: placeholder (not yet implemented)")
        self.size_extractor = _StubSizeExtractor()

        # --- Consistency judge -----------------------------------------
        required = self.cfg["inspection"]["required_label_types"]
        self.judge = ConsistencyJudge(required_label_types=required)

        self._warmed_up = True
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Pipeline warm-up complete in %.1f ms", elapsed)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_sync(self, image: np.ndarray) -> InspectionResult:
        """
        Run the full inspection pipeline on a single BGR frame.

        Args:
            image: ``(H, W, 3)`` uint8 numpy array in BGR colour order
                   (as returned by ``cv2.imread`` / camera capture).

        Returns:
            :class:`InspectionResult` populated with all stage outputs.
        """
        if not self._warmed_up:
            self.warm_up()

        pipeline_start = time.perf_counter()
        stage_latencies: Dict[str, float] = {}

        # ----------------------------------------------------------------
        # Stage 1 — Label Detection
        # ----------------------------------------------------------------
        t = time.perf_counter()
        raw_detections = self._run_detector(image)
        stage_latencies["detection_ms"] = (time.perf_counter() - t) * 1000
        logger.debug(
            "Detection: %d labels in %.1f ms",
            len(raw_detections),
            stage_latencies["detection_ms"],
        )

        # ----------------------------------------------------------------
        # Stage 2 — Template Matching / Knowledge Base Lookup
        # ----------------------------------------------------------------
        t = time.perf_counter()
        matched_detections = self._run_matching(image, raw_detections)
        stage_latencies["matching_ms"] = (time.perf_counter() - t) * 1000

        # ----------------------------------------------------------------
        # Stage 3 — OCR / Size Extraction
        # ----------------------------------------------------------------
        t = time.perf_counter()
        enriched_detections = self._run_ocr(image, matched_detections)
        stage_latencies["ocr_ms"] = (time.perf_counter() - t) * 1000

        # ----------------------------------------------------------------
        # Stage 4 — Consistency Judgement
        # ----------------------------------------------------------------
        t = time.perf_counter()
        judge_result = self.judge.judge(enriched_detections)
        stage_latencies["judge_ms"] = (time.perf_counter() - t) * 1000

        # ----------------------------------------------------------------
        # Stage 5 — Snapshot (optional)
        # ----------------------------------------------------------------
        snapshot_path: Optional[str] = None
        if self.cfg["logging"].get("save_snapshots"):
            t = time.perf_counter()
            snapshot_path = self._save_snapshot(image, judge_result.is_pass)
            stage_latencies["snapshot_ms"] = (time.perf_counter() - t) * 1000

        # ----------------------------------------------------------------
        # Assemble result
        # ----------------------------------------------------------------
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        needs_teaching = any(d.needs_teaching for d in enriched_detections)
        unknown_crops = [
            d.cropped_image
            for d in enriched_detections
            if d.label_type == LabelType.UNKNOWN and d.cropped_image is not None
        ]

        result = InspectionResult(
            is_pass=judge_result.is_pass,
            error_code=judge_result.error_code,
            error_detail=judge_result.error_detail,
            sizes_found=judge_result.sizes_dict,
            detections=enriched_detections,
            latency_ms=total_ms,
            stage_latencies=stage_latencies,
            snapshot_path=snapshot_path,
            needs_teaching=needs_teaching,
            unknown_label_crops=unknown_crops,
        )

        budget_ms = self.cfg["inspection"].get("latency_budget_ms", 200)
        if total_ms > budget_ms:
            logger.warning(
                "Latency %.1f ms exceeds budget %d ms", total_ms, budget_ms
            )
        else:
            logger.info("Pipeline: %s", result.summary())

        return result

    # ------------------------------------------------------------------
    # Private stage helpers (stubbed — replace with real implementations)
    # ------------------------------------------------------------------

    def _run_detector(self, image: np.ndarray) -> list[LabelDetection]:
        """Call the label detector model and return raw detections."""
        return self.detector.detect(image)

    def _run_matching(
        self, image: np.ndarray, detections: list[LabelDetection]
    ) -> list[LabelDetection]:
        """Match each detected region against the knowledge base."""
        for det in detections:
            self.knowledge_base.match(image, det)
        return detections

    def _run_ocr(
        self, image: np.ndarray, detections: list[LabelDetection]
    ) -> list[LabelDetection]:
        """Extract size text from each detection using OCR."""
        for det in detections:
            self.size_extractor.extract(image, det)
        return detections

    def _save_snapshot(self, image: np.ndarray, is_pass: bool) -> str:
        """Save a JPEG snapshot of the frame and return the file path."""
        try:
            import cv2  # type: ignore

            snapshot_dir = Path(self.cfg["logging"]["snapshot_dir"])
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            label = "PASS" if is_pass else "FAIL"
            fname = snapshot_dir / f"{ts}_{label}.jpg"
            cv2.imwrite(str(fname), image)
            return str(fname)
        except Exception as exc:
            logger.error("Snapshot failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Stub implementations (removed when real modules are wired in)
# ---------------------------------------------------------------------------

class _StubDetector:
    """Returns an empty detection list — stand-in until YOLO weights exist."""

    def detect(self, image: np.ndarray) -> list[LabelDetection]:
        logger.debug("_StubDetector: returning empty detections")
        return []


class _StubKnowledgeBase:
    """No-op knowledge-base match."""

    def match(self, image: np.ndarray, detection: LabelDetection) -> None:
        pass


class _StubSizeExtractor:
    """No-op OCR size extractor."""

    def extract(self, image: np.ndarray, detection: LabelDetection) -> None:
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if image_path:
        try:
            import cv2  # type: ignore

            img = cv2.imread(image_path)
            if img is None:
                print(f"ERROR: Could not read image: {image_path}")
                sys.exit(1)
        except ImportError:
            print("ERROR: opencv-python is required. pip install opencv-python")
            sys.exit(1)
    else:
        # Create a blank 1080p test frame
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        print("No image path provided — using blank 1920×1080 test frame.")

    pipeline = InspectionPipeline()
    pipeline.warm_up()
    result = pipeline.run_sync(img)

    print("\n=== Inspection Result ===")
    print(f"  Pass:         {result.is_pass}")
    print(f"  Error code:   {result.error_code}")
    print(f"  Error detail: {result.error_detail}")
    print(f"  Sizes found:  {result.sizes_found}")
    print(f"  Latency:      {result.latency_ms:.1f} ms")
    print(f"  Stages:       {result.stage_latencies}")
    print(f"  Snapshot:     {result.snapshot_path}")
    print(f"  Timestamp:    {result.timestamp}")
