#!/usr/bin/env python3
"""
AI Garment Multi-Label Size Inspection System
==============================================

Usage
-----
  python main.py ui                     # Launch desktop UI (PyQt6)
  python main.py api                    # Launch FastAPI REST server
  python main.py inspect <image_path>   # Inspect a single image (CLI)
  python main.py teach                  # Launch teaching mode standalone
  python main.py setup                  # Run environment setup check
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (before anything else)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_config(path: str = "config.yaml") -> dict:
    """Load YAML config from *path*, falling back to empty dict on error."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        logger.info("Config loaded from '%s'.", path)
        return cfg
    except FileNotFoundError:
        logger.warning("Config file '%s' not found — using defaults.", path)
        return {}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to load config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def launch_ui(config_path: str = "config.yaml") -> None:
    """Start the PyQt6 desktop inspection application."""
    cfg = _load_config(config_path)

    from PyQt6.QtWidgets import QApplication  # type: ignore

    app = QApplication(sys.argv)
    app.setApplicationName("AI Size Inspector")
    app.setOrganizationName("FactoryAI")

    # Build pipeline
    try:
        from src.core.detector import LabelDetector
        from src.core.feature_extractor import FeatureExtractor
    except ImportError:
        from core.detector import LabelDetector  # type: ignore
        from core.feature_extractor import FeatureExtractor  # type: ignore

    detector = LabelDetector(cfg.get("models", {}).get("label_detector", {}))
    extractor = FeatureExtractor(cfg.get("models", {}).get("feature_extractor", {}))

    # Minimal pipeline stub — replace with real Pipeline once api/kb-extractor land
    class _StubPipeline:
        def __init__(self):
            self.camera = _open_camera(cfg.get("camera", {}).get("device_index", 0))
            self.session_manager = None

        def run(self, frame):
            import time
            from src.core.models import InspectionResult, ErrorCode
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            latency = (time.perf_counter() - t0) * 1000
            is_pass = len(detections) > 0
            return InspectionResult(
                is_pass=is_pass,
                error_code=ErrorCode.PASS if is_pass else ErrorCode.MISSING_REQUIRED_LABEL,
                error_detail="" if is_pass else "No labels detected",
                sizes_found={d.label_type.value: (d.size_value or "?") for d in detections},
                detections=detections,
                latency_ms=latency,
            )

    class _StubAlarm:
        def beep_pass(self):
            try:
                import winsound  # type: ignore
                winsound.Beep(880, 100)
            except Exception:  # pylint: disable=broad-except
                pass

        def alarm_ng(self):
            try:
                import winsound  # type: ignore
                winsound.Beep(440, 500)
            except Exception:  # pylint: disable=broad-except
                pass

    pipeline = _StubPipeline()
    alarm = _StubAlarm()

    # Warmup in background
    _warmup_models(detector, extractor)

    try:
        from src.ui.inspection_window import InspectionWindow
    except ImportError:
        from ui.inspection_window import InspectionWindow  # type: ignore

    window = InspectionWindow(pipeline, alarm)
    window.show()

    exit_code = app.exec()
    sys.exit(exit_code)


def launch_api(config_path: str = "config.yaml", host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI REST server via uvicorn."""
    try:
        import uvicorn  # type: ignore
    except ImportError:
        logger.error("uvicorn is not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    logger.info("Starting FastAPI server at http://%s:%d", host, port)
    # The FastAPI app is expected at src.api.app:app
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


def cli_inspect(image_path: str, config_path: str = "config.yaml") -> None:
    """Inspect a single image file and print the result to stdout."""
    import cv2  # type: ignore

    path = Path(image_path)
    if not path.exists():
        logger.error("Image file not found: %s", image_path)
        sys.exit(1)

    frame = cv2.imread(str(path))
    if frame is None:
        logger.error("Failed to decode image: %s", image_path)
        sys.exit(1)

    cfg = _load_config(config_path)

    try:
        from src.core.detector import LabelDetector
        from src.core.feature_extractor import FeatureExtractor
    except ImportError:
        from core.detector import LabelDetector  # type: ignore
        from core.feature_extractor import FeatureExtractor  # type: ignore

    detector = LabelDetector(cfg.get("models", {}).get("label_detector", {}))
    extractor = FeatureExtractor(cfg.get("models", {}).get("feature_extractor", {}))

    import time
    t0 = time.perf_counter()
    detections = detector.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n{'='*60}")
    print(f"  Image    : {image_path}")
    print(f"  Detections: {len(detections)}")
    print(f"  Latency  : {latency_ms:.1f} ms")
    print(f"{'='*60}")

    if not detections:
        print("  ❌ No labels detected.")
    else:
        for i, det in enumerate(detections, 1):
            print(
                f"  [{i}] {det.label_type.value:<20} conf={det.confidence:.3f}"
                + (f"  size={det.size_value}" if det.size_value else "")
                + (" ⚠ needs teaching" if det.needs_teaching else "")
            )
            bbox = det.bbox
            print(f"       bbox=({bbox.x1:.0f},{bbox.y1:.0f})-({bbox.x2:.0f},{bbox.y2:.0f})")

    # Embedding for first detection
    if detections and detections[0].cropped_image is not None:
        emb = extractor.extract(detections[0].cropped_image)
        print(f"\n  Embedding shape : {emb.shape}  |  norm={float((emb**2).sum()**0.5):.4f}")

    print(f"{'='*60}\n")


def launch_teach(config_path: str = "config.yaml") -> None:
    """Launch the teaching window in standalone mode (no live camera)."""
    cfg = _load_config(config_path)

    from PyQt6.QtWidgets import QApplication  # type: ignore

    app = QApplication(sys.argv)
    app.setApplicationName("AI Size Inspector — Teaching Mode")

    import numpy as np
    dummy_image = np.full((300, 400, 3), 200, dtype=np.uint8)

    try:
        from src.ui.teaching_window import TeachingWindow
    except ImportError:
        from ui.teaching_window import TeachingWindow  # type: ignore

    session = {
        "session_id": "standalone-teach",
        "detection": type("Det", (), {"cropped_image": dummy_image, "needs_teaching": True})(),
        "candidates": [],
    }
    dlg = TeachingWindow(session, None)
    dlg.show()
    sys.exit(app.exec())


def run_setup() -> None:
    """Run the environment setup / dependency check script."""
    setup_script = Path("scripts/setup_check.py")
    if setup_script.exists():
        import runpy
        runpy.run_path(str(setup_script), run_name="__main__")
    else:
        logger.warning("scripts/setup_check.py not found. Running inline dependency check.")
        _inline_setup_check()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _open_camera(device_index: int = 0):
    """Return an OpenCV VideoCapture, or None if unavailable."""
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            logger.info("Camera opened on device index %d.", device_index)
            return cap
        logger.warning("Camera device %d not available.", device_index)
        return None
    except Exception:  # pylint: disable=broad-except
        return None


def _warmup_models(detector, extractor) -> None:
    """Warmup models in a daemon thread so the UI opens immediately."""
    import threading

    def _do():
        try:
            detector.warmup()
            extractor.warmup()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Model warmup warning: %s", exc)

    t = threading.Thread(target=_do, daemon=True, name="model-warmup")
    t.start()


def _inline_setup_check() -> None:
    """Quick inline dependency check printed to stdout."""
    checks = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("ultralytics", "Ultralytics (YOLOv8)"),
        ("cv2", "OpenCV"),
        ("paddleocr", "PaddleOCR"),
        ("PyQt6", "PyQt6"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "uvicorn"),
        ("faiss", "FAISS"),
        ("sqlalchemy", "SQLAlchemy"),
    ]
    print("\n── Dependency Check ──────────────────────────")
    all_ok = True
    for mod, name in checks:
        try:
            __import__(mod)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}  (not installed)")
            all_ok = False

    try:
        import torch  # type: ignore
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "—"
        print(f"\n  GPU : {'✅ ' + gpu_name if cuda_ok else '⚠ CUDA not available (CPU mode)'}")
    except ImportError:
        pass

    print("──────────────────────────────────────────────")
    print("  Status:", "All OK ✅" if all_ok else "Some packages missing ❌")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Garment Multi-Label Size Inspection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml", metavar="PATH",
        help="Path to config.yaml (default: config.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    subparsers.add_parser("ui", help="Launch desktop inspection UI (PyQt6)")
    subparsers.add_parser("api", help="Launch FastAPI REST server")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a single image file")
    inspect_parser.add_argument("image_path", help="Path to the image file")

    subparsers.add_parser("teach", help="Launch teaching mode (standalone)")
    subparsers.add_parser("setup", help="Run environment setup / dependency check")

    args = parser.parse_args()

    if args.command == "ui":
        launch_ui(config_path=args.config)
    elif args.command == "api":
        launch_api(config_path=args.config)
    elif args.command == "inspect":
        cli_inspect(args.image_path, config_path=args.config)
    elif args.command == "teach":
        launch_teach(config_path=args.config)
    elif args.command == "setup":
        run_setup()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
