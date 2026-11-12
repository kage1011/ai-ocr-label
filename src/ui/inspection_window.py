"""
Inspection Window — PyQt6 main window for factory-line workers.

Layout
------
┌─────────────────────────────────────────────────────┐
│  Title: "AI Size Inspector v1.0 | GPU: … | Today: …"│
├──────────────────┬──────────┬────────────────────────┤
│  Camera Feed     │  STATUS  │  Info Panel            │
│  (640×480)       │  (big    │  - Sizes found         │
│  bbox overlay    │  circle) │  - Error detail        │
│                  │          │  - Stats               │
├──────────────────┴──────────┴────────────────────────┤
│  [  Inspect  ]   [  Teaching Mode  ]                 │
└─────────────────────────────────────────────────────┘

Shortcuts: Space → Inspect, Escape → Clear/Idle
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

try:
    from src.core.models import InspectionResult, LabelDetection, LabelType, ErrorCode
except ImportError:  # allow standalone testing
    from core.models import InspectionResult, LabelDetection, LabelType, ErrorCode  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
_GREEN = "#2ECC71"
_RED = "#E74C3C"
_GRAY = "#95A5A6"
_DARK_BG = "#1A1A2E"
_PANEL_BG = "#16213E"
_TEXT = "#EAEAEA"


# ---------------------------------------------------------------------------
# Pipeline runner thread
# ---------------------------------------------------------------------------
class _PipelineThread(QThread):
    """Runs the inspection pipeline in a background thread."""

    result_ready = pyqtSignal(object)   # InspectionResult
    error_raised = pyqtSignal(str)

    def __init__(self, pipeline, frame: np.ndarray) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._frame = frame

    def run(self) -> None:  # noqa: D401
        try:
            result: InspectionResult = self._pipeline.run(self._frame)
            self.result_ready.emit(result)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Pipeline error: %s", exc)
            self.error_raised.emit(str(exc))


# ---------------------------------------------------------------------------
# Status indicator widget
# ---------------------------------------------------------------------------
class _StatusCircle(QWidget):
    """Paints a large coloured filled circle with centred text."""

    _IDLE = _GRAY
    _PASS = _GREEN
    _FAIL = _RED

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._color = self._IDLE
        self._text = "IDLE"
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_idle(self) -> None:
        self._color, self._text = self._IDLE, "IDLE"
        self.update()

    def set_pass(self) -> None:
        self._color, self._text = self._PASS, "PASS"
        self.update()

    def set_fail(self) -> None:
        self._color, self._text = self._FAIL, "NG"
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height()) - 10
        x = (self.width() - side) // 2
        y = (self.height() - side) // 2

        painter.setBrush(QColor(self._color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(x, y, side, side)

        font = QFont("Arial", side // 4, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor("white"))
        painter.drawText(x, y, side, side, Qt.AlignmentFlag.AlignCenter, self._text)
        painter.end()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class InspectionWindow(QMainWindow):
    """
    Main inspection window for factory-line workers.

    Parameters
    ----------
    pipeline:
        Object with a ``run(frame: np.ndarray) -> InspectionResult`` method
        and optionally a ``camera`` attribute with a ``read() -> (bool, ndarray)``
        method (OpenCV-compatible).
    alarm_controller:
        Object with ``beep_pass()`` and ``alarm_ng()`` methods.
    """

    def __init__(self, pipeline, alarm_controller) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._alarm = alarm_controller
        self._last_frame: Optional[np.ndarray] = None
        self._last_result: Optional[InspectionResult] = None
        self._thread: Optional[_PipelineThread] = None
        self._stats_pass = 0
        self._stats_total = 0

        # Camera live-view timer (30 fps)
        self._cam_timer = QTimer(self)
        self._cam_timer.setInterval(33)
        self._cam_timer.timeout.connect(self._refresh_camera)

        self._init_ui()
        self._setup_shortcuts()
        self._update_title()

        # Try to start live camera preview
        if hasattr(self._pipeline, "camera") and self._pipeline.camera is not None:
            self._cam_timer.start()
        else:
            self._show_placeholder()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        self.setWindowTitle("AI Size Inspector v1.0")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(f"background-color: {_DARK_BG}; color: {_TEXT};")

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # ── Top content row ────────────────────────────────────────────
        content_row = QHBoxLayout()
        content_row.setSpacing(8)
        root_layout.addLayout(content_row, stretch=1)

        # Left: camera feed
        self._feed_label = QLabel()
        self._feed_label.setFixedSize(640, 480)
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setStyleSheet(
            "background-color: #000; border: 2px solid #555;"
        )
        content_row.addWidget(self._feed_label)

        # Centre: status indicator
        centre_col = QVBoxLayout()
        centre_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_circle = _StatusCircle()
        centre_col.addWidget(self._status_circle)
        content_row.addLayout(centre_col)

        # Right: info panel
        info_panel = self._build_info_panel()
        content_row.addWidget(info_panel, stretch=1)

        # ── Bottom button row ──────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)
        root_layout.addLayout(btn_row)

        self._btn_inspect = QPushButton("▶  Inspect  [Space]")
        self._btn_inspect.setFixedHeight(56)
        self._btn_inspect.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self._btn_inspect.setStyleSheet(
            "QPushButton { background-color: #2980B9; color: white; border-radius: 6px; }"
            "QPushButton:hover { background-color: #3498DB; }"
            "QPushButton:disabled { background-color: #555; }"
        )
        self._btn_inspect.clicked.connect(self.on_inspect_triggered)
        btn_row.addWidget(self._btn_inspect)

        self._btn_teach = QPushButton("🎓  Teaching Mode")
        self._btn_teach.setFixedHeight(56)
        self._btn_teach.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self._btn_teach.setStyleSheet(
            "QPushButton { background-color: #8E44AD; color: white; border-radius: 6px; }"
            "QPushButton:hover { background-color: #9B59B6; }"
            "QPushButton:disabled { background-color: #555; }"
        )
        self._btn_teach.setEnabled(False)
        self._btn_teach.clicked.connect(self._on_teach_clicked)
        btn_row.addWidget(self._btn_teach)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — press Space to inspect.")

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {_PANEL_BG}; border-radius: 8px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Sizes found
        sizes_box = QGroupBox("Sizes Found")
        sizes_box.setStyleSheet("QGroupBox { color: #AAA; border: 1px solid #444; }")
        sizes_layout = QVBoxLayout(sizes_box)
        self._sizes_label = QLabel("—")
        self._sizes_label.setWordWrap(True)
        self._sizes_label.setFont(QFont("Consolas", 13))
        self._sizes_label.setStyleSheet(f"color: {_TEXT};")
        sizes_layout.addWidget(self._sizes_label)
        layout.addWidget(sizes_box)

        # Error detail
        err_box = QGroupBox("Error Detail")
        err_box.setStyleSheet("QGroupBox { color: #AAA; border: 1px solid #444; }")
        err_layout = QVBoxLayout(err_box)
        self._error_label = QLabel("—")
        self._error_label.setWordWrap(True)
        self._error_label.setFont(QFont("Arial", 11))
        self._error_label.setStyleSheet("color: #E74C3C;")
        err_layout.addWidget(self._error_label)
        layout.addWidget(err_box)

        # Stats
        stats_box = QGroupBox("Today's Statistics")
        stats_box.setStyleSheet("QGroupBox { color: #AAA; border: 1px solid #444; }")
        stats_layout = QVBoxLayout(stats_box)
        self._stats_label = QLabel("PASS: 0 / TOTAL: 0  (—%)")
        self._stats_label.setFont(QFont("Arial", 12))
        self._stats_label.setStyleSheet(f"color: {_TEXT};")
        stats_layout.addWidget(self._stats_label)
        layout.addWidget(stats_box)

        layout.addStretch()
        return panel

    def _setup_shortcuts(self) -> None:
        space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space.activated.connect(self.on_inspect_triggered)

        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self._clear_idle)

    # ------------------------------------------------------------------
    # Camera / frame helpers
    # ------------------------------------------------------------------

    def _refresh_camera(self) -> None:
        """Read one frame from the camera and update the feed label."""
        try:
            ok, frame = self._pipeline.camera.read()
            if ok and frame is not None:
                self._last_frame = frame.copy()
                self._display_frame(frame)
        except Exception:  # pylint: disable=broad-except
            pass

    def _display_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[LabelDetection]] = None,
    ) -> None:
        """Convert *frame* (BGR) to QPixmap and show it in the feed label."""
        if detections:
            frame = self.draw_detections(frame.copy(), detections)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            640, 480, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._feed_label.setPixmap(pix)

    def _show_placeholder(self) -> None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "No Camera",
            (220, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (100, 100, 100),
            2,
        )
        self._display_frame(placeholder)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def on_inspect_triggered(self) -> None:
        """Capture the current frame and run the inspection pipeline."""
        if self._thread is not None and self._thread.isRunning():
            return  # already running

        # Grab frame
        if self._last_frame is not None:
            frame = self._last_frame.copy()
        elif hasattr(self._pipeline, "camera") and self._pipeline.camera is not None:
            ok, frame = self._pipeline.camera.read()
            if not ok or frame is None:
                self._status_bar.showMessage("❌ Failed to capture frame.")
                return
        else:
            self._status_bar.showMessage("❌ No camera / frame available.")
            return

        self._btn_inspect.setEnabled(False)
        self._status_bar.showMessage("⏳ Inspecting…")

        self._thread = _PipelineThread(self._pipeline, frame)
        self._thread.result_ready.connect(self.update_display)
        self._thread.error_raised.connect(self._on_pipeline_error)
        self._thread.finished.connect(lambda: self._btn_inspect.setEnabled(True))
        self._thread.start()

    def update_display(self, result: InspectionResult) -> None:
        """Update all UI elements based on *result*."""
        self._last_result = result

        # Status indicator
        if result.is_pass:
            self._status_circle.set_pass()
            self._alarm.beep_pass()
            self._stats_pass += 1
        else:
            self._status_circle.set_fail()
            self._alarm.alarm_ng()

        self._stats_total += 1

        # Sizes
        if result.sizes_found:
            sizes_text = "\n".join(
                f"{ltype}: {sz}" for ltype, sz in result.sizes_found.items()
            )
        else:
            sizes_text = "No sizes detected"
        self._sizes_label.setText(sizes_text)

        # Error
        if result.is_pass:
            self._error_label.setText("—")
            self._error_label.setStyleSheet("color: #2ECC71;")
        else:
            self._error_label.setText(
                f"{result.error_code.value}\n{result.error_detail}"
            )
            self._error_label.setStyleSheet("color: #E74C3C;")

        # Update stats
        self.update_stats()

        # Update camera feed with detections
        if self._last_frame is not None:
            self._display_frame(self._last_frame, result.detections)

        # Teaching mode: enable button if any detection needs teaching
        needs_teaching = any(d.needs_teaching for d in result.detections)
        self._btn_teach.setEnabled(needs_teaching)

        latency_text = f"{result.latency_ms:.1f} ms"
        status = "✅ PASS" if result.is_pass else "❌ NG"
        self._status_bar.showMessage(f"{status}  |  Latency: {latency_text}")
        self._update_title()

    def show_teaching_mode(self, session) -> None:
        """Open the TeachingWindow for *session*."""
        try:
            from src.ui.teaching_window import TeachingWindow
        except ImportError:
            from ui.teaching_window import TeachingWindow  # type: ignore

        dlg = TeachingWindow(session, getattr(self._pipeline, "session_manager", None), parent=self)
        dlg.exec()

    def draw_detections(
        self, frame: np.ndarray, detections: List[LabelDetection]
    ) -> np.ndarray:
        """Draw coloured bounding boxes and labels onto *frame* (BGR). Returns modified copy."""
        COLOR_MAP = {
            LabelType.SIZE_LABEL: (0, 255, 0),    # green
            LabelType.CARE_LABEL: (255, 165, 0),  # orange
            LabelType.HANG_TAG:   (0, 191, 255),  # sky-blue
            LabelType.UNKNOWN:    (0, 0, 255),    # red
        }
        for det in detections:
            color = COLOR_MAP.get(det.label_type, (200, 200, 200))
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_parts = [det.label_type.value, f"{det.confidence:.2f}"]
            if det.size_value:
                label_parts.append(det.size_value)
            text = " | ".join(label_parts)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return frame

    def update_stats(self) -> None:
        pct = (
            f"{self._stats_pass / self._stats_total * 100:.1f}%"
            if self._stats_total > 0
            else "—%"
        )
        self._stats_label.setText(
            f"PASS: {self._stats_pass} / TOTAL: {self._stats_total}  ({pct})"
        )

    # ------------------------------------------------------------------
    # Private slots / helpers
    # ------------------------------------------------------------------

    def _on_teach_clicked(self) -> None:
        if self._last_result is None:
            return
        for det in self._last_result.detections:
            if det.needs_teaching:
                # Build a minimal teaching session dict
                session = {"detection": det, "timestamp": time.time()}
                self.show_teaching_mode(session)
                break

    def _on_pipeline_error(self, msg: str) -> None:
        self._status_circle.set_fail()
        self._status_bar.showMessage(f"❌ Pipeline error: {msg}")
        self._btn_inspect.setEnabled(True)

    def _clear_idle(self) -> None:
        self._status_circle.set_idle()
        self._sizes_label.setText("—")
        self._error_label.setText("—")
        self._error_label.setStyleSheet("color: #E74C3C;")
        self._last_result = None
        self._btn_teach.setEnabled(False)
        self._status_bar.showMessage("Ready — press Space to inspect.")

    def _update_title(self) -> None:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
            else:
                gpu_name = "CPU"
        except Exception:  # pylint: disable=broad-except
            gpu_name = "Unknown"

        pct = (
            f"{self._stats_pass / self._stats_total * 100:.0f}%"
            if self._stats_total > 0
            else "—"
        )
        self.setWindowTitle(
            f"AI Size Inspector v1.0  |  GPU: {gpu_name}  |  Today: "
            f"{self._stats_pass}/{self._stats_total} PASS ({pct})"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cam_timer.stop()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        super().closeEvent(event)
