"""
Teaching Window — PyQt6 dialog for teaching the AI about unknown label types.

Flow
----
1. Worker sees an unknown label detected → clicks "Teaching Mode"
2. TeachingWindow opens showing the cropped label image
3. Worker draws a rectangle over the size text region
4. OCR preview updates live for that region
5. Worker selects / types the correct size, optionally fills brand/PO
6. "Confirm & Teach AI" saves the template to the knowledge base
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canvas widget
# ---------------------------------------------------------------------------

class LabelImageCanvas(QWidget):
    """
    Displays a label image and lets the user draw a rubber-band ROI rectangle.

    Signals
    -------
    roi_selected(x, y, w, h):
        Emitted on mouse-release with normalised (0–1) coordinates of the
        selected region, relative to the displayed image.
    """

    roi_selected = pyqtSignal(float, float, float, float)

    def __init__(
        self,
        image: np.ndarray,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._orig: np.ndarray = image  # BGR
        self._pixmap: QPixmap = self._to_pixmap(image)
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self._roi: Optional[Tuple[float, float, float, float]] = None

        self.setMinimumSize(400, 300)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def roi(self) -> Optional[Tuple[float, float, float, float]]:
        """Last confirmed ROI as (x, y, w, h) normalised 0–1, or None."""
        return self._roi

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _scaled_pixmap(self) -> Tuple[QPixmap, int, int]:
        """Return the pixmap scaled to current widget size, plus x/y offsets."""
        pix = self._pixmap.scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x_off = (self.width() - pix.width()) // 2
        y_off = (self.height() - pix.height()) // 2
        return pix, x_off, y_off

    def _widget_to_norm(
        self, pt: QPoint, x_off: int, y_off: int, pw: int, ph: int
    ) -> Tuple[float, float]:
        """Convert widget coordinates to normalised image coordinates."""
        nx = max(0.0, min(1.0, (pt.x() - x_off) / pw))
        ny = max(0.0, min(1.0, (pt.y() - y_off) / ph))
        return nx, ny

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        pix, x_off, y_off = self._scaled_pixmap()
        painter.drawPixmap(x_off, y_off, pix)

        if self._start and self._end:
            pen = QPen(QColor("#F1C40F"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            rect = QRect(self._start, self._end).normalized()
            painter.drawRect(rect)

        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.pos()
            self._end = event.pos()
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._start is not None:
            self._end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._start:
            self._end = event.pos()
            self.update()

            _, x_off, y_off = self._scaled_pixmap()
            pix, _, _ = self._scaled_pixmap()
            pw, ph = pix.width(), pix.height()

            x1n, y1n = self._widget_to_norm(self._start, x_off, y_off, pw, ph)
            x2n, y2n = self._widget_to_norm(self._end, x_off, y_off, pw, ph)

            # Ensure (x, y) is top-left
            x = min(x1n, x2n)
            y = min(y1n, y2n)
            w = abs(x2n - x1n)
            h = abs(y2n - y1n)

            if w > 0.01 and h > 0.01:  # ignore tiny accidental clicks
                self._roi = (x, y, w, h)
                self.roi_selected.emit(x, y, w, h)


# ---------------------------------------------------------------------------
# Teaching window
# ---------------------------------------------------------------------------

class TeachingWindow(QDialog):
    """
    Teaching mode dialog.

    Parameters
    ----------
    teaching_session:
        A dict (or object) with at least:
        - ``"detection"``: a ``LabelDetection`` instance whose ``cropped_image``
          is the BGR numpy array to annotate.
        - ``"session_id"`` (optional str): unique ID for this session.
        - ``"candidates"`` (optional list[str]): OCR size candidates.
    session_manager:
        Object with a ``confirm_teaching(session_id, roi, size)`` method, or None.
    """

    teaching_confirmed = pyqtSignal(str, tuple, str)  # session_id, roi, confirmed_size

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        teaching_session,
        session_manager,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._session = teaching_session
        self._sm = session_manager
        self._roi: Optional[Tuple[float, float, float, float]] = None

        # Resolve image and metadata from session
        if isinstance(teaching_session, dict):
            det = teaching_session.get("detection")
            self._session_id: str = str(teaching_session.get("session_id", "unknown"))
            self._candidates: list[str] = list(teaching_session.get("candidates", []))
        else:
            det = getattr(teaching_session, "detection", None)
            self._session_id = str(getattr(teaching_session, "session_id", "unknown"))
            self._candidates = list(getattr(teaching_session, "candidates", []))

        if det is not None and hasattr(det, "cropped_image") and det.cropped_image is not None:
            self._image: np.ndarray = det.cropped_image
        else:
            # Fallback: grey placeholder
            self._image = np.full((300, 400, 3), 180, dtype=np.uint8)
            cv2.putText(
                self._image, "No Image", (120, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2,
            )

        self._init_ui()
        self._populate_candidates()
        self.setMinimumSize(900, 620)
        self.setWindowTitle("Teaching Mode — Mark Size Location")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        self.setStyleSheet(
            "QDialog { background-color: #1A1A2E; color: #EAEAEA; }"
            "QGroupBox { border: 1px solid #444; color: #AAA; margin-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
            "QLineEdit { background: #16213E; color: #EEE; border: 1px solid #555; padding: 4px; border-radius: 4px; }"
            "QListWidget { background: #16213E; color: #EEE; border: 1px solid #555; }"
            "QLabel { color: #EAEAEA; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header
        header = QLabel("🏷  New label type detected!  Please mark the SIZE location.")
        header.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        header.setStyleSheet("color: #F39C12; padding: 6px;")
        root.addWidget(header)

        # Main area: canvas + right panel
        main_row = QHBoxLayout()
        main_row.setSpacing(10)
        root.addLayout(main_row, stretch=1)

        # Canvas
        canvas_group = QGroupBox("Label Image — click & drag to mark the size region")
        canvas_layout = QVBoxLayout(canvas_group)
        self._canvas = LabelImageCanvas(self._image)
        self._canvas.roi_selected.connect(self._on_roi_selected)
        canvas_layout.addWidget(self._canvas)
        main_row.addWidget(canvas_group, stretch=2)

        # Right panel
        right_panel = self._build_right_panel()
        main_row.addWidget(right_panel, stretch=1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        root.addLayout(btn_row)

        self._btn_confirm = QPushButton("✅  Confirm & Teach AI")
        self._btn_confirm.setFixedHeight(48)
        self._btn_confirm.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        self._btn_confirm.setStyleSheet(
            "QPushButton { background-color: #27AE60; color: white; border-radius: 6px; }"
            "QPushButton:hover { background-color: #2ECC71; }"
            "QPushButton:disabled { background-color: #555; }"
        )
        self._btn_confirm.clicked.connect(self._on_confirm)
        btn_row.addWidget(self._btn_confirm)

        btn_skip = QPushButton("❌  Skip")
        btn_skip.setFixedHeight(48)
        btn_skip.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        btn_skip.setStyleSheet(
            "QPushButton { background-color: #C0392B; color: white; border-radius: 6px; }"
            "QPushButton:hover { background-color: #E74C3C; }"
        )
        btn_skip.clicked.connect(self.reject)
        btn_row.addWidget(btn_skip)

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # OCR preview
        ocr_box = QGroupBox("OCR Preview (selected region)")
        ocr_layout = QVBoxLayout(ocr_box)
        self._ocr_preview = QLabel("Draw a rectangle to see OCR here.")
        self._ocr_preview.setWordWrap(True)
        self._ocr_preview.setFont(QFont("Consolas", 11))
        self._ocr_preview.setStyleSheet("color: #3498DB; padding: 4px;")
        ocr_layout.addWidget(self._ocr_preview)
        layout.addWidget(ocr_box)

        # Candidates list
        cand_box = QGroupBox("Size Candidates (click to select)")
        cand_layout = QVBoxLayout(cand_box)
        self._candidates_list = QListWidget()
        self._candidates_list.setMaximumHeight(150)
        self._candidates_list.itemClicked.connect(self._on_candidate_clicked)
        cand_layout.addWidget(self._candidates_list)
        layout.addWidget(cand_box)

        # Manual size input
        size_box = QGroupBox("Confirmed Size")
        size_layout = QVBoxLayout(size_box)
        self._size_input = QLineEdit()
        self._size_input.setPlaceholderText("e.g.  M, L, XL, 32, 42/44 …")
        self._size_input.setFont(QFont("Arial", 13))
        size_layout.addWidget(self._size_input)
        layout.addWidget(size_box)

        # Brand / PO
        meta_box = QGroupBox("Optional Metadata")
        meta_layout = QVBoxLayout(meta_box)
        self._brand_input = QLineEdit()
        self._brand_input.setPlaceholderText("Brand code (optional)")
        meta_layout.addWidget(QLabel("Brand:"))
        meta_layout.addWidget(self._brand_input)
        self._po_input = QLineEdit()
        self._po_input.setPlaceholderText("PO code (optional)")
        meta_layout.addWidget(QLabel("PO:"))
        meta_layout.addWidget(self._po_input)
        layout.addWidget(meta_box)

        layout.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _populate_candidates(self) -> None:
        self._candidates_list.clear()
        for cand in self._candidates:
            self._candidates_list.addItem(QListWidgetItem(str(cand)))

    def _run_ocr_on_roi(
        self, roi: Tuple[float, float, float, float]
    ) -> str:
        """Crop the ROI from the image and run OCR, returning text or empty string."""
        x, y, w, h = roi
        ih, iw = self._image.shape[:2]
        x1 = max(0, int(x * iw))
        y1 = max(0, int(y * ih))
        x2 = min(iw, int((x + w) * iw))
        y2 = min(ih, int((y + h) * ih))
        crop = self._image[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        try:
            from paddleocr import PaddleOCR  # type: ignore
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            result = ocr.ocr(crop, cls=True)
            texts = []
            for line in (result or []):
                for item in (line or []):
                    if isinstance(item, list) and len(item) == 2:
                        txt = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                        texts.append(txt)
            return " ".join(texts).strip()
        except Exception:  # pylint: disable=broad-except
            # Fallback: return raw pixel mean as placeholder
            return f"[OCR unavailable] crop {x2 - x1}×{y2 - y1}px"

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_roi_selected(
        self, x: float, y: float, w: float, h: float
    ) -> None:
        self._roi = (x, y, w, h)
        self._ocr_preview.setText("⏳ Running OCR on selection…")
        # Run OCR (synchronous for simplicity; wrap in QThread for large images)
        text = self._run_ocr_on_roi((x, y, w, h))
        if text:
            self._ocr_preview.setText(text)
            # Auto-add to candidates if not already there
            existing = [
                self._candidates_list.item(i).text()
                for i in range(self._candidates_list.count())
            ]
            if text not in existing:
                self._candidates_list.addItem(QListWidgetItem(text))
        else:
            self._ocr_preview.setText("(no text found in selection)")

    def _on_candidate_clicked(self, item: QListWidgetItem) -> None:
        self._size_input.setText(item.text())

    def _on_confirm(self) -> None:
        size = self._size_input.text().strip()
        if not size:
            QMessageBox.warning(self, "Missing Size", "Please enter or select the confirmed size.")
            return

        roi = self._roi if self._roi else self._canvas.roi
        if roi is None:
            reply = QMessageBox.question(
                self,
                "No ROI Selected",
                "No region of interest was drawn. Teach without an ROI?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            roi = (0.0, 0.0, 1.0, 1.0)  # full image as fallback

        # Notify session manager if available
        if self._sm is not None and hasattr(self._sm, "confirm_teaching"):
            try:
                self._sm.confirm_teaching(
                    self._session_id,
                    roi,
                    size,
                    brand=self._brand_input.text().strip() or None,
                    po=self._po_input.text().strip() or None,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Session manager error: %s", exc)

        self.teaching_confirmed.emit(self._session_id, roi, size)
        logger.info(
            "Teaching confirmed — session=%s size=%s roi=%s",
            self._session_id, size, roi,
        )
        self.accept()
