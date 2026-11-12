"""
ROI-guided size extraction using PaddleOCR.

The extractor crops a normalised ROI from the label image, runs OCR only on
that small region, then parses the resulting text to find a garment size string.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard size patterns
# ---------------------------------------------------------------------------

# Named alphabetic sizes (longest first so regex alternation matches greedily)
_ALPHA_SIZES = [
    "4XL", "3XL", "2XL", "XXL", "XL", "XXXL",
    "XS", "S", "M", "L", "F",
]

# Numeric size ranges commonly found on garment labels
_NUMERIC_PATTERNS = [
    r"\b(5[0-9]|4[0-9]|3[6-9])\b",   # EU: 36-59
    r"\b([1-2][0-9]|[6-9])\b",         # US women: 6-29  (also catches some waist)
    r"\b(2[89]|3[0-9]|4[0-4])\b",      # Waist: 28-44
]

# Compiled pattern: alpha first, then numeric
_ALPHA_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _ALPHA_SIZES) + r")\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(
    r"\b([3-5][0-9]|[6-9]|[1-2][0-9]|2[89])\b"
)


class SizeExtractor:
    """
    Extract garment size text from a label image using PaddleOCR.

    Parameters
    ----------
    config : dict
        Expected keys: ``lang`` (str, default "en"),
        ``use_gpu`` (bool, default False),
        ``padding_px`` (int, default 10).
    """

    def __init__(self, config: dict) -> None:
        self.lang: str = config.get("lang", "en")
        self.use_gpu: bool = bool(config.get("use_gpu", False))
        self.padding_px: int = int(config.get("padding_px", 10))
        self._ocr = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_size(
        self,
        label_image: np.ndarray,
        roi_normalized: tuple,
        ocr_hint: str = "",
        regex_pattern: str = "",
    ) -> tuple[Optional[str], float]:
        """
        Extract the size string from a region-of-interest on the label image.

        Parameters
        ----------
        label_image : np.ndarray
            Full label crop (BGR, H×W×3).
        roi_normalized : tuple
            (x, y, w, h) in [0, 1] relative to ``label_image``.
        ocr_hint : str
            Free-text hint (currently unused at runtime but logged for debug).
        regex_pattern : str
            Custom regex to find the size in raw OCR text.

        Returns
        -------
        (size_string, confidence) : tuple[Optional[str], float]
            ``size_string`` is the normalised uppercase size, or ``None``.
            ``confidence`` is the OCR engine's mean confidence for the matched
            token (0.0 when nothing was found).
        """
        if label_image is None or label_image.size == 0:
            logger.warning("extract_size called with empty image.")
            return None, 0.0

        try:
            crop = self._crop_roi(label_image, roi_normalized, self.padding_px)
        except Exception as exc:
            logger.error("ROI crop failed: %s", exc, exc_info=True)
            return None, 0.0

        if crop is None or crop.size == 0:
            logger.warning("Crop is empty after applying ROI %s", roi_normalized)
            return None, 0.0

        try:
            ocr = self._get_ocr()
            result = ocr.ocr(crop, cls=True)
        except Exception as exc:
            logger.error("PaddleOCR inference failed: %s", exc, exc_info=True)
            return None, 0.0

        if not result or not result[0]:
            logger.debug("OCR returned no text for roi=%s hint=%s", roi_normalized, ocr_hint)
            return None, 0.0

        # Collect all text lines with their confidences
        texts: list[str] = []
        confidences: list[float] = []
        for line in result[0]:
            # line = [box_points, (text, confidence)]
            if line and len(line) >= 2:
                txt, conf = line[1]
                texts.append(str(txt))
                confidences.append(float(conf))

        full_text = " ".join(texts)
        logger.debug("OCR raw text from ROI: %r  (hint: %r)", full_text, ocr_hint)

        size = self._parse_size(full_text, regex_pattern)
        if size is None:
            return None, 0.0

        # Confidence = mean of all OCR line confidences (best proxy available)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        logger.debug("Parsed size=%s  confidence=%.3f", size, mean_conf)
        return size, mean_conf

    def extract_full_text(self, image: np.ndarray) -> str:
        """
        Run OCR over the entire image and return a concatenated string.
        Intended for debugging and teaching-mode previews.

        Parameters
        ----------
        image : np.ndarray
            Any BGR image.

        Returns
        -------
        str
            All recognised text lines joined by newlines.
        """
        if image is None or image.size == 0:
            return ""
        try:
            ocr = self._get_ocr()
            result = ocr.ocr(image, cls=True)
        except Exception as exc:
            logger.error("extract_full_text OCR failed: %s", exc, exc_info=True)
            return ""

        if not result or not result[0]:
            return ""

        lines = [line[1][0] for line in result[0] if line and len(line) >= 2]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_ocr(self):
        """Lazy-load PaddleOCR to avoid slow import at startup."""
        if self._ocr is None:
            logger.info("Initialising PaddleOCR (lang=%s, gpu=%s)…", self.lang, self.use_gpu)
            from paddleocr import PaddleOCR  # noqa: PLC0415
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR ready.")
        return self._ocr

    def _crop_roi(self, image: np.ndarray, roi: tuple, padding: int) -> np.ndarray:
        """
        Crop a normalised ROI from the image with optional pixel padding.

        Parameters
        ----------
        image : np.ndarray
            Source image (H, W, C) or (H, W).
        roi : tuple
            (x, y, w, h) normalised to [0, 1].
        padding : int
            Extra pixels added on every side (clamped to image bounds).

        Returns
        -------
        np.ndarray
            Cropped sub-image.
        """
        h_img, w_img = image.shape[:2]
        x_n, y_n, w_n, h_n = roi

        x1 = int(x_n * w_img) - padding
        y1 = int(y_n * h_img) - padding
        x2 = int((x_n + w_n) * w_img) + padding
        y2 = int((y_n + h_n) * h_img) + padding

        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid ROI crop bounds: x1={x1} y1={y1} x2={x2} y2={y2} "
                f"for image {w_img}×{h_img}"
            )

        return image[y1:y2, x1:x2]

    def _parse_size(self, text: str, regex_pattern: str = "") -> Optional[str]:
        """
        Parse a garment size from raw OCR text.

        Strategy (in order):
        1. If ``regex_pattern`` is given, try it first.
        2. Match standard alphabetic sizes (XS → 4XL, F, 2XL, 3XL, 4XL, XXXL …).
        3. Match numeric sizes (EU 36-54, US women 6-20, waist 28-44).

        Parameters
        ----------
        text : str
            Raw OCR output.
        regex_pattern : str
            User-supplied regex; the first capture group (or full match) is used.

        Returns
        -------
        Optional[str]
            Normalised UPPERCASE size string, or ``None``.
        """
        if not text:
            return None

        upper = text.upper().strip()

        # --- 1. Custom regex ---
        if regex_pattern:
            try:
                m = re.search(regex_pattern, upper, re.IGNORECASE)
                if m:
                    captured = m.group(1) if m.lastindex else m.group(0)
                    return captured.upper().strip()
            except re.error as exc:
                logger.warning("Invalid regex_pattern %r: %s", regex_pattern, exc)

        # --- 2. Alphabetic sizes ---
        m = _ALPHA_RE.search(upper)
        if m:
            return m.group(0).upper()

        # --- 3. Numeric sizes ---
        m = _NUMERIC_RE.search(upper)
        if m:
            return m.group(0)

        logger.debug("No size pattern matched in text: %r", text[:120])
        return None
