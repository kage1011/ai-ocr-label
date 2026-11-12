"""
src/core/judge.py
ConsistencyJudge — evaluates a list of LabelDetection objects and decides
whether a garment passes or fails the size-consistency check.
"""
from __future__ import annotations

import logging
from typing import List, Set

from .models import ErrorCode, JudgeResult, LabelDetection, LabelType

logger = logging.getLogger(__name__)


class ConsistencyJudge:
    """
    Checks that:
    1. All required label types are present in the detection list.
    2. Every label that *should* carry a size has a readable OCR result.
    3. Every readable size value agrees across all labels (no mismatches).

    Usage::

        judge = ConsistencyJudge(
            required_label_types=["size_label", "care_label", "hang_tag"]
        )
        result = judge.judge(detections)
    """

    # Label types that are expected to carry a printable size mark
    SIZE_BEARING_TYPES: Set[LabelType] = {LabelType.SIZE_LABEL, LabelType.HANG_TAG}

    def __init__(self, required_label_types: List[str] | None = None) -> None:
        """
        Args:
            required_label_types: List of label type *string values* that must
                all be present on a passing garment.  Defaults to
                ["size_label", "care_label", "hang_tag"].
        """
        if required_label_types is None:
            required_label_types = ["size_label", "care_label", "hang_tag"]
        self.required: List[LabelType] = [
            LabelType(t) for t in required_label_types
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(self, detections: List[LabelDetection]) -> JudgeResult:
        """
        Evaluate a list of detections from a single garment frame.

        Args:
            detections: All label regions detected by the pipeline for the
                current frame.

        Returns:
            :class:`JudgeResult` with ``is_pass=True`` when no errors are
            found, or ``is_pass=False`` with an appropriate
            :class:`ErrorCode` and human-readable ``error_detail``.
        """
        # --- 1. Missing required labels ---------------------------------
        present_types: Set[LabelType] = {d.label_type for d in detections}
        missing = [
            lt.value for lt in self.required if lt not in present_types
        ]
        if missing:
            detail = f"Missing required label(s): {', '.join(missing)}"
            logger.warning("JUDGE FAIL — %s", detail)
            return JudgeResult(
                is_pass=False,
                error_code=ErrorCode.MISSING_REQUIRED_LABEL,
                error_detail=detail,
                missing_labels=missing,
            )

        # --- 2. Unreadable sizes ----------------------------------------
        size_bearing = [
            d for d in detections if d.label_type in self.SIZE_BEARING_TYPES
        ]
        unreadable = [d for d in size_bearing if not d.has_readable_size()]
        if unreadable:
            types_unread = [d.label_type.value for d in unreadable]
            detail = (
                f"Could not read size from label(s): {', '.join(types_unread)}"
            )
            logger.warning("JUDGE FAIL — %s", detail)
            return JudgeResult(
                is_pass=False,
                error_code=ErrorCode.SIZE_NOT_FOUND,
                error_detail=detail,
            )

        # --- 3. Build sizes dict ----------------------------------------
        sizes_dict: dict[str, str] = {}
        for d in size_bearing:
            if d.has_readable_size():
                sizes_dict[d.label_type.value] = d.size_value.strip().upper()

        # --- 4. Size mismatch check -------------------------------------
        unique_sizes = set(sizes_dict.values())
        if len(unique_sizes) > 1:
            detail = (
                f"Size mismatch across labels — found: {sizes_dict}"
            )
            logger.warning("JUDGE FAIL — %s", detail)
            return JudgeResult(
                is_pass=False,
                error_code=ErrorCode.SIZE_MISMATCH,
                error_detail=detail,
                sizes_dict=sizes_dict,
            )

        # --- 5. All checks passed ---------------------------------------
        logger.debug("JUDGE PASS — sizes=%s", sizes_dict)
        return JudgeResult(
            is_pass=True,
            error_code=ErrorCode.PASS,
            error_detail="All checks passed",
            sizes_dict=sizes_dict,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_size(self, raw: str) -> str:
        """
        Normalise raw OCR text to a canonical size token.

        Examples: "  XL  " → "XL", "extra large" → "XL"
        """
        raw = raw.strip().upper()
        # Basic alias map; extend as needed
        aliases = {
            "EXTRA SMALL": "XS",
            "EXTRA LARGE": "XL",
            "DOUBLE EXTRA LARGE": "XXL",
            "2XL": "XXL",
            "3XL": "XXXL",
        }
        return aliases.get(raw, raw)
