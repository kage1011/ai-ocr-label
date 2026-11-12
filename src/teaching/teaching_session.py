"""
Teaching Session Manager
Handles unknown label types by guiding operators through a teaching workflow.
"""

from __future__ import annotations

import os
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TeachingSession:
    session_id: str
    label_type: str
    label_image: np.ndarray          # The unknown label crop
    label_image_path: str             # Saved temp path for UI display
    full_frame_path: str              # Full camera frame path
    bbox: tuple                       # Where label was found in frame
    suggested_text: str               # OCR full text from label (all text found)
    detected_size_candidates: List[str]  # Possible size strings found by OCR
    roi_normalized: Optional[tuple] = None   # Set by user during teaching (x, y, w, h)
    confirmed_size: Optional[str] = None     # Set by user during teaching
    brand_code: str = ""
    po_code: str = ""
    is_complete: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    # Internal: feature embedding extracted at session creation time
    _embedding: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


class TeachingSessionManager:
    """
    Manages the lifecycle of teaching sessions triggered when the pipeline
    encounters a label whose type cannot be matched to any existing template.
    """

    def __init__(
        self,
        knowledge_base,
        feature_extractor_fn: Callable[[np.ndarray], np.ndarray],
        size_extractor,
        temp_dir: str = "data/teaching_temp",
    ):
        """
        Args:
            knowledge_base:         Instance of KnowledgeBase (from knowledge_base.py).
            feature_extractor_fn:   Callable(image_np) -> embedding np.ndarray.
            size_extractor:         Instance of SizeExtractor (from size_extractor.py).
            temp_dir:               Directory for temporary label/frame images.
        """
        self._kb = knowledge_base
        self._extract_features = feature_extractor_fn
        self._size_extractor = size_extractor
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Active sessions: session_id -> TeachingSession
        self._active_sessions: Dict[str, TeachingSession] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(
        self,
        label_image: np.ndarray,
        label_type: str,
        full_frame: np.ndarray,
        bbox: tuple,
    ) -> TeachingSession:
        """
        Create a new teaching session for an unrecognised label.

        Steps:
        1. Generate a unique session_id.
        2. Save label crop and full frame to temp_dir.
        3. Run OCR (via size_extractor) to get raw text and size candidates.
        4. Extract feature embedding for later KB insertion.
        5. Store and return the TeachingSession.

        Args:
            label_image:  Cropped label image (BGR numpy array).
            label_type:   Detected label category string.
            full_frame:   Full camera frame (BGR numpy array).
            bbox:         (x1, y1, x2, y2) pixel coordinates in full_frame.

        Returns:
            TeachingSession populated with OCR text, size candidates, and paths.
        """
        session_id = str(uuid.uuid4())
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # --- Save images ---
        label_filename = f"label_{session_id}_{ts_str}.jpg"
        frame_filename = f"frame_{session_id}_{ts_str}.jpg"
        label_path = str(self.temp_dir / label_filename)
        frame_path = str(self.temp_dir / frame_filename)

        cv2.imwrite(label_path, label_image)
        cv2.imwrite(frame_path, full_frame)

        # --- OCR: get all text and size candidates ---
        suggested_text = ""
        size_candidates: List[str] = []
        try:
            ocr_result = self._size_extractor.extract_all_text(label_image)
            suggested_text = ocr_result.get("full_text", "")
            size_candidates = ocr_result.get("size_candidates", [])
        except Exception as exc:
            logger.warning("OCR failed during teaching session creation: %s", exc)

        # --- Feature embedding ---
        embedding: Optional[np.ndarray] = None
        try:
            embedding = self._extract_features(label_image)
        except Exception as exc:
            logger.warning("Feature extraction failed during teaching session: %s", exc)

        session = TeachingSession(
            session_id=session_id,
            label_type=label_type,
            label_image=label_image,
            label_image_path=label_path,
            full_frame_path=frame_path,
            bbox=bbox,
            suggested_text=suggested_text,
            detected_size_candidates=size_candidates,
            _embedding=embedding,
        )

        self._active_sessions[session_id] = session
        logger.info("Teaching session created: %s (label_type=%s)", session_id, label_type)
        return session

    def confirm_teaching(
        self,
        session_id: str,
        roi_normalized: tuple,
        confirmed_size: str,
        brand_code: str = "",
        po_code: str = "",
    ) -> int:
        """
        Finalise a teaching session: add a new template to the knowledge base.

        Args:
            session_id:     UUID of the teaching session.
            roi_normalized: (x, y, w, h) normalised to [0, 1] within the label crop.
            confirmed_size: The ground-truth size string confirmed by the operator.
            brand_code:     Optional brand identifier.
            po_code:        Optional purchase-order identifier.

        Returns:
            The new template_id assigned by the knowledge base.

        Raises:
            KeyError:  If session_id is not found.
            ValueError: If the session is already complete.
        """
        session = self._get_or_raise(session_id)
        if session.is_complete:
            raise ValueError(f"Session {session_id} is already complete.")

        # Persist the user-provided information on the session object
        session.roi_normalized = roi_normalized
        session.confirmed_size = confirmed_size
        session.brand_code = brand_code
        session.po_code = po_code

        # Add template to knowledge base
        template_id: int = self._kb.add_template(
            label_image=session.label_image,
            label_type=session.label_type,
            roi_normalized=roi_normalized,
            size_value=confirmed_size,
            brand_code=brand_code,
            po_code=po_code,
            embedding=session._embedding,
            reference_image_path=session.label_image_path,
        )

        session.is_complete = True
        logger.info(
            "Teaching confirmed: session=%s template_id=%d size=%s",
            session_id, template_id, confirmed_size,
        )
        return template_id

    def get_session(self, session_id: str) -> Optional[TeachingSession]:
        """Return the session or None if not found."""
        return self._active_sessions.get(session_id)

    def list_pending_sessions(self) -> List[TeachingSession]:
        """Return all sessions that have not yet been confirmed."""
        return [s for s in self._active_sessions.values() if not s.is_complete]

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Remove completed sessions older than *max_age_hours* and delete their
        temporary image files.

        Returns:
            Number of sessions removed.
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            sid
            for sid, s in self._active_sessions.items()
            if s.is_complete and s.created_at < cutoff
        ]

        removed = 0
        for sid in to_remove:
            session = self._active_sessions.pop(sid, None)
            if session is None:
                continue
            self._delete_temp_file(session.label_image_path)
            self._delete_temp_file(session.full_frame_path)
            removed += 1

        if removed:
            logger.info("Cleaned up %d old teaching sessions.", removed)
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, session_id: str) -> TeachingSession:
        session = self._active_sessions.get(session_id)
        if session is None:
            raise KeyError(f"Teaching session not found: {session_id}")
        return session

    @staticmethod
    def _delete_temp_file(path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            logger.warning("Could not delete temp file %s: %s", path, exc)
