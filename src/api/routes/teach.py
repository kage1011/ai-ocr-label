"""
Teaching router — /api/teach/
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    TeachConfirmRequest,
    TeachConfirmResponse,
    TeachSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/teach", tags=["teaching"])


def _session_to_response(session, http_request: Request) -> TeachSessionResponse:
    """Convert a TeachingSession dataclass to its API response schema."""
    # Build a web-accessible URL for the label crop image
    label_filename = session.label_image_path.replace("\\", "/").rsplit("/", 1)[-1]
    label_image_url = f"/teaching_temp/{label_filename}"

    return TeachSessionResponse(
        session_id=session.session_id,
        label_type=session.label_type,
        label_image_url=label_image_url,
        suggested_text=session.suggested_text,
        size_candidates=session.detected_size_candidates,
        created_at=session.created_at,
    )


@router.get("/sessions", response_model=List[TeachSessionResponse])
async def list_teaching_sessions(http_request: Request) -> List[TeachSessionResponse]:
    """Return all pending (incomplete) teaching sessions."""
    manager = http_request.app.state.teaching_manager
    sessions = manager.list_pending_sessions()
    return [_session_to_response(s, http_request) for s in sessions]


@router.get("/sessions/{session_id}", response_model=TeachSessionResponse)
async def get_teaching_session(session_id: str, http_request: Request) -> TeachSessionResponse:
    """Return a single teaching session by its UUID."""
    manager = http_request.app.state.teaching_manager
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Teaching session '{session_id}' not found.")
    return _session_to_response(session, http_request)


@router.post("/sessions/{session_id}/confirm", response_model=TeachConfirmResponse)
async def confirm_teaching(
    session_id: str,
    request: TeachConfirmRequest,
    http_request: Request,
) -> TeachConfirmResponse:
    """
    Confirm a teaching session: register the user-defined ROI and confirmed
    size as a new template in the knowledge base.
    """
    manager = http_request.app.state.teaching_manager

    roi_normalized = (request.roi_x, request.roi_y, request.roi_w, request.roi_h)

    try:
        template_id = manager.confirm_teaching(
            session_id=session_id,
            roi_normalized=roi_normalized,
            confirmed_size=request.confirmed_size,
            brand_code=request.brand_code,
            po_code=request.po_code,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Teaching session '{session_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Error confirming teaching session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    return TeachConfirmResponse(
        success=True,
        template_id=template_id,
        message=(
            f"Template {template_id} created for label type '{request.confirmed_size}'. "
            f"The pipeline will use it immediately."
        ),
    )
