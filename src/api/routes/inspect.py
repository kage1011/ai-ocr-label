"""
Inspection router — POST /api/inspect/
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

import numpy as np
import cv2
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import InspectRequest, InspectResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inspect", tags=["inspection"])


def _decode_image(image_base64: str) -> np.ndarray:
    """Decode a base64 image string to a BGR numpy array."""
    try:
        # Strip optional data-URI header (e.g. "data:image/jpeg;base64,...")
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_base64)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None — invalid image data.")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc


@router.post("/", response_model=InspectResponse)
async def inspect_label(request: InspectRequest, http_request: Request) -> InspectResponse:
    """
    Decode the submitted image, run the full inspection pipeline, and return
    a structured result.  If a label cannot be matched to any template, a
    teaching session is created and its ID is included in the response.
    """
    state = http_request.app.state

    # --- Decode image ---
    t0 = time.monotonic()
    image = _decode_image(request.image_base64)

    # --- Run pipeline ---
    try:
        result = await state.pipeline.run_async(image, label_types=request.label_types)
    except Exception as exc:
        logger.exception("Pipeline error during inspection: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    latency_ms = (time.monotonic() - t0) * 1000.0

    # --- Handle teaching trigger ---
    teaching_session_id: Optional[str] = None
    if result.needs_teaching and result.unknown_label_image is not None:
        try:
            session = state.teaching_manager.create_session(
                label_image=result.unknown_label_image,
                label_type=result.unknown_label_type or "unknown",
                full_frame=image,
                bbox=result.unknown_bbox or (0, 0, image.shape[1], image.shape[0]),
            )
            teaching_session_id = session.session_id
        except Exception as exc:
            logger.warning("Could not create teaching session: %s", exc)

    # --- Snapshot URL ---
    snapshot_url: Optional[str] = None
    if result.snapshot_path:
        # Convert local path to a web-accessible URL
        snapshot_filename = result.snapshot_path.replace("\\", "/").rsplit("/", 1)[-1]
        snapshot_url = f"/snapshots/{snapshot_filename}"

    return InspectResponse(
        is_pass=result.is_pass,
        error_code=result.error_code,
        error_detail=result.error_detail,
        sizes_found=result.sizes_found,
        latency_ms=round(latency_ms, 2),
        snapshot_url=snapshot_url,
        needs_teaching=result.needs_teaching,
        teaching_session_id=teaching_session_id,
    )


@router.get("/health")
async def health_check(http_request: Request) -> dict:
    """
    Lightweight liveness / readiness probe.

    Returns:
        ``{"status": "ok", "gpu_available": bool, "model_loaded": bool}``
    """
    state = http_request.app.state
    gpu_available = False
    model_loaded = False

    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        model_loaded = state.pipeline.is_ready()
    except Exception:
        pass

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "model_loaded": model_loaded,
    }
