"""
Pydantic schemas for all API request and response bodies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

class InspectRequest(BaseModel):
    """Payload for POST /api/inspect/"""
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG image of the garment.")
    label_types: Optional[List[str]] = Field(
        default=None,
        description="Restrict inspection to these label types. If None, all configured types are checked.",
    )


class InspectResponse(BaseModel):
    """Result returned by POST /api/inspect/"""
    is_pass: bool = Field(..., description="True if all required labels pass inspection.")
    error_code: str = Field(..., description="PASS, SIZE_MISMATCH, MISSING_REQUIRED_LABEL, etc.")
    error_detail: str = Field(..., description="Human-readable description of any failure.")
    sizes_found: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of label_type -> size string extracted from that label.",
    )
    latency_ms: float = Field(..., description="End-to-end processing latency in milliseconds.")
    snapshot_url: Optional[str] = Field(
        default=None,
        description="URL to the saved annotated snapshot image, if snapshot saving is enabled.",
    )
    needs_teaching: bool = Field(
        default=False,
        description="True when at least one label could not be matched to a known template.",
    )
    teaching_session_id: Optional[str] = Field(
        default=None,
        description="UUID of the teaching session created for the unrecognised label.",
    )


# ---------------------------------------------------------------------------
# Teaching
# ---------------------------------------------------------------------------

class TeachConfirmRequest(BaseModel):
    """Payload for POST /api/teach/sessions/{session_id}/confirm"""
    session_id: str = Field(..., description="UUID of the teaching session to confirm.")
    roi_x: float = Field(..., ge=0.0, le=1.0, description="Normalised X of the size-ROI top-left.")
    roi_y: float = Field(..., ge=0.0, le=1.0, description="Normalised Y of the size-ROI top-left.")
    roi_w: float = Field(..., ge=0.0, le=1.0, description="Normalised width of the size-ROI.")
    roi_h: float = Field(..., ge=0.0, le=1.0, description="Normalised height of the size-ROI.")
    confirmed_size: str = Field(..., description="Ground-truth size string confirmed by operator.")
    brand_code: str = Field(default="", description="Optional brand identifier.")
    po_code: str = Field(default="", description="Optional purchase-order identifier.")


class TeachConfirmResponse(BaseModel):
    """Result returned after a successful teaching confirmation."""
    success: bool
    template_id: int = Field(..., description="ID of the newly created template in the knowledge base.")
    message: str = Field(..., description="Human-readable confirmation message.")


class TeachSessionResponse(BaseModel):
    """Public representation of a pending teaching session (no raw image bytes)."""
    session_id: str
    label_type: str
    label_image_url: str = Field(..., description="URL path to the cropped label image for display.")
    suggested_text: str = Field(..., description="All OCR text found on the label.")
    size_candidates: List[str] = Field(
        default_factory=list,
        description="Possible size strings extracted by OCR.",
    )
    created_at: datetime


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

class TemplateInfo(BaseModel):
    """Summary of a single knowledge-base template."""
    id: int
    brand_code: str
    po_code: str
    label_type: str
    usage_count: int = Field(..., description="Number of times this template has been matched.")
    created_at: datetime


class TemplateListResponse(BaseModel):
    """Paginated list of templates."""
    templates: List[TemplateInfo]
    total: int = Field(..., description="Total number of templates matching the filter.")


# ---------------------------------------------------------------------------
# Dashboard / Stats
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    """Aggregate inspection statistics."""
    total_templates: int
    total_inspections: int
    pass_count: int
    ng_count: int
    pass_rate: float = Field(..., description="Pass count / total inspections (0–1). NaN if no inspections.")
    templates_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Map of label_type -> template count.",
    )
