"""
Templates router — /api/templates/
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import StatsResponse, TemplateInfo, TemplateListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/templates", tags=["templates"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(http_request: Request) -> StatsResponse:
    """
    Return aggregate inspection and template statistics.

    Note: /stats is declared before /{template_id} to avoid FastAPI treating
    'stats' as an integer path parameter.
    """
    kb = http_request.app.state.knowledge_base
    try:
        stats = kb.get_stats()
    except Exception as exc:
        logger.exception("Failed to retrieve stats: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not fetch stats: {exc}") from exc

    total = stats.get("total_inspections", 0)
    pass_count = stats.get("pass_count", 0)
    pass_rate = (pass_count / total) if total > 0 else float("nan")

    return StatsResponse(
        total_templates=stats.get("total_templates", 0),
        total_inspections=total,
        pass_count=pass_count,
        ng_count=stats.get("ng_count", 0),
        pass_rate=pass_rate,
        templates_by_type=stats.get("templates_by_type", {}),
    )


@router.get("/", response_model=TemplateListResponse)
async def list_templates(
    http_request: Request,
    label_type: Optional[str] = None,
    brand: Optional[str] = None,
) -> TemplateListResponse:
    """
    List all templates, optionally filtered by *label_type* and/or *brand*.
    """
    kb = http_request.app.state.knowledge_base
    try:
        raw = kb.list_templates(label_type=label_type, brand_code=brand)
    except Exception as exc:
        logger.exception("Failed to list templates: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not list templates: {exc}") from exc

    templates = [
        TemplateInfo(
            id=t["id"],
            brand_code=t.get("brand_code", ""),
            po_code=t.get("po_code", ""),
            label_type=t["label_type"],
            usage_count=t.get("usage_count", 0),
            created_at=t["created_at"],
        )
        for t in raw
    ]

    return TemplateListResponse(templates=templates, total=len(templates))


@router.delete("/{template_id}")
async def delete_template(template_id: int, http_request: Request) -> dict:
    """
    Permanently delete a template from the knowledge base.

    Returns ``{"deleted": true, "template_id": <id>}`` on success.
    """
    kb = http_request.app.state.knowledge_base
    try:
        deleted = kb.delete_template(template_id)
    except Exception as exc:
        logger.exception("Failed to delete template %d: %s", template_id, exc)
        raise HTTPException(status_code=500, detail=f"Could not delete template: {exc}") from exc

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found.")

    return {"deleted": True, "template_id": template_id}
