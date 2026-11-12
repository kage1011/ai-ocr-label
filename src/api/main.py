"""
FastAPI application entry point for the AI Garment Size Inspector.

Start with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes.inspect import router as inspect_router
from src.api.routes.teach import router as teach_router
from src.api.routes.templates import router as templates_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_config(path: str = "config.yaml") -> dict:
    """Load YAML config relative to the project root (CWD)."""
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("config.yaml not found at %s — using defaults.", config_path.resolve())
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Application factory / lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load config, initialise KB, pipeline, and teaching manager.
    Shutdown: flush resources.
    """
    config = _load_config()
    app.state.config = config

    # --- Knowledge base ---
    try:
        from src.core.knowledge_base import KnowledgeBase  # type: ignore
        db_path = config.get("database", {}).get("path", "data/knowledge_base.db")
        faiss_path = config.get("database", {}).get("faiss_index_path", "data/faiss_index.bin")
        ref_dir = config.get("database", {}).get("reference_images_dir", "data/reference_images")
        kb = KnowledgeBase(db_path=db_path, faiss_index_path=faiss_path, reference_images_dir=ref_dir)
        app.state.knowledge_base = kb
        logger.info("KnowledgeBase initialised.")
    except Exception as exc:
        logger.warning("KnowledgeBase not available: %s", exc)
        app.state.knowledge_base = None

    # --- Inspection pipeline ---
    try:
        from src.core.pipeline import InspectionPipeline  # type: ignore
        pipeline = InspectionPipeline(config=config, knowledge_base=app.state.knowledge_base)
        await pipeline.warm_up()
        app.state.pipeline = pipeline
        logger.info("Inspection pipeline warmed up.")
    except Exception as exc:
        logger.warning("Pipeline not available: %s", exc)
        app.state.pipeline = None

    # --- Size extractor (for OCR during teaching) ---
    try:
        from src.core.size_extractor import SizeExtractor  # type: ignore
        ocr_cfg = config.get("models", {}).get("ocr", {})
        size_extractor = SizeExtractor(config=ocr_cfg)
        app.state.size_extractor = size_extractor
    except Exception as exc:
        logger.warning("SizeExtractor not available: %s", exc)
        app.state.size_extractor = None

    # --- Teaching session manager ---
    try:
        from src.teaching.teaching_session import TeachingSessionManager  # type: ignore

        feature_extractor_fn = None
        if app.state.pipeline is not None:
            feature_extractor_fn = app.state.pipeline.extract_features

        temp_dir = "data/teaching_temp"
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        teaching_manager = TeachingSessionManager(
            knowledge_base=app.state.knowledge_base,
            feature_extractor_fn=feature_extractor_fn or (lambda img: None),
            size_extractor=app.state.size_extractor,
            temp_dir=temp_dir,
        )
        app.state.teaching_manager = teaching_manager
        logger.info("TeachingSessionManager initialised.")
    except Exception as exc:
        logger.warning("TeachingSessionManager not available: %s", exc)
        app.state.teaching_manager = None

    logger.info("AI Garment Size Inspector API is ready.")
    yield

    # --- Shutdown ---
    logger.info("Shutting down AI Garment Size Inspector API.")
    if getattr(app.state, "pipeline", None) is not None:
        try:
            await app.state.pipeline.shutdown()
        except Exception as exc:
            logger.warning("Error during pipeline shutdown: %s", exc)


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Garment Size Inspector",
    version="1.0.0",
    description=(
        "REST API for multi-label garment size inspection. "
        "Supports real-time inspection, supervisor teaching mode, "
        "and template management."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow web UI (adjust origins for production)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(inspect_router)
app.include_router(teach_router)
app.include_router(templates_router)

# ---------------------------------------------------------------------------
# Static file mounts
# ---------------------------------------------------------------------------

# Annotated snapshot images served at /snapshots/<filename>
_snapshots_dir = Path("data/snapshots")
_snapshots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(_snapshots_dir)), name="snapshots")

# Temporary teaching crop images served at /teaching_temp/<filename>
_teaching_temp_dir = Path("data/teaching_temp")
_teaching_temp_dir.mkdir(parents=True, exist_ok=True)
app.mount("/teaching_temp", StaticFiles(directory=str(_teaching_temp_dir)), name="teaching_temp")

# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
async def root() -> dict:
    """Return basic API information and links to documentation."""
    return {
        "name": "AI Garment Size Inspector",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "inspect": "/api/inspect/",
            "health": "/api/inspect/health",
            "teach_sessions": "/api/teach/sessions",
            "templates": "/api/templates/",
            "stats": "/api/templates/stats",
        },
    }
