"""
Knowledge Base module for AI Garment Multi-Label Size Inspection System.
Stores label templates in SQLite (via SQLAlchemy) and embedding vectors in FAISS.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, LargeBinary,
    String, create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM Base and Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class LabelTemplate(Base):
    __tablename__ = "label_templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand_code = Column(String(64), nullable=False, default="")
    po_code = Column(String(64), nullable=False, default="")
    label_type = Column(String(64), nullable=False)
    reference_image_path = Column(String(512), nullable=False, default="")
    feature_embedding = Column(LargeBinary, nullable=False)
    size_field_hint = Column(String(256), nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    usage_count = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime, nullable=True)

    size_rules = relationship("SizeExtractionRule", back_populates="template",
                              cascade="all, delete-orphan")


class SizeExtractionRule(Base):
    __tablename__ = "size_extraction_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(Integer, ForeignKey("label_templates.id"), nullable=False)
    roi_x_norm = Column(Float, nullable=False, default=0.0)
    roi_y_norm = Column(Float, nullable=False, default=0.0)
    roi_w_norm = Column(Float, nullable=False, default=1.0)
    roi_h_norm = Column(Float, nullable=False, default=1.0)
    ocr_hint = Column(String(256), nullable=False, default="")
    regex_pattern = Column(String(512), nullable=True)

    template = relationship("LabelTemplate", back_populates="size_rules")


class InspectionLog(Base):
    __tablename__ = "inspection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(Integer, ForeignKey("label_templates.id"), nullable=True)
    result = Column(String(8), nullable=False)        # PASS | NG
    error_code = Column(String(64), nullable=False, default="")
    detected_sizes = Column(String(2048), nullable=False, default="{}")  # JSON
    error_detail = Column(String(1024), nullable=False, default="")
    snapshot_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# TemplateMatch result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TemplateMatch:
    template_id: int
    similarity: float
    roi_normalized: tuple  # (x, y, w, h)
    ocr_hint: str
    regex_pattern: str


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Persistent knowledge base backed by SQLite + FAISS.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (created if not exists).
    faiss_index_path : str
        Path to persist the FAISS index binary (created if not exists).
    embedding_dim : int
        Dimension of the feature embedding vectors (default 512).
    """

    def __init__(self, db_path: str, faiss_index_path: str, embedding_dim: int = 512) -> None:
        self.db_path = str(db_path)
        self.faiss_index_path = str(faiss_index_path)
        self.embedding_dim = embedding_dim

        # Derive reference images directory from DB location
        self.ref_images_dir = Path(db_path).parent / "reference_images"
        self.ref_images_dir.mkdir(parents=True, exist_ok=True)

        # SQLAlchemy setup
        engine = create_engine(f"sqlite:///{self.db_path}", echo=False, future=True)
        Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine, expire_on_commit=False)
        logger.info("SQLite database ready at %s", self.db_path)

        # FAISS index — inner-product (cosine after L2-normalising)
        self._faiss_index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        # Maps FAISS row index → SQLite template id
        self._index_to_id: list[int] = []

        if Path(self.faiss_index_path).exists():
            self._load_faiss_index()
        else:
            logger.info("No FAISS index found at %s — starting empty.", self.faiss_index_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_template(
        self,
        embedding: np.ndarray,
        label_type: str,
        threshold: float = 0.85,
    ) -> Optional[TemplateMatch]:
        """
        Search the FAISS index for the closest matching template.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding, shape (D,) or (1, D).
        label_type : str
            Label type filter applied post-search.
        threshold : float
            Minimum cosine similarity (0-1) to accept a match.

        Returns
        -------
        TemplateMatch or None
        """
        if self._faiss_index.ntotal == 0:
            logger.debug("FAISS index is empty — no templates to match against.")
            return None

        query = self._normalize_embedding(embedding).reshape(1, -1)
        k = min(10, self._faiss_index.ntotal)  # retrieve top-k candidates

        similarities, indices = self._faiss_index.search(query, k)

        with self._Session() as session:
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < 0:
                    continue  # FAISS padding
                if float(sim) < threshold:
                    break  # results are sorted descending

                template_id = self._index_to_id[idx]
                tpl: Optional[LabelTemplate] = session.get(LabelTemplate, template_id)
                if tpl is None:
                    continue
                if tpl.label_type != label_type:
                    continue

                # Fetch the first (primary) size rule
                rule: Optional[SizeExtractionRule] = None
                if tpl.size_rules:
                    rule = tpl.size_rules[0]

                roi = (
                    rule.roi_x_norm if rule else 0.0,
                    rule.roi_y_norm if rule else 0.0,
                    rule.roi_w_norm if rule else 1.0,
                    rule.roi_h_norm if rule else 1.0,
                )
                ocr_hint = rule.ocr_hint if rule else ""
                regex_pattern = rule.regex_pattern if rule else ""

                # Update usage stats
                tpl.usage_count = (tpl.usage_count or 0) + 1
                tpl.last_used_at = datetime.utcnow()
                session.commit()

                logger.debug(
                    "Template match: id=%d type=%s similarity=%.4f",
                    template_id, label_type, sim,
                )
                return TemplateMatch(
                    template_id=template_id,
                    similarity=float(sim),
                    roi_normalized=roi,
                    ocr_hint=ocr_hint,
                    regex_pattern=regex_pattern or "",
                )

        logger.debug("No template matched for label_type=%s above threshold=%.2f", label_type, threshold)
        return None

    def add_template(
        self,
        label_type: str,
        reference_image: np.ndarray,
        embedding: np.ndarray,
        roi_normalized: tuple,
        brand_code: str = "",
        po_code: str = "",
        ocr_hint: str = "",
        regex_pattern: str = "",
    ) -> int:
        """
        Add a new label template to the database and FAISS index.

        Parameters
        ----------
        label_type : str
        reference_image : np.ndarray
            BGR image (H, W, 3).
        embedding : np.ndarray
            Feature vector, shape (D,).
        roi_normalized : tuple
            (x, y, w, h) all in [0, 1] relative to label image.
        brand_code, po_code : str
        ocr_hint : str
            Free-text hint for the OCR engine (e.g. "SIZE: S/M/L").
        regex_pattern : str
            Optional regex to extract size from raw OCR text.

        Returns
        -------
        int
            The new template's primary key.
        """
        import cv2  # local import to keep startup fast

        norm_emb = self._normalize_embedding(embedding)
        emb_bytes = norm_emb.astype(np.float32).tobytes()

        with self._Session() as session:
            tpl = LabelTemplate(
                brand_code=brand_code,
                po_code=po_code,
                label_type=label_type,
                reference_image_path="",  # filled in below
                feature_embedding=emb_bytes,
                size_field_hint=ocr_hint,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                usage_count=0,
            )
            session.add(tpl)
            session.flush()  # get auto-generated id

            # Persist reference image
            img_filename = f"template_{tpl.id}_{label_type}.png"
            img_path = self.ref_images_dir / img_filename
            cv2.imwrite(str(img_path), reference_image)
            tpl.reference_image_path = str(img_path)

            x, y, w, h = roi_normalized if len(roi_normalized) == 4 else (0.0, 0.0, 1.0, 1.0)
            rule = SizeExtractionRule(
                template_id=tpl.id,
                roi_x_norm=float(x),
                roi_y_norm=float(y),
                roi_w_norm=float(w),
                roi_h_norm=float(h),
                ocr_hint=ocr_hint,
                regex_pattern=regex_pattern or None,
            )
            session.add(rule)
            session.commit()
            template_id = tpl.id

        # Add to FAISS index
        vec = norm_emb.astype(np.float32).reshape(1, -1)
        self._faiss_index.add(vec)
        self._index_to_id.append(template_id)
        self._save_faiss_index()

        logger.info("Added template id=%d type=%s brand=%s", template_id, label_type, brand_code)
        return template_id

    def log_inspection(
        self,
        result: str,
        error_code: str,
        detected_sizes: dict,
        template_id: Optional[int] = None,
        snapshot_path: Optional[str] = None,
        error_detail: str = "",
    ) -> None:
        """Insert an inspection result into the inspection_logs table."""
        with self._Session() as session:
            log = InspectionLog(
                template_id=template_id,
                result=result.upper(),
                error_code=error_code,
                detected_sizes=json.dumps(detected_sizes, ensure_ascii=False),
                error_detail=error_detail,
                snapshot_path=snapshot_path,
                created_at=datetime.utcnow(),
            )
            session.add(log)
            session.commit()
        logger.debug("Inspection logged: result=%s error_code=%s", result, error_code)

    def get_stats(self) -> dict:
        """
        Return aggregate statistics.

        Returns
        -------
        dict with keys:
            total_templates, total_inspections, pass_count, ng_count,
            pass_rate (float 0-1), templates_by_type (dict[str, int])
        """
        with self._Session() as session:
            total_templates: int = session.query(func.count(LabelTemplate.id)).scalar() or 0
            total_inspections: int = session.query(func.count(InspectionLog.id)).scalar() or 0
            pass_count: int = (
                session.query(func.count(InspectionLog.id))
                .filter(InspectionLog.result == "PASS")
                .scalar()
                or 0
            )
            ng_count: int = total_inspections - pass_count
            pass_rate: float = (pass_count / total_inspections) if total_inspections > 0 else 0.0

            # Templates grouped by type
            rows = (
                session.query(LabelTemplate.label_type, func.count(LabelTemplate.id))
                .group_by(LabelTemplate.label_type)
                .all()
            )
            templates_by_type: dict[str, int] = {r[0]: r[1] for r in rows}

        return {
            "total_templates": total_templates,
            "total_inspections": total_inspections,
            "pass_count": pass_count,
            "ng_count": ng_count,
            "pass_rate": pass_rate,
            "templates_by_type": templates_by_type,
        }

    def rebuild_faiss_index(self) -> None:
        """
        Reload all embeddings from SQLite and rebuild the FAISS index from scratch.
        Useful after bulk imports or database migrations.
        """
        logger.info("Rebuilding FAISS index from SQLite…")
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        new_id_map: list[int] = []

        with self._Session() as session:
            templates = session.query(LabelTemplate).order_by(LabelTemplate.id).all()
            for tpl in templates:
                if not tpl.feature_embedding:
                    logger.warning("Template id=%d has no embedding — skipping.", tpl.id)
                    continue
                vec = np.frombuffer(tpl.feature_embedding, dtype=np.float32).copy()
                if vec.shape[0] != self.embedding_dim:
                    logger.warning(
                        "Template id=%d embedding dim mismatch (%d vs %d) — skipping.",
                        tpl.id, vec.shape[0], self.embedding_dim,
                    )
                    continue
                new_index.add(vec.reshape(1, -1))
                new_id_map.append(tpl.id)

        self._faiss_index = new_index
        self._index_to_id = new_id_map
        self._save_faiss_index()
        logger.info("FAISS index rebuilt: %d vectors.", new_index.ntotal)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_faiss_index(self) -> None:
        """Persist FAISS index and id mapping to disk."""
        try:
            Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._faiss_index, self.faiss_index_path)
            # Save the id map as a companion .npy file
            id_map_path = self.faiss_index_path + ".ids.npy"
            np.save(id_map_path, np.array(self._index_to_id, dtype=np.int64))
            logger.debug("FAISS index saved (%d vectors).", self._faiss_index.ntotal)
        except Exception as exc:
            logger.error("Failed to save FAISS index: %s", exc, exc_info=True)

    def _load_faiss_index(self) -> None:
        """Load a previously saved FAISS index and id mapping from disk."""
        try:
            self._faiss_index = faiss.read_index(self.faiss_index_path)
            id_map_path = self.faiss_index_path + ".ids.npy"
            if Path(id_map_path).exists():
                self._index_to_id = np.load(id_map_path).tolist()
            else:
                # Fallback: assume sequential ids starting at 1
                self._index_to_id = list(range(1, self._faiss_index.ntotal + 1))
                logger.warning("ID map file not found — using sequential fallback.")
            logger.info(
                "FAISS index loaded from %s (%d vectors).",
                self.faiss_index_path, self._faiss_index.ntotal,
            )
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s — starting fresh.", exc, exc_info=True)
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self._index_to_id = []

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize a 1-D embedding vector for cosine similarity via inner product."""
        vec = np.array(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm
