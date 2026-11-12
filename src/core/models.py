"""
src/core/models.py
Shared dataclasses and enums for the AI Garment Multi-Label Size Inspection System.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum


class LabelType(str, Enum):
    """Enumeration of recognised garment label categories."""
    SIZE_LABEL = "size_label"
    CARE_LABEL = "care_label"
    HANG_TAG = "hang_tag"
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    """Inspection outcome / failure reason codes."""
    PASS = "PASS"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    MISSING_REQUIRED_LABEL = "MISSING_REQUIRED_LABEL"
    SIZE_NOT_FOUND = "SIZE_NOT_FOUND"
    TEMPLATE_NOT_FOUND = "TEMPLATE_NOT_FOUND"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 0.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self):
        """Return (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class LabelDetection:
    """Single detected label region with associated metadata."""
    label_type: LabelType
    bbox: BBox
    confidence: float
    cropped_image: Optional[object] = None   # numpy ndarray
    template_id: Optional[int] = None
    size_value: Optional[str] = None
    ocr_confidence: Optional[float] = None
    needs_teaching: bool = False

    def has_readable_size(self) -> bool:
        """Return True when OCR produced a non-empty size string."""
        return bool(self.size_value and self.size_value.strip())


@dataclass
class JudgeResult:
    """Output of the ConsistencyJudge for one garment frame."""
    is_pass: bool
    error_code: ErrorCode
    error_detail: str
    sizes_dict: Dict[str, str] = field(default_factory=dict)
    missing_labels: List[str] = field(default_factory=list)


@dataclass
class InspectionResult:
    """Top-level result returned by InspectionPipeline.run_sync()."""
    is_pass: bool
    error_code: ErrorCode
    error_detail: str
    sizes_found: Dict[str, str] = field(default_factory=dict)
    detections: List[LabelDetection] = field(default_factory=list)
    latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    snapshot_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    needs_teaching: bool = False
    unknown_label_crops: List[object] = field(default_factory=list)  # numpy arrays

    def summary(self) -> str:
        status = "PASS ✓" if self.is_pass else f"FAIL ✗ [{self.error_code}]"
        return (
            f"{status} | sizes={self.sizes_found} | "
            f"labels={len(self.detections)} | {self.latency_ms:.1f} ms"
        )
