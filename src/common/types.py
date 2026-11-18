"""Canonical typed data structures for the Face Biometric Research Workbench (Milestone P1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

CANONICAL_LANDMARK_NAMES: tuple[str, ...] = (
    "right_eye_subject",   # left eye in viewer/image coordinates
    "left_eye_subject",    # right eye in viewer/image coordinates
    "nose_tip",
    "right_mouth_subject", # left mouth corner in viewer/image coordinates
    "left_mouth_subject",  # right mouth corner in viewer/image coordinates
)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in original image pixel coordinates."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def as_xywh(self) -> list[float]:
        return [float(self.x), float(self.y), float(self.w), float(self.h)]

    def as_xyxy(self) -> list[float]:
        return [float(self.x), float(self.y), float(self.x2), float(self.y2)]

    def to_dict(self) -> dict[str, float]:
        return {
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "w": round(float(self.w), 3),
            "h": round(float(self.h), 3),
            "x2": round(float(self.x2), 3),
            "y2": round(float(self.y2), 3),
        }


@dataclass(frozen=True)
class FacialLandmarks:
    """Ordered 5-point facial landmarks in original image pixel coordinates."""

    points: tuple[tuple[float, float], ...]
    order: tuple[str, ...] = CANONICAL_LANDMARK_NAMES

    def as_array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "points": [[round(float(px), 3), round(float(py), 3)] for px, py in self.points],
            "named": {
                name: [round(float(px), 3), round(float(py), 3)]
                for name, (px, py) in zip(self.order, self.points)
            },
        }


@dataclass(frozen=True)
class DetectedFace:
    """Single face detection output from a FaceDetector."""

    bbox: BoundingBox
    landmarks: FacialLandmarks
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounding_box": self.bbox.to_dict(),
            "landmarks": self.landmarks.to_dict(),
            "confidence": round(float(self.confidence), 6),
        }


@dataclass(frozen=True)
class DetectionBatchResult:
    """Structured result for single-image face detection."""

    detections: list[DetectedFace]
    execution_time_ms: float
    original_size_wh: tuple[int, int]
    detector_input_size_wh: tuple[int, int]
    scale_xy: tuple[float, float]
    model_id: str
    model_version: str


@dataclass(frozen=True)
class AlignedFace:
    """Output of 5-landmark similarity face alignment."""

    aligned_bgr: np.ndarray
    output_size_wh: tuple[int, int]
    affine_matrix_2x3: np.ndarray
    source_bbox: BoundingBox
    crop_bounds_xyxy: tuple[int, int, int, int]
    inter_ocular_px: float
    execution_time_ms: float


@dataclass(frozen=True)
class QualityAssessment:
    """Deterministic classical face quality indicators and configurable warnings."""

    is_valid_image: bool
    bbox_width_px: float
    bbox_height_px: float
    bbox_area_px: float
    relative_face_area: float
    sharpness_laplacian_var: float
    brightness_mean: float
    contrast_std: float
    warnings: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid_image": self.is_valid_image,
            "bbox_width_px": round(float(self.bbox_width_px), 3),
            "bbox_height_px": round(float(self.bbox_height_px), 3),
            "bbox_area_px": round(float(self.bbox_area_px), 3),
            "relative_face_area": round(float(self.relative_face_area), 6),
            "sharpness_laplacian_var": round(float(self.sharpness_laplacian_var), 4),
            "brightness_mean": round(float(self.brightness_mean), 4),
            "contrast_std": round(float(self.contrast_std), 4),
            "warnings": list(self.warnings),
            "execution_time_ms": round(float(self.execution_time_ms), 4),
        }


@dataclass
class FaceInstanceResult:
    """Independent per-detection face-instance record (never resolved to an identity)."""

    image_id: str
    face_instance_id: str
    bounding_box: dict[str, float]
    landmarks: dict[str, Any]
    detection_confidence: float
    alignment: dict[str, Any]
    quality: dict[str, Any]
    raw_embedding: list[float] | None
    embedding: list[float] | None
    embedding_dim: int | None
    l2_normalized: bool
    embedding_status: str
    models: dict[str, Any]
    stage_latency_ms: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImagePipelineResult:
    """Top-level pipeline output for a single input image."""

    image_id: str
    image_path: str | None
    image_dimensions: dict[str, int]
    num_faces: int
    faces: list[FaceInstanceResult]
    models: dict[str, Any]
    stage_latency_ms: dict[str, float | None]
    cold_start: bool

    def __iter__(self):
        return iter(self.faces)

    def __len__(self) -> int:
        return len(self.faces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "image_dimensions": dict(self.image_dimensions),
            "num_faces": self.num_faces,
            "cold_start": self.cold_start,
            "models": dict(self.models),
            "stage_latency_ms": dict(self.stage_latency_ms),
            "faces": [f.to_dict() for f in self.faces],
        }
