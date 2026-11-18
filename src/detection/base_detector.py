"""Common detector interface for modular face detection evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.common.types import DetectionBatchResult


class InvalidImageError(ValueError):
    """Raised when an input image is None, empty, corrupt, or has unsupported shape/dtype."""


def validate_bgr_image(image: np.ndarray | None) -> tuple[int, int]:
    """Validate an in-memory BGR uint8 image and return `(width, height)`."""
    if image is None or not isinstance(image, np.ndarray):
        raise InvalidImageError("Input image must be a non-None numpy.ndarray.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(
            f"Expected 3-channel BGR image of shape (H, W, 3); got shape={image.shape}."
        )
    height, width = int(image.shape[0]), int(image.shape[1])
    if height < 8 or width < 8:
        raise InvalidImageError(
            f"Image dimensions ({width}x{height}) are too small for face detection (min 8x8)."
        )
    if image.dtype != np.uint8:
        raise InvalidImageError(f"Expected uint8 BGR image; got dtype={image.dtype}.")
    return width, height


class FaceDetector(ABC):
    """Abstract base interface for face detectors in the research workbench."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Canonical model identifier."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model version or release identifier."""

    @property
    @abstractmethod
    def initialization_time_ms(self) -> float:
        """One-time model load latency in milliseconds."""

    @abstractmethod
    def detect(
        self,
        image_bgr: np.ndarray,
        input_size_wh: tuple[int, int] | None = None,
    ) -> DetectionBatchResult:
        """Detect all faces in `image_bgr` and return coordinates in original image space."""
