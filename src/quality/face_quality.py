"""Classical deterministic face quality assessment (Milestone P1).

Calculates geometric and photometric indicators (bounding-box dimensions, relative
face size, Laplacian sharpness, brightness, contrast, and basic validity) and
attaches configurable non-blocking warnings rather than silently discarding faces.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config_loader import load_yaml_config
from src.common.types import BoundingBox, QualityAssessment
from src.detection.base_detector import InvalidImageError, validate_bgr_image


class FaceQualityAssessor:
    """Computes classical image/face quality metrics and configurable quality warnings."""

    def __init__(
        self,
        warning_thresholds: dict[str, float] | None = None,
        config_path: str | Path = "configs/models.yaml",
    ) -> None:
        cfg_all = load_yaml_config(config_path)
        default_thresh: dict[str, Any] = cfg_all.get("quality", {}).get("warning_thresholds", {})
        merged = {**default_thresh, **(warning_thresholds or {})}

        self.min_face_width_px = float(merged.get("min_face_width_px", 40.0))
        self.min_face_height_px = float(merged.get("min_face_height_px", 40.0))
        self.min_relative_face_area = float(merged.get("min_relative_face_area", 0.002))
        self.min_sharpness_laplacian_var = float(merged.get("min_sharpness_laplacian_var", 35.0))
        self.min_brightness_mean = float(merged.get("min_brightness_mean", 35.0))
        self.max_brightness_mean = float(merged.get("max_brightness_mean", 225.0))
        self.min_contrast_std = float(merged.get("min_contrast_std", 18.0))

    def assess(
        self,
        source_image_bgr: np.ndarray,
        bbox: BoundingBox,
        aligned_crop_bgr: np.ndarray | None = None,
    ) -> QualityAssessment:
        """Assess face quality using source frame geometry and aligned/cropped ROI photometry."""
        t0 = time.perf_counter()
        warnings: list[str] = []

        try:
            img_w, img_h = validate_bgr_image(source_image_bgr)
        except InvalidImageError:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return QualityAssessment(
                is_valid_image=False,
                bbox_width_px=0.0,
                bbox_height_px=0.0,
                bbox_area_px=0.0,
                relative_face_area=0.0,
                sharpness_laplacian_var=0.0,
                brightness_mean=0.0,
                contrast_std=0.0,
                warnings=["invalid_source_image"],
                execution_time_ms=elapsed,
            )

        # Clip bbox to valid image bounds without copying pixel buffer
        x1 = int(max(0, min(img_w, np.floor(bbox.x))))
        y1 = int(max(0, min(img_h, np.floor(bbox.y))))
        x2 = int(max(0, min(img_w, np.ceil(bbox.x2))))
        y2 = int(max(0, min(img_h, np.ceil(bbox.y2))))

        clipped_w = float(max(0, x2 - x1))
        clipped_h = float(max(0, y2 - y1))
        clipped_area = clipped_w * clipped_h
        img_area = float(img_w * img_h)
        rel_area = clipped_area / img_area if img_area > 0 else 0.0

        # Select photometric evaluation buffer (prefer aligned crop if valid, else source ROI view)
        roi_bgr: np.ndarray | None = None
        if (
            aligned_crop_bgr is not None
            and isinstance(aligned_crop_bgr, np.ndarray)
            and aligned_crop_bgr.ndim == 3
            and aligned_crop_bgr.shape[2] == 3
            and aligned_crop_bgr.dtype == np.uint8
            and aligned_crop_bgr.shape[0] >= 8
            and aligned_crop_bgr.shape[1] >= 8
        ):
            roi_bgr = aligned_crop_bgr
        elif clipped_w >= 4 and clipped_h >= 4:
            roi_bgr = source_image_bgr[y1:y2, x1:x2]

        if roi_bgr is None or roi_bgr.size == 0:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return QualityAssessment(
                is_valid_image=False,
                bbox_width_px=float(bbox.w),
                bbox_height_px=float(bbox.h),
                bbox_area_px=float(bbox.area),
                relative_face_area=rel_area,
                sharpness_laplacian_var=0.0,
                brightness_mean=0.0,
                contrast_std=0.0,
                warnings=["degenerate_or_out_of_bounds_face_roi"],
                execution_time_ms=elapsed,
            )

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_var = float(laplacian.var())
        brightness_mean = float(gray.mean())
        contrast_std = float(gray.std())

        is_valid = bool(
            clipped_w >= 4.0
            and clipped_h >= 4.0
            and np.isfinite(sharpness_var)
            and np.isfinite(brightness_mean)
            and np.isfinite(contrast_std)
        )

        if bbox.w < self.min_face_width_px or bbox.h < self.min_face_height_px:
            warnings.append("face_dimensions_below_minimum")
        if rel_area < self.min_relative_face_area:
            warnings.append("relative_face_area_low")
        if sharpness_var < self.min_sharpness_laplacian_var:
            warnings.append("low_sharpness_possible_blur")
        if brightness_mean < self.min_brightness_mean:
            warnings.append("low_brightness_underexposed")
        elif brightness_mean > self.max_brightness_mean:
            warnings.append("high_brightness_overexposed")
        if contrast_std < self.min_contrast_std:
            warnings.append("low_contrast")

        elapsed = (time.perf_counter() - t0) * 1000.0
        return QualityAssessment(
            is_valid_image=is_valid,
            bbox_width_px=float(bbox.w),
            bbox_height_px=float(bbox.h),
            bbox_area_px=float(bbox.area),
            relative_face_area=rel_area,
            sharpness_laplacian_var=sharpness_var,
            brightness_mean=brightness_mean,
            contrast_std=contrast_std,
            warnings=warnings,
            execution_time_ms=elapsed,
        )
