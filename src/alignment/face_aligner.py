"""Geometric 5-point similarity face alignment using YuNet landmarks (Milestone P1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config_loader import load_yaml_config
from src.common.types import (
    CANONICAL_LANDMARK_NAMES,
    AlignedFace,
    BoundingBox,
    FacialLandmarks,
)
from src.detection.base_detector import validate_bgr_image


class InvalidLandmarksError(ValueError):
    """Raised when facial landmarks are missing, non-finite, or geometrically degenerate."""


def _estimate_similarity_2d_umeyama(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Closed-form 2D proper similarity transform (scale + rotation + translation, no reflection).

    Maps `src_pts` (N, 2) -> `dst_pts` (N, 2) minimizing least-squares Euclidean error
    per Umeyama (1991), returning a 2x3 float64 affine matrix compatible with `cv2.warpAffine`.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    src_var = float(np.mean(np.sum(src_centered**2, axis=1)))
    if src_var < 1e-9:
        raise InvalidLandmarksError("Source landmarks have near-zero spatial variance (degenerate).")

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    u_mat, singular_vals, vt_mat = np.linalg.svd(cov)

    det_sign = np.linalg.det(u_mat) * np.linalg.det(vt_mat)
    diag = np.ones(2, dtype=np.float64)
    if det_sign < 0:
        diag[-1] = -1.0

    rot = u_mat @ np.diag(diag) @ vt_mat
    scale = float(np.sum(singular_vals * diag) / src_var)
    if not np.isfinite(scale) or scale <= 1e-6:
        raise InvalidLandmarksError(f"Degenerate similarity scale={scale} estimated from landmarks.")

    trans = dst_mean - scale * (rot @ src_mean)
    mat_2x3 = np.zeros((2, 3), dtype=np.float64)
    mat_2x3[:, :2] = scale * rot
    mat_2x3[:, 2] = trans
    return mat_2x3


class FaceAligner:
    """Aligns detected faces into canonical recognition crops via 5-point similarity warping."""

    def __init__(
        self,
        output_size_wh: tuple[int, int] | None = None,
        template_5pt: list[list[float]] | np.ndarray | None = None,
        template_reference_size_wh: tuple[int, int] | None = None,
        min_inter_ocular_px: float | None = None,
        border_value: tuple[int, int, int] | None = None,
        config_path: str | Path = "configs/models.yaml",
    ) -> None:
        cfg_all = load_yaml_config(config_path)
        align_cfg: dict[str, Any] = cfg_all.get("alignment", {})

        default_out = align_cfg.get("output_size", [112, 112])
        self.output_size_wh: tuple[int, int] = (
            (int(output_size_wh[0]), int(output_size_wh[1]))
            if output_size_wh is not None
            else (int(default_out[0]), int(default_out[1]))
        )
        if self.output_size_wh[0] < 16 or self.output_size_wh[1] < 16:
            raise ValueError(f"Invalid alignment output_size_wh={self.output_size_wh}")

        default_ref = align_cfg.get("template_reference_size", [112, 112])
        ref_w, ref_h = (
            (int(template_reference_size_wh[0]), int(template_reference_size_wh[1]))
            if template_reference_size_wh is not None
            else (int(default_ref[0]), int(default_ref[1]))
        )

        raw_template = (
            template_5pt
            if template_5pt is not None
            else align_cfg.get(
                "canonical_5pt_112",
                [
                    [38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041],
                ],
            )
        )
        base_template = np.asarray(raw_template, dtype=np.float64)
        if base_template.shape != (5, 2) or not np.all(np.isfinite(base_template)):
            raise ValueError(f"Alignment template must have finite shape (5, 2); got {base_template.shape}")

        # Scale canonical template proportionally when output_size differs from reference_size
        scale_vec = np.array(
            [self.output_size_wh[0] / float(ref_w), self.output_size_wh[1] / float(ref_h)],
            dtype=np.float64,
        )
        self._dst_template = base_template * scale_vec

        self.min_inter_ocular_px = float(
            min_inter_ocular_px
            if min_inter_ocular_px is not None
            else align_cfg.get("min_inter_ocular_px", 4.0)
        )
        default_border = align_cfg.get("border_value", [0, 0, 0])
        self.border_value: tuple[int, int, int] = (
            border_value
            if border_value is not None
            else (int(default_border[0]), int(default_border[1]), int(default_border[2]))
        )

    @property
    def template_points(self) -> np.ndarray:
        return self._dst_template.copy()

    def validate_landmarks(self, landmarks: FacialLandmarks | np.ndarray) -> np.ndarray:
        """Validate 5-point landmarks and return `(5, 2)` float64 coordinates."""
        if isinstance(landmarks, FacialLandmarks):
            if tuple(landmarks.order) != CANONICAL_LANDMARK_NAMES:
                raise InvalidLandmarksError(
                    f"Landmark order mismatch: expected {CANONICAL_LANDMARK_NAMES}, got {landmarks.order}"
                )
            pts = landmarks.as_array().astype(np.float64, copy=False)
        elif isinstance(landmarks, np.ndarray):
            pts = landmarks.astype(np.float64, copy=False)
        else:
            raise InvalidLandmarksError("Landmarks must be a FacialLandmarks instance or numpy.ndarray.")

        if pts.shape != (5, 2):
            raise InvalidLandmarksError(f"Expected 5 landmarks of shape (5, 2); got {pts.shape}.")
        if not np.all(np.isfinite(pts)):
            raise InvalidLandmarksError("Landmarks contain NaN or infinite coordinates.")

        inter_ocular = float(np.linalg.norm(pts[1] - pts[0]))
        if inter_ocular < self.min_inter_ocular_px:
            raise InvalidLandmarksError(
                f"Inter-ocular distance ({inter_ocular:.2f}px) is below minimum threshold "
                f"({self.min_inter_ocular_px:.2f}px)."
            )

        # Ensure points are not collinear (rank of centered coordinates must be 2)
        centered = pts - pts.mean(axis=0)
        if np.linalg.matrix_rank(centered, tol=1e-3) < 2:
            raise InvalidLandmarksError("Landmarks are collinear and cannot define a 2D similarity warp.")

        return pts

    def align(
        self,
        image_bgr: np.ndarray,
        landmarks: FacialLandmarks | np.ndarray,
        source_bbox: BoundingBox | None = None,
    ) -> AlignedFace:
        """Warp `image_bgr` directly into an aligned BGR face crop without copying `image_bgr`."""
        img_w, img_h = validate_bgr_image(image_bgr)
        pts = self.validate_landmarks(landmarks)

        t0 = time.perf_counter()
        mat_2x3 = _estimate_similarity_2d_umeyama(pts, self._dst_template)
        out_w, out_h = self.output_size_wh

        # Warp directly from input ndarray buffer (no source frame copy)
        aligned = cv2.warpAffine(
            image_bgr,
            mat_2x3,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )

        # Back-project output canvas corners into original image space to preserve crop bounds
        inv_mat = cv2.invertAffineTransform(mat_2x3)
        canvas_corners = np.array(
            [[0.0, 0.0, 1.0], [out_w, 0.0, 1.0], [out_w, out_h, 1.0], [0.0, out_h, 1.0]],
            dtype=np.float64,
        )
        img_corners = (inv_mat @ canvas_corners.T).T  # (4, 2)
        x1 = int(max(0, np.floor(img_corners[:, 0].min())))
        y1 = int(max(0, np.floor(img_corners[:, 1].min())))
        x2 = int(min(img_w, np.ceil(img_corners[:, 0].max())))
        y2 = int(min(img_h, np.ceil(img_corners[:, 1].max())))

        inter_ocular = float(np.linalg.norm(pts[1] - pts[0]))
        if source_bbox is None:
            min_xy = pts.min(axis=0)
            max_xy = pts.max(axis=0)
            source_bbox = BoundingBox(
                x=float(min_xy[0]),
                y=float(min_xy[1]),
                w=float(max_xy[0] - min_xy[0]),
                h=float(max_xy[1] - min_xy[1]),
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return AlignedFace(
            aligned_bgr=aligned,
            output_size_wh=(out_w, out_h),
            affine_matrix_2x3=mat_2x3,
            source_bbox=source_bbox,
            crop_bounds_xyxy=(x1, y1, x2, y2),
            inter_ocular_px=inter_ocular,
            execution_time_ms=elapsed_ms,
        )
