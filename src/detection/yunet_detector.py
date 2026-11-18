"""YuNet face detector implementation backed by OpenCV FaceDetectorYN (Milestone P1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config_loader import load_yaml_config, resolve_project_path
from src.common.types import (
    CANONICAL_LANDMARK_NAMES,
    BoundingBox,
    DetectedFace,
    DetectionBatchResult,
    FacialLandmarks,
)
from src.detection.base_detector import FaceDetector, InvalidImageError, validate_bgr_image


class YuNetDetector(FaceDetector):
    """Face detector using the official OpenCV Zoo YuNet ONNX model.

    Loads the ONNX model once at initialization, supports configurable input
    resolutions and resize modes, preserves original image coordinates when
    resizing, and supports multiple faces per image.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float | None = None,
        nms_threshold: float | None = None,
        top_k: int | None = None,
        resize_mode: str | None = None,
        max_input_dim: int | None = None,
        fixed_input_size_wh: tuple[int, int] | None = None,
        model_id: str = "yunet_2023mar",
        config_path: str | Path = "configs/models.yaml",
    ) -> None:
        cfg_all = load_yaml_config(config_path)
        model_cfg: dict[str, Any] = cfg_all["models"][model_id]
        det_params: dict[str, Any] = model_cfg.get("detection_params", {})

        resolved_path = resolve_project_path(model_path or model_cfg["model_path"])
        if not resolved_path.is_file():
            raise FileNotFoundError(f"YuNet ONNX model artifact not found at: {resolved_path}")

        self._model_id = model_id
        self._model_name = str(model_cfg.get("model_name", "YuNet"))
        self._model_version = str(model_cfg.get("model_version", "2023mar"))
        self._model_path = resolved_path

        self.score_threshold = float(
            score_threshold if score_threshold is not None else det_params.get("score_threshold", 0.6)
        )
        self.nms_threshold = float(
            nms_threshold if nms_threshold is not None else det_params.get("nms_threshold", 0.3)
        )
        self.top_k = int(top_k if top_k is not None else det_params.get("top_k", 5000))
        self.resize_mode = str(
            resize_mode if resize_mode is not None else det_params.get("resize_mode", "preserve_aspect_max_dim")
        )
        if self.resize_mode not in {"native", "preserve_aspect_max_dim", "fixed"}:
            raise ValueError(f"Unsupported resize_mode='{self.resize_mode}'")

        self.max_input_dim = int(
            max_input_dim if max_input_dim is not None else det_params.get("max_input_dim", 1280)
        )
        default_fixed = det_params.get("fixed_input_size", [640, 640])
        self.fixed_input_size_wh: tuple[int, int] = (
            (int(fixed_input_size_wh[0]), int(fixed_input_size_wh[1]))
            if fixed_input_size_wh is not None
            else (int(default_fixed[0]), int(default_fixed[1]))
        )

        # Load the ONNX model ONCE
        t0 = time.perf_counter()
        self._detector = cv2.FaceDetectorYN.create(
            str(self._model_path),
            "",
            self.fixed_input_size_wh,
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )
        self._init_time_ms = (time.perf_counter() - t0) * 1000.0
        self._current_input_size_wh = self.fixed_input_size_wh

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def initialization_time_ms(self) -> float:
        return self._init_time_ms

    def set_thresholds(
        self,
        score_threshold: float | None = None,
        nms_threshold: float | None = None,
        top_k: int | None = None,
    ) -> None:
        """Update configurable detection thresholds without reloading the ONNX model."""
        if score_threshold is not None:
            self.score_threshold = float(score_threshold)
            self._detector.setScoreThreshold(self.score_threshold)
        if nms_threshold is not None:
            self.nms_threshold = float(nms_threshold)
            self._detector.setNMSThreshold(self.nms_threshold)
        if top_k is not None:
            self.top_k = int(top_k)
            self._detector.setTopK(self.top_k)

    def _compute_target_size(
        self,
        orig_w: int,
        orig_h: int,
        override_size_wh: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if override_size_wh is not None:
            tw, th = int(override_size_wh[0]), int(override_size_wh[1])
            if tw < 8 or th < 8:
                raise InvalidImageError(f"Invalid override input_size_wh={override_size_wh}")
            return tw, th

        if self.resize_mode == "native":
            return orig_w, orig_h

        if self.resize_mode == "fixed":
            return self.fixed_input_size_wh

        # preserve_aspect_max_dim
        longest = max(orig_w, orig_h)
        if longest <= self.max_input_dim:
            return orig_w, orig_h
        scale = self.max_input_dim / float(longest)
        tw = max(8, int(round(orig_w * scale)))
        th = max(8, int(round(orig_h * scale)))
        return tw, th

    def detect(
        self,
        image_bgr: np.ndarray,
        input_size_wh: tuple[int, int] | None = None,
    ) -> DetectionBatchResult:
        """Run YuNet detection and map bounding boxes & 5 landmarks to original coordinates."""
        result, _ = self.detect_profiled(image_bgr, input_size_wh=input_size_wh)
        return result

    def detect_profiled(
        self,
        image_bgr: np.ndarray,
        input_size_wh: tuple[int, int] | None = None,
    ) -> tuple[DetectionBatchResult, dict[str, float]]:
        """Run YuNet detection and return `(DetectionBatchResult, stage_profile_ms)`.

        Explicitly avoids redundant color conversions (YuNet consumes BGR directly) and avoids
        unnecessary memory copies whenever `det_input` is already C-contiguous.
        """
        orig_w, orig_h = validate_bgr_image(image_bgr)
        det_w, det_h = self._compute_target_size(orig_w, orig_h, input_size_wh)

        t_start = time.perf_counter()

        # 1. Aspect-preserving or configured resize (zero-copy pass-through at native resolution)
        t_res0 = time.perf_counter()
        if (det_w, det_h) != (orig_w, orig_h):
            interp = cv2.INTER_AREA if (det_w < orig_w or det_h < orig_h) else cv2.INTER_LINEAR
            det_input = cv2.resize(image_bgr, (det_w, det_h), interpolation=interp)
            scale_x = float(orig_w) / float(det_w)
            scale_y = float(orig_h) / float(det_h)
        else:
            det_input = image_bgr
            scale_x = 1.0
            scale_y = 1.0
        resize_ms = (time.perf_counter() - t_res0) * 1000.0

        # 2. Color conversion: 0.0 ms — OpenCV imdecode outputs BGR and YuNet expects BGR natively
        color_conversion_ms = 0.0

        # 3. Memory contiguity check: copy ONLY if input slice is non-contiguous
        t_mem0 = time.perf_counter()
        if not det_input.flags["C_CONTIGUOUS"]:
            det_input = np.ascontiguousarray(det_input)
        memory_copy_ms = (time.perf_counter() - t_mem0) * 1000.0

        # 4. Dynamic input size update (only invoked when dimensions actually change)
        t_size0 = time.perf_counter()
        if self._current_input_size_wh != (det_w, det_h):
            self._detector.setInputSize((det_w, det_h))
            self._current_input_size_wh = (det_w, det_h)
        set_input_size_ms = (time.perf_counter() - t_size0) * 1000.0

        # 5. Core YuNet ONNX inference via OpenCV FaceDetectorYN
        t_inf0 = time.perf_counter()
        retval, raw_faces = self._detector.detect(det_input)
        yunet_inference_ms = (time.perf_counter() - t_inf0) * 1000.0

        if retval != 1:
            raise RuntimeError(f"OpenCV FaceDetectorYN.detect failed with retval={retval}")

        # 6. Bounding-box & 5-landmark coordinate transformation back to original image space
        t_proj0 = time.perf_counter()
        detections: list[DetectedFace] = []
        if raw_faces is not None and len(raw_faces) > 0:
            for row in raw_faces:
                bx = float(row[0]) * scale_x
                by = float(row[1]) * scale_y
                bw = float(row[2]) * scale_x
                bh = float(row[3]) * scale_y

                pts = (
                    (float(row[4]) * scale_x, float(row[5]) * scale_y),
                    (float(row[6]) * scale_x, float(row[7]) * scale_y),
                    (float(row[8]) * scale_x, float(row[9]) * scale_y),
                    (float(row[10]) * scale_x, float(row[11]) * scale_y),
                    (float(row[12]) * scale_x, float(row[13]) * scale_y),
                )
                score = float(row[14])

                detections.append(
                    DetectedFace(
                        bbox=BoundingBox(x=bx, y=by, w=bw, h=bh),
                        landmarks=FacialLandmarks(points=pts, order=CANONICAL_LANDMARK_NAMES),
                        confidence=score,
                    )
                )

            # Deterministic ordering: highest confidence first, tie-broken by (x, y)
            detections.sort(key=lambda d: (-d.confidence, d.bbox.x, d.bbox.y))
        coordinate_transform_ms = (time.perf_counter() - t_proj0) * 1000.0
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        profile = {
            "resize_ms": resize_ms,
            "color_conversion_ms": color_conversion_ms,
            "memory_copy_ms": memory_copy_ms,
            "set_input_size_ms": set_input_size_ms,
            "yunet_inference_ms": yunet_inference_ms,
            "coordinate_transform_ms": coordinate_transform_ms,
            "total_detect_ms": elapsed_ms,
        }

        return (
            DetectionBatchResult(
                detections=detections,
                execution_time_ms=elapsed_ms,
                original_size_wh=(orig_w, orig_h),
                detector_input_size_wh=(det_w, det_h),
                scale_xy=(scale_x, scale_y),
                model_id=self._model_id,
                model_version=self._model_version,
            ),
            profile,
        )
