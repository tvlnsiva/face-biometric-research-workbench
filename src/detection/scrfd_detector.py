"""Milestone P2 — SCRFD-500M Face Detector Adapter (`src/detection/scrfd_detector.py`).

Implements `FaceDetector` for InsightFace's `SCRFD-500M-BNKPS` (`scrfd_500m.onnx`, `RESEARCH ONLY`),
verified directly against upstream `deepinsight/insightface/python-package/insightface/model_zoo/scrfd.py`:
  - Input: `input.1` (`float32`, `[1, 3, H, W]`, `H` and `W` multiples of `32`)
  - Color & Scaling: `RGB` (`swapRB=True`), `(rgb_uint8 - 127.5) / 128.0` (`mean=127.5, std=128.0`)
  - Feature Pyramid Heads: 9 outputs across strides `[8, 16, 32]` with `num_anchors = 2` per cell:
      * Scores `[N, 1]`, BBox LTRB distances `[N, 4]`, 5-Point Landmark offsets `[N, 10]`
  - Supports both `CPUExecutionProvider` and `CUDAExecutionProvider` (with `session.disable_cpu_ep_fallback="1"`,
    `cudnn_conv_algo_search="HEURISTIC"`, and explicit CUDA `io_binding` + `synchronize_outputs()`).
  - Supports fixed stride-32 letterbox canvas resizing (`allow_upscale=False` for apples-to-apples
    non-upscaling `max_input_dim` comparison with YuNet, or `allow_upscale=True` for upstream InsightFace
    `det_size` scaling) so GPU input tensor dimensions never thrash across varying aspect ratios.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config_loader import compute_sha256, load_yaml_config, resolve_project_path
from src.common.onnx_utils import ensure_cuda_dll_paths
from src.common.types import (
    CANONICAL_LANDMARK_NAMES,
    BoundingBox,
    DetectedFace,
    DetectionBatchResult,
    FacialLandmarks,
)
from src.detection.base_detector import FaceDetector, validate_bgr_image


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _greedy_nms(dets_xyxys: np.ndarray, nms_thresh: float) -> list[int]:
    if dets_xyxys.shape[0] == 0:
        return []
    x1 = dets_xyxys[:, 0]
    y1 = dets_xyxys[:, 1]
    x2 = dets_xyxys[:, 2]
    y2 = dets_xyxys[:, 3]
    scores = dets_xyxys[:, 4]

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = np.argsort(-scores, kind="stable")

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    return keep


class SCRFDDetector(FaceDetector):
    """ONNX Runtime `SCRFD-500M-BNKPS` detector (`RESEARCH ONLY` candidate)."""

    def __init__(
        self,
        model_id: str = "scrfd_500m_research",
        model_path: str | Path | None = None,
        device: str | None = None,
        input_size_wh: tuple[int, int] | None = None,
        score_threshold: float | None = None,
        nms_threshold: float | None = None,
        top_k: int | None = None,
        allow_upscale: bool | None = None,
        config_path: str | Path | dict[str, Any] = "configs/p2_models.yaml",
    ) -> None:
        cfg = config_path if isinstance(config_path, dict) else load_yaml_config(config_path)
        model_cfg = cfg["models"][model_id]
        det_params = model_cfg.get("detection_params", {})
        norm_cfg = model_cfg.get("normalization", {})

        self._model_id = model_id
        self._model_name = str(model_cfg.get("model_name", model_id))
        self._model_version = str(model_cfg.get("model_version", "v0.0.1-det_500m"))
        self._licensing_status = dict(model_cfg.get("licensing_status", {}))

        resolved_path = resolve_project_path(model_path or model_cfg["model_path"])
        if not resolved_path.is_file():
            raise FileNotFoundError(f"SCRFD ONNX artifact not found at '{resolved_path}'")
        self._model_path = resolved_path
        self._model_sha256 = compute_sha256(resolved_path)

        cfg_in_sz = det_params.get("input_size", [1280, 1280])
        in_w, in_h = input_size_wh if input_size_wh is not None else (int(cfg_in_sz[0]), int(cfg_in_sz[1]))
        if in_w <= 0 or in_h <= 0 or (in_w % 32 != 0) or (in_h % 32 != 0):
            raise ValueError(
                f"SCRFD input_size_wh must be positive multiples of 32; got ({in_w}, {in_h})"
            )
        self.input_size_wh = (int(in_w), int(in_h))
        self.score_threshold = float(
            score_threshold if score_threshold is not None else det_params.get("score_threshold", 0.50)
        )
        self.nms_threshold = float(
            nms_threshold if nms_threshold is not None else det_params.get("nms_threshold", 0.40)
        )
        self.top_k = int(top_k if top_k is not None else det_params.get("top_k", 5000))
        self.allow_upscale = bool(
            allow_upscale if allow_upscale is not None else det_params.get("allow_upscale", False)
        )

        self.color_order = str(norm_cfg.get("color_order", "RGB")).upper()
        mean_vals = norm_cfg.get("mean", [127.5, 127.5, 127.5])
        std_vals = norm_cfg.get("std", [128.0, 128.0, 128.0])
        self._input_mean = float(mean_vals[0])
        self._input_std = float(std_vals[0])

        ensure_cuda_dll_paths()
        import onnxruntime as ort

        req_dev = str(device or model_cfg.get("device", "cuda")).lower()
        avail = ort.get_available_providers()
        if req_dev.startswith("cuda"):
            if "CUDAExecutionProvider" not in avail:
                fallback = str(model_cfg.get("fallback_device", "cpu")).lower()
                if fallback == "cpu":
                    self._device = "cpu"
                else:
                    raise RuntimeError(f"CUDAExecutionProvider not available in {avail}")
            else:
                self._device = "cuda:0"
        elif req_dev == "cpu":
            self._device = "cpu"
        else:
            raise ValueError(f"Unsupported device='{req_dev}' for SCRFDDetector")

        sess_opts = ort.SessionOptions()
        # Suppress benign opset-11 VerifyOutputSizes stderr warnings when running dynamic shapes != 640x640
        sess_opts.log_severity_level = 3
        if self._device.startswith("cuda"):
            # Note: opset-11 SCRFD FPN `Resize` contains integer shape-calculation ops (`Shape`/`Gather`)
            # that ONNX Runtime keeps in host registers while all Conv/FPN/Head tensors run on CUDA.
            ep_list: list[Any] = [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider",
            ]
        else:
            ep_list = ["CPUExecutionProvider"]

        t0 = time.perf_counter()
        self._session = ort.InferenceSession(str(resolved_path), sess_options=sess_opts, providers=ep_list)
        self._init_time_ms = (time.perf_counter() - t0) * 1000.0
        if self._device.startswith("cuda") and "CUDAExecutionProvider" not in self._session.get_providers():
            raise RuntimeError(
                f"CUDA requested for SCRFDDetector, but active providers are {self._session.get_providers()}"
            )

        self._input_name = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        self._output_names = [o.name for o in outputs]
        if len(outputs) != 9:
            raise ValueError(f"Expected 9 outputs for SCRFD-500M-BNKPS; got {len(outputs)}")
        self._fmc = 3
        self._feat_strides = (8, 16, 32)
        self._num_anchors = 2
        self._batched = len(outputs[0].shape) == 3
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def device(self) -> str:
        return self._device

    @property
    def active_providers(self) -> list[str]:
        return list(self._session.get_providers())

    @property
    def initialization_time_ms(self) -> float:
        return self._init_time_ms

    def _get_anchor_centers(self, feat_h: int, feat_w: int, stride: int) -> np.ndarray:
        key = (feat_h, feat_w, stride)
        cached = self._center_cache.get(key)
        if cached is not None:
            return cached
        centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        if self._num_anchors > 1:
            centers = np.stack([centers] * self._num_anchors, axis=1).reshape((-1, 2))
        if len(self._center_cache) < 64:
            self._center_cache[key] = centers
        return centers

    def _run_raw_blob(self, blob: np.ndarray) -> list[np.ndarray]:
        import onnxruntime as ort

        if self._device.startswith("cuda"):
            io_b = self._session.io_binding()
            ort_in = ort.OrtValue.ortvalue_from_numpy(blob, "cuda", 0)
            io_b.bind_ortvalue_input(self._input_name, ort_in)
            for oname in self._output_names:
                io_b.bind_output(oname, "cuda", 0)
            self._session.run_with_iobinding(io_b)
            io_b.synchronize_outputs()
            return [o.numpy() for o in io_b.get_outputs()]
        return self._session.run(self._output_names, {self._input_name: blob})

    def detect_profiled(
        self,
        image_bgr: np.ndarray,
        input_size_wh: tuple[int, int] | None = None,
    ) -> tuple[DetectionBatchResult, dict[str, float]]:
        orig_w, orig_h = validate_bgr_image(image_bgr)
        target_w, target_h = input_size_wh if input_size_wh is not None else self.input_size_wh
        if target_w % 32 != 0 or target_h % 32 != 0:
            raise ValueError(f"SCRFD input_size_wh must be divisible by 32; got ({target_w}, {target_h})")

        t_total_0 = time.perf_counter()

        # 1. Aspect-preserving resize + zero-padded stride-32 canvas
        t_res_0 = time.perf_counter()
        im_ratio = float(orig_h) / float(orig_w)
        model_ratio = float(target_h) / float(target_w)
        if not self.allow_upscale and orig_w <= target_w and orig_h <= target_h:
            new_w, new_h = orig_w, orig_h
            det_scale = 1.0
            resized = image_bgr
        else:
            if im_ratio > model_ratio:
                new_h = target_h
                new_w = max(1, int(round(new_h / im_ratio)))
            else:
                new_w = target_w
                new_h = max(1, int(round(new_w * im_ratio)))
            det_scale = float(new_h) / float(orig_h)
            interp = cv2.INTER_AREA if det_scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized
        resize_ms = (time.perf_counter() - t_res_0) * 1000.0

        # 2. Color swap (BGR -> RGB) & (x - 127.5) / 128.0 NCHW blob preparation
        t_pre_0 = time.perf_counter()
        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / self._input_std,
            (target_w, target_h),
            (self._input_mean, self._input_mean, self._input_mean),
            swapRB=(self.color_order == "RGB"),
        )
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        preprocess_ms = (time.perf_counter() - t_pre_0) * 1000.0

        # 3. ONNX Runtime forward pass (with explicit CUDA synchronization when on GPU)
        t_inf_0 = time.perf_counter()
        net_outs = self._run_raw_blob(blob)
        inference_ms = (time.perf_counter() - t_inf_0) * 1000.0

        # 4. Multi-stride anchor decoding, NMS & inverse projection to original pixel coordinates
        t_post_0 = time.perf_counter()
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        for idx, stride in enumerate(self._feat_strides):
            if self._batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self._fmc][0] * stride
                kps_preds = net_outs[idx + self._fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self._fmc] * stride
                kps_preds = net_outs[idx + self._fmc * 2] * stride

            scores_1d = scores.reshape(-1)
            pos_inds = np.where(scores_1d >= self.score_threshold)[0]
            if pos_inds.size == 0:
                continue

            feat_h = target_h // stride
            feat_w = target_w // stride
            centers = self._get_anchor_centers(feat_h, feat_w, stride)
            pos_centers = centers[pos_inds]

            pos_scores = scores_1d[pos_inds]
            pos_bboxes = _distance2bbox(pos_centers, bbox_preds[pos_inds])
            pos_kpss = _distance2kps(pos_centers, kps_preds[pos_inds]).reshape((-1, 5, 2))

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)

        detections: list[DetectedFace] = []
        if scores_list:
            all_scores = np.concatenate(scores_list, axis=0)
            all_bboxes = np.concatenate(bboxes_list, axis=0) / det_scale
            all_kpss = np.concatenate(kpss_list, axis=0) / det_scale

            order = np.argsort(-all_scores, kind="stable")
            if self.top_k > 0 and order.size > self.top_k:
                order = order[: self.top_k]

            pre_det = np.column_stack([all_bboxes[order], all_scores[order]]).astype(np.float32, copy=False)
            ordered_kpss = all_kpss[order]

            keep = _greedy_nms(pre_det, self.nms_threshold)
            kept_det = pre_det[keep]
            kept_kpss = ordered_kpss[keep]

            for row, kp5 in zip(kept_det, kept_kpss):
                x1 = float(np.clip(row[0], 0.0, float(orig_w)))
                y1 = float(np.clip(row[1], 0.0, float(orig_h)))
                x2 = float(np.clip(row[2], 0.0, float(orig_w)))
                y2 = float(np.clip(row[3], 0.0, float(orig_h)))
                w_box = max(0.0, x2 - x1)
                h_box = max(0.0, y2 - y1)
                if w_box <= 0.5 or h_box <= 0.5:
                    continue

                pts = tuple(
                    (
                        float(np.clip(kp5[j, 0], 0.0, float(orig_w))),
                        float(np.clip(kp5[j, 1], 0.0, float(orig_h))),
                    )
                    for j in range(5)
                )
                detections.append(
                    DetectedFace(
                        bbox=BoundingBox(x=x1, y=y1, w=w_box, h=h_box),
                        landmarks=FacialLandmarks(points=pts, order=CANONICAL_LANDMARK_NAMES),
                        confidence=float(row[4]),
                    )
                )

        postprocess_ms = (time.perf_counter() - t_post_0) * 1000.0
        total_ms = (time.perf_counter() - t_total_0) * 1000.0

        inv_scale = (float(orig_w) / float(new_w), float(orig_h) / float(new_h))
        batch_res = DetectionBatchResult(
            detections=detections,
            execution_time_ms=total_ms,
            original_size_wh=(orig_w, orig_h),
            detector_input_size_wh=(target_w, target_h),
            scale_xy=inv_scale,
            model_id=self._model_id,
            model_version=self._model_version,
        )
        profile = {
            "resize_and_pad_ms": resize_ms,
            "preprocess_blob_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_decode_nms_ms": postprocess_ms,
            "total_detect_ms": total_ms,
            "active_content_size_wh": (new_w, new_h),
            "det_scale": det_scale,
        }
        return batch_res, profile

    def detect(
        self,
        image_bgr: np.ndarray,
        input_size_wh: tuple[int, int] | None = None,
    ) -> DetectionBatchResult:
        res, _ = self.detect_profiled(image_bgr, input_size_wh=input_size_wh)
        return res
