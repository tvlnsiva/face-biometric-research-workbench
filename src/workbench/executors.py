"""Authoritative Backend Execution Adapters (`src/workbench/executors.py`).

Thin, non-duplicative adapters around the existing P1, P1.1, P2, and P3 Python modules:
  - Detection (`YuNetDetector`, `ResearchSCRFDWrapper`, `gaussian_soft_nms`)
  - Alignment (`FaceAligner`, `_estimate_similarity_2d_umeyama`, `estimate_affine_least_squares_6dof`)
  - Quality (`FaceQualityAssessor`, `compute_extended_quality_features`, degradations)
  - Recognition & Matching (`ReferenceResNet50Embedder`, `MobileFaceNetEmbedder`, `SFaceEmbedder`,
    `GuardedFixedBatchEmbedder`, `extract_with_flip_tta`, `compute_pair_scores`, `evaluate_calibrated_verification`)
  - 1:N Retrieval (`ExactFlatIndex`, `IVFFlatIndex`, `IVFPQIndex`, `HNSWIndex`, `compute_recall_at_k`)
  - Multi-stage Pipeline Execution & Dataset Benchmarks
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from benchmarks.detection.benchmark_yunet import compute_iou
from src.alignment.face_aligner import FaceAligner, _estimate_similarity_2d_umeyama
from src.common.config_loader import resolve_project_path
from src.common.types import CANONICAL_LANDMARK_NAMES, BoundingBox, DetectedFace, FacialLandmarks
from src.matching.retrieval_indexes import (
    ExactFlatIndex,
    HNSWIndex,
    IVFFlatIndex,
    IVFPQIndex,
    compute_recall_at_k,
)
from src.p3_research.algorithmic_variants import (
    ARCFACE_5PT_112,
    INTERPOLATION_FLAGS,
    GuardedFixedBatchEmbedder,
    aggregate_subject_template,
    apply_downscale_upscale,
    apply_gamma,
    apply_gaussian_blur,
    apply_jpeg_compression,
    compute_extended_quality_features,
    compute_pair_scores,
    estimate_affine_least_squares_6dof,
    evaluate_calibrated_verification,
    extract_with_flip_tta,
    gaussian_soft_nms,
    perturb_crop_roll,
    perturb_landmarks_gaussian,
)
from src.pipeline import decode_bgr_image
from src.quality.face_quality import FaceQualityAssessor
from src.recognition.base_embedder import EmbeddingBatchOutput, FaceEmbedder, l2_normalize_embeddings
from src.recognition.resnet50_embedder import ResNet50Preprocessor
from src.workbench.contracts import ExperimentRecord, WorkbenchError
from src.workbench.datasets_io import DatasetCatalog
from src.workbench.model_manager import ModelResourceManager
from src.workbench.registry import build_algorithm_registry, validate_pipeline_configuration


def encode_bgr_png_data_url(bgr: np.ndarray) -> str:
    """Encode a BGR uint8 crop or visualization image as a lossless PNG data URI."""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise WorkbenchError("Failed to encode crop image to PNG.", code="ENCODE_ERROR", status_code=500)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def make_crop_diff_heatmap_bgr(crop_a: np.ndarray, crop_b: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute per-pixel absolute difference heatmap (COLORMAP_TURBO) and mean abs diff [0..255]."""
    if crop_a.shape != crop_b.shape:
        crop_b = cv2.resize(crop_b, (crop_a.shape[1], crop_a.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(crop_a.astype(np.float32) - crop_b.astype(np.float32))
    mae = float(np.mean(diff))
    gray_diff = np.clip(np.mean(diff, axis=2) * 2.5, 0.0, 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(gray_diff, cv2.COLORMAP_TURBO)
    return heat, round(mae, 3)


def evaluate_detections_with_boxes(
    detections: list[DetectedFace],
    gt_boxes_raw: list[dict[str, float]],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Greedy confidence-ordered IoU matching returning box-level TP/FP/FN annotations and size recall."""
    gt_boxes = [
        BoundingBox(x=float(b["x"]), y=float(b["y"]), w=float(b["w"]), h=float(b["h"]))
        for b in gt_boxes_raw
    ]
    matched_gt: set[int] = set()
    det_matches: list[dict[str, Any]] = []
    ious_tp: list[float] = []

    for det_idx, det in enumerate(detections):
        best_iou = 0.0
        best_gt = -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            val = compute_iou(det.bbox, g_box)
            if val > best_iou:
                best_iou = val
                best_gt = g_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            ious_tp.append(best_iou)
            det_matches.append({"det_index": det_idx, "status": "tp", "gt_index": best_gt, "iou": round(best_iou, 4)})
        else:
            det_matches.append({"det_index": det_idx, "status": "fp", "gt_index": None, "iou": round(best_iou, 4)})

    tp = len(ious_tp)
    fp = len(detections) - tp
    fn = len(gt_boxes) - tp
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if len(gt_boxes) == 0 else 0.0)
    recall = float(tp / len(gt_boxes)) if len(gt_boxes) > 0 else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    buckets = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    unmatched_gt_list: list[dict[str, Any]] = []
    for g_idx, g_box in enumerate(gt_boxes):
        min_s = min(g_box.w, g_box.h)
        b_key = "small" if min_s < 32.0 else ("medium" if min_s <= 64.0 else "large")
        buckets[b_key][1] += 1
        if g_idx in matched_gt:
            buckets[b_key][0] += 1
        else:
            unmatched_gt_list.append(
                {
                    "gt_index": g_idx,
                    "bounding_box": g_box.to_dict(),
                    "size_category": b_key,
                }
            )

    def _b_rec(k: str) -> float | None:
        hit, tot = buckets[k]
        return round(float(hit / tot), 4) if tot > 0 else None

    return {
        "available": True,
        "iou_threshold": iou_threshold,
        "gt_count": len(gt_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(ious_tp)), 4) if ious_tp else 0.0,
        "small_lt32_recall": _b_rec("small"),
        "medium_32_64_recall": _b_rec("medium"),
        "large_gt64_recall": _b_rec("large"),
        "size_bucket_totals": {k: {"tp": v[0], "total": v[1]} for k, v in buckets.items()},
        "detection_matches": det_matches,
        "unmatched_gt_boxes": unmatched_gt_list,
    }


def _face_size_category(bbox: BoundingBox) -> str:
    m = min(bbox.w, bbox.h)
    if m < 32.0:
        return "small (<32px)"
    if m <= 64.0:
        return "medium (32-64px)"
    return "large (>64px)"


def compute_empirical_operating_point(
    genuine_scores: list[float] | np.ndarray,
    impostor_scores: list[float] | np.ndarray,
    threshold: float,
    higher_is_more_similar: bool = True,
) -> dict[str, Any]:
    """Compute exact empirical TP/FP/TN/FN and TAR/FAR/FRR/TNR directly from genuine & impostor
    score arrays, handling zero denominators explicitly without NaN or Infinity.
    """
    gen_arr = np.asarray(genuine_scores, dtype=np.float64)
    imp_arr = np.asarray(impostor_scores, dtype=np.float64)
    tau = float(threshold)
    n_gen = int(gen_arr.size)
    n_imp = int(imp_arr.size)

    if higher_is_more_similar:
        tp = int(np.sum(gen_arr >= tau)) if n_gen > 0 else 0
        fp = int(np.sum(imp_arr >= tau)) if n_imp > 0 else 0
    else:
        tp = int(np.sum(gen_arr <= tau)) if n_gen > 0 else 0
        fp = int(np.sum(imp_arr <= tau)) if n_imp > 0 else 0

    fn = n_gen - tp
    tn = n_imp - fp
    tar = (tp / n_gen) if n_gen > 0 else None
    frr = (fn / n_gen) if n_gen > 0 else None
    far = (fp / n_imp) if n_imp > 0 else None
    tnr = (tn / n_imp) if n_imp > 0 else None

    return {
        "decision_threshold": round(tau, 6),
        "higher_is_more_similar": bool(higher_is_more_similar),
        "decision_rule": "score >= threshold -> MATCH" if higher_is_more_similar else "score <= threshold -> MATCH",
        "genuine_count": n_gen,
        "impostor_count": n_imp,
        "total_pairs": n_gen + n_imp,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tar": round(float(tar), 6) if tar is not None else None,
        "frr": round(float(frr), 6) if frr is not None else None,
        "far": round(float(far), 6) if far is not None else None,
        "tnr": round(float(tnr), 6) if tnr is not None else None,
    }


class WorkbenchExecutionEngine:
    """Central stateless/cached execution coordinator calling existing Python research modules."""

    def __init__(
        self,
        model_manager: ModelResourceManager,
        dataset_catalog: DatasetCatalog,
    ) -> None:
        self.models = model_manager
        self.datasets = dataset_catalog
        self.registry = build_algorithm_registry()
        self._infer_lock = threading.RLock()
        self._gallery_lock = threading.RLock()
        self._gallery_cache: dict[str, dict[str, Any]] = {}
        self._p3_calib_lookup = self._load_p3_calibration_table()

    @staticmethod
    def _load_p3_calibration_table() -> dict[str, Any]:
        p3_path = resolve_project_path("reports/p3_research/p3_experimental_results.json")
        if not p3_path.is_file():
            return {}
        try:
            raw = json.loads(p3_path.read_text(encoding="utf-8"))
            return raw.get("exp_p3_06_matching_and_tta", {})
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    # STAGE 1: FACE DETECTION EXECUTION
    # ------------------------------------------------------------------
    def _execute_detector_on_bgr(
        self,
        image_bgr: np.ndarray,
        detector_id: str,
        params: dict[str, Any],
    ) -> tuple[list[DetectedFace], dict[str, Any], dict[str, Any]]:
        min_face_sz = float(params.get("min_face_size_px", 0.0))
        if not math.isfinite(min_face_sz) or min_face_sz < 0.0 or min_face_sz > 2048.0:
            raise WorkbenchError(f"`min_face_size_px` must be in [0, 2048]; got {min_face_sz}.", code="INVALID_PARAM")

        if detector_id == "yunet_2023mar":
            ent = self.models.get_detector("yunet_2023mar")
            yunet = ent.instance
            resize_mode = str(params.get("resize_mode", "preserve_aspect_max_dim"))
            if resize_mode not in {"preserve_aspect_max_dim", "native", "fixed"}:
                raise WorkbenchError(f"Unsupported YuNet `resize_mode`='{resize_mode}'.", code="INVALID_PARAM")
            max_dim = int(params.get("max_input_dim", 1280))
            fw = int(params.get("fixed_width", 640))
            fh = int(params.get("fixed_height", 640))
            score_th = float(params.get("score_threshold", 0.60))
            nms_mode = str(params.get("nms_mode", "hard_greedy"))
            nms_th = float(params.get("nms_threshold", 0.30))
            soft_sigma = float(params.get("soft_nms_sigma", 0.50))
            top_k = int(params.get("top_k", 5000))

            if not (0.01 <= score_th <= 0.999) or not math.isfinite(score_th):
                raise WorkbenchError(f"`score_threshold` must be in [0.01, 0.999]; got {score_th}.", code="INVALID_PARAM")
            if not (0.05 <= nms_th <= 0.99) or not math.isfinite(nms_th):
                raise WorkbenchError(f"`nms_threshold` must be in [0.05, 0.99]; got {nms_th}.", code="INVALID_PARAM")
            if not (0.05 <= soft_sigma <= 5.0) or not math.isfinite(soft_sigma):
                raise WorkbenchError(f"`soft_nms_sigma` must be in [0.05, 5.0]; got {soft_sigma}.", code="INVALID_PARAM")
            if not (64 <= max_dim <= 4096) or not (64 <= fw <= 4096) or not (64 <= fh <= 4096):
                raise WorkbenchError("Detector input dimensions must be between 64 and 4096 px.", code="INVALID_PARAM")
            if not (1 <= top_k <= 20000):
                raise WorkbenchError(f"`top_k` must be in [1, 20000]; got {top_k}.", code="INVALID_PARAM")

            with self._infer_lock:
                ent = self.models.get_detector("yunet_2023mar")
                yunet = ent.instance
                yunet.resize_mode = resize_mode
                yunet.max_input_dim = max_dim
                yunet.fixed_input_size_wh = (fw, fh)
                eff_internal_nms = 0.95 if nms_mode == "gaussian_soft_nms" else nms_th
                yunet.set_thresholds(score_threshold=score_th, nms_threshold=eff_internal_nms, top_k=top_k)
                batch_res, prof = yunet.detect_profiled(image_bgr)
            dets = list(batch_res.detections)
            if nms_mode == "gaussian_soft_nms" and len(dets) > 1:
                arr = np.array(
                    [[d.bbox.x, d.bbox.y, d.bbox.x2, d.bbox.y2, d.confidence] for d in dets],
                    dtype=np.float32,
                )
                keep, new_scores = gaussian_soft_nms(arr, sigma=soft_sigma, score_thresh=score_th)
                dets = [
                    DetectedFace(bbox=dets[i].bbox, landmarks=dets[i].landmarks, confidence=float(sc))
                    for i, sc in zip(keep, new_scores)
                ]
            meta = {
                "detector_id": detector_id,
                "model_version": batch_res.model_version,
                "model_sha256": ent.model_sha256,
                "original_size_wh": list(batch_res.original_size_wh),
                "detector_input_size_wh": list(batch_res.detector_input_size_wh),
                "scale_xy": [round(float(batch_res.scale_xy[0]), 6), round(float(batch_res.scale_xy[1]), 6)],
                "device": "cpu",
                "precision": "fp32",
            }
        elif detector_id == "scrfd_500m_research":
            req_dev = str(params.get("device", "cuda"))
            ent = self.models.get_detector("scrfd_500m_research", device=req_dev)
            wrapper = ent.instance
            max_dim = int(params.get("max_dim", 1280))
            if max_dim < 64 or max_dim % 32 != 0:
                raise WorkbenchError(f"SCRFD `max_dim` must be a multiple of 32; got {max_dim}.", code="INVALID_PARAM")
            canvas_mode = str(params.get("canvas_mode", "square_padded"))
            inv_mode = str(params.get("inverse_scale_mode", "exact_dual_xy"))
            allow_up = bool(params.get("allow_upscale", False))
            score_th = float(params.get("score_threshold", 0.50))
            nms_mode = str(params.get("nms_mode", "hard_greedy"))
            nms_th = float(params.get("nms_threshold", 0.40))
            soft_sigma = float(params.get("soft_nms_sigma", 0.50))
            top_k = int(params.get("top_k", 500))
            if not (0.01 <= score_th <= 0.999) or not math.isfinite(score_th):
                raise WorkbenchError(f"`score_threshold` must be in [0.01, 0.999]; got {score_th}.", code="INVALID_PARAM")
            if not (0.05 <= nms_th <= 0.99) or not math.isfinite(nms_th):
                raise WorkbenchError(f"`nms_threshold` must be in [0.05, 0.99]; got {nms_th}.", code="INVALID_PARAM")
            if not (1 <= top_k <= 20000):
                raise WorkbenchError(f"`top_k` must be in [1, 20000]; got {top_k}.", code="INVALID_PARAM")

            with self._infer_lock:
                ent = self.models.get_detector("scrfd_500m_research", device=req_dev)
                wrapper = ent.instance
                prev_topk = wrapper.base.top_k
                try:
                    wrapper.base.top_k = top_k
                    batch_res, prof = wrapper.detect_variant(
                        image_bgr=image_bgr,
                        max_dim=max_dim,
                        canvas_mode=canvas_mode,
                        inverse_scale_mode=inv_mode,
                        score_threshold=score_th,
                        nms_threshold=nms_th,
                        nms_mode=nms_mode,
                        soft_nms_sigma=soft_sigma,
                        allow_upscale=allow_up,
                    )
                finally:
                    wrapper.base.top_k = prev_topk
            dets = list(batch_res.detections)
            meta = {
                "detector_id": detector_id,
                "model_version": batch_res.model_version,
                "model_sha256": ent.model_sha256,
                "original_size_wh": list(batch_res.original_size_wh),
                "detector_input_size_wh": list(batch_res.detector_input_size_wh),
                "scale_xy": [round(float(batch_res.scale_xy[0]), 6), round(float(batch_res.scale_xy[1]), 6)],
                "device": ent.device,
                "precision": "fp32",
            }
        else:
            raise WorkbenchError(f"Unsupported detector_id='{detector_id}'.", code="UNKNOWN_DETECTOR")

        if min_face_sz > 0.0:
            dets = [d for d in dets if min(d.bbox.w, d.bbox.h) >= min_face_sz]

        return dets, prof, meta

    def run_detection(
        self,
        image_ref: str,
        detector_id: str = "yunet_2023mar",
        params: dict[str, Any] | None = None,
        compare_baseline: bool = False,
    ) -> dict[str, Any]:
        """Execute face detection on a single image and return pixel-accurate boxes, landmarks, and GT eval."""
        cfg_params = dict(params or {})
        img_meta = self.datasets.inspect_image(image_ref)
        path = self.datasets.resolve_image_ref(image_ref)
        bgr, _, _, decode_ms = decode_bgr_image(path)

        dets, prof, exec_meta = self._execute_detector_on_bgr(bgr, detector_id, cfg_params)
        score_floor = float(cfg_params.get("score_threshold", 0.50))

        gt_eval: dict[str, Any]
        if img_meta["ground_truth_available"] and img_meta["ground_truth_boxes"] is not None:
            gt_eval = evaluate_detections_with_boxes(dets, img_meta["ground_truth_boxes"], iou_threshold=0.50)
            match_by_idx = {m["det_index"]: m for m in gt_eval["detection_matches"]}
        else:
            gt_eval = {
                "available": False,
                "precision": None,
                "recall": None,
                "f1": None,
                "mean_iou": None,
                "small_lt32_recall": None,
                "medium_32_64_recall": None,
                "large_gt64_recall": None,
                "unmatched_gt_boxes": [],
                "reason": "Ground-truth bounding boxes are unavailable for this image.",
            }
            match_by_idx = {}

        serialized_dets: list[dict[str, Any]] = []
        for idx, d in enumerate(dets):
            m_info = match_by_idx.get(idx)
            is_low_conf = d.confidence < (score_floor + 0.12)
            serialized_dets.append(
                {
                    "face_index": idx,
                    "face_instance_id": f"{img_meta['image_id']}::face_{idx:03d}",
                    "bounding_box": d.bbox.to_dict(),
                    "dimensions_wh": [round(d.bbox.w, 2), round(d.bbox.h, 2)],
                    "size_category": _face_size_category(d.bbox),
                    "confidence": round(float(d.confidence), 6),
                    "low_confidence_flag": bool(is_low_conf),
                    "landmarks": d.landmarks.to_dict(),
                    "gt_status": m_info["status"] if m_info else "unverified",
                    "gt_iou": m_info["iou"] if m_info else None,
                }
            )

        total_ms = float(prof.get("total_detect_ms", 0.0))
        img_per_s = round(1000.0 / max(total_ms, 1e-3), 2)
        faces_per_s = round(len(dets) * 1000.0 / max(total_ms, 1e-3), 2)

        baseline_cmp: dict[str, Any] | None = None
        if compare_baseline:
            base_params = (
                {"resize_mode": "preserve_aspect_max_dim", "max_input_dim": 1280, "score_threshold": 0.60, "nms_threshold": 0.30}
                if detector_id == "yunet_2023mar"
                else {"device": exec_meta["device"], "max_dim": 1280, "canvas_mode": "square_padded", "inverse_scale_mode": "single_scalar", "score_threshold": 0.50, "nms_threshold": 0.40}
            )
            b_dets, b_prof, _ = self._execute_detector_on_bgr(bgr, detector_id, base_params)
            b_gt = (
                evaluate_detections_with_boxes(b_dets, img_meta["ground_truth_boxes"], 0.50)
                if img_meta["ground_truth_available"] and img_meta["ground_truth_boxes"] is not None
                else {"available": False, "f1": None, "recall": None, "precision": None}
            )
            baseline_cmp = {
                "baseline_params": base_params,
                "baseline_detection_count": len(b_dets),
                "current_detection_count": len(dets),
                "baseline_latency_ms": round(float(b_prof.get("total_detect_ms", 0.0)), 3),
                "current_latency_ms": round(total_ms, 3),
                "baseline_gt": b_gt,
            }

        return {
            "image_meta": img_meta,
            "execution_meta": exec_meta,
            "parameters_used": cfg_params,
            "detection_count": len(serialized_dets),
            "detections": serialized_dets,
            "gt_evaluation": gt_eval,
            "latency_and_throughput": {
                "image_decode_ms": round(decode_ms, 3),
                "total_detect_ms": round(total_ms, 3),
                "stage_breakdown_ms": {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in prof.items()},
                "images_per_sec": img_per_s,
                "faces_per_sec": faces_per_s,
            },
            "hardware_snapshot": self.models.get_hardware_snapshot(),
            "baseline_comparison": baseline_cmp,
        }

    def run_detection_sweep(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Sweep a detection parameter (`max_dim`/`max_input_dim`, `score_threshold`, `nms_threshold`, `canvas_mode`)
        over a single image or a multi-image detection split (`nasa_p11_multiscale` / `nasa_p1_detection`).
        """
        detector_id = str(config.get("detector_id", "scrfd_500m_research"))
        sweep_param = str(config.get("sweep_param", "score_threshold"))
        sweep_values = list(config.get("sweep_values", [0.30, 0.45, 0.50, 0.60, 0.75, 0.85]))
        base_params = dict(config.get("base_params", {}))
        dataset_id = str(config.get("dataset_id", "nasa_p11_multiscale"))
        single_image_ref = config.get("image_ref")

        if single_image_ref:
            targets = [{"image_ref": single_image_ref}]
            split_label = f"single_image:{single_image_ref}"
        else:
            ds_items = self.datasets.get_dataset_items(dataset_id, kind="images", limit=100)["items"]
            targets = [{"image_ref": it["image_ref"], "gt": it.get("ground_truth_boxes")} for it in ds_items]
            split_label = dataset_id

        decoded_frames = []
        for t in targets:
            p = self.datasets.resolve_image_ref(t["image_ref"])
            bgr, _, _, _ = decode_bgr_image(p)
            gt_info = self.datasets.get_ground_truth_for_path(p)
            decoded_frames.append((t["image_ref"], bgr, gt_info["boxes"] if gt_info else None))

        points: list[dict[str, Any]] = []
        total_steps = max(1, len(sweep_values))
        for step_idx, val in enumerate(sweep_values):
            if cancel_check and cancel_check():
                raise WorkbenchError("Sweep cancelled by user.", code="JOB_CANCELLED")
            if progress_cb:
                progress_cb((step_idx / total_steps) * 100.0, f"Sweeping {sweep_param}={val} ({step_idx+1}/{total_steps})")

            cur_params = {**base_params, sweep_param: val}
            latencies: list[float] = []
            tot_tp, tot_fp, tot_fn, tot_dets = 0, 0, 0, 0
            all_ious: list[float] = []
            small_hit, small_tot = 0, 0
            gt_any = False

            for _, bgr, gt_boxes in decoded_frames:
                dets, prof, _ = self._execute_detector_on_bgr(bgr, detector_id, cur_params)
                latencies.append(float(prof.get("total_detect_ms", 0.0)))
                tot_dets += len(dets)
                if gt_boxes is not None:
                    gt_any = True
                    ev = evaluate_detections_with_boxes(dets, gt_boxes, 0.50)
                    tot_tp += ev["tp"]
                    tot_fp += ev["fp"]
                    tot_fn += ev["fn"]
                    if ev["tp"] > 0:
                        all_ious.extend([m["iou"] for m in ev["detection_matches"] if m["status"] == "tp"])
                    small_hit += ev["size_bucket_totals"]["small"]["tp"]
                    small_tot += ev["size_bucket_totals"]["small"]["total"]

            p50_ms = float(np.percentile(latencies, 50)) if latencies else 0.0
            p95_ms = float(np.percentile(latencies, 95)) if latencies else 0.0
            sum_s = sum(latencies) / 1000.0
            prec = (tot_tp / (tot_tp + tot_fp)) if (gt_any and (tot_tp + tot_fp) > 0) else None
            rec = (tot_tp / (tot_tp + tot_fn)) if (gt_any and (tot_tp + tot_fn) > 0) else None
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else (0.0 if gt_any else None)

            points.append(
                {
                    "param_value": val,
                    "detection_count": tot_dets,
                    "tp": tot_tp if gt_any else None,
                    "fp": tot_fp if gt_any else None,
                    "fn": tot_fn if gt_any else None,
                    "precision": round(prec, 4) if prec is not None else None,
                    "recall": round(rec, 4) if rec is not None else None,
                    "f1": round(f1, 4) if f1 is not None else None,
                    "small_lt32_recall": round(small_hit / small_tot, 4) if (gt_any and small_tot > 0) else None,
                    "mean_iou": round(float(np.mean(all_ious)), 4) if all_ious else (0.0 if gt_any else None),
                    "p50_latency_ms": round(p50_ms, 3),
                    "p95_latency_ms": round(p95_ms, 3),
                    "images_per_sec": round(len(decoded_frames) / max(sum_s, 1e-6), 2),
                    "faces_per_sec": round(tot_dets / max(sum_s, 1e-6), 2),
                }
            )

        if progress_cb:
            progress_cb(100.0, "Detection sweep completed.")

        best_f1 = max((pt["f1"] for pt in points if pt["f1"] is not None), default=None)
        return {
            "detector_id": detector_id,
            "sweep_param": sweep_param,
            "dataset_split": split_label,
            "points": points,
            "summary_metrics": {
                "num_sweep_points": len(points),
                "best_f1": best_f1,
                "min_p50_latency_ms": min((pt["p50_latency_ms"] for pt in points), default=0.0),
            },
        }

    # ------------------------------------------------------------------
    # STAGE 2: LANDMARK & ALIGNMENT LAB EXECUTION
    # ------------------------------------------------------------------
    def _compute_single_alignment(
        self,
        bgr: np.ndarray,
        src_pts_5x2: np.ndarray,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        transform_mode = str(params.get("transform_mode", "similarity_umeyama_4dof"))
        interpolation = str(params.get("interpolation", "linear"))
        if interpolation not in INTERPOLATION_FLAGS:
            raise WorkbenchError(f"Unsupported interpolation='{interpolation}'.", code="INVALID_PARAM")

        crop_w = int(params.get("crop_width", 112))
        crop_h = int(params.get("crop_height", 112))
        zoom_factor = float(params.get("zoom_factor", 1.0))
        extra_rot = float(params.get("extra_rotation_deg", 0.0))
        tx = float(params.get("translate_x_px", 0.0))
        ty = float(params.get("translate_y_px", 0.0))
        jitter_sigma = float(params.get("jitter_sigma_px", 0.0))
        seed = int(params.get("seed", 42))

        work_src = np.asarray(src_pts_5x2, dtype=np.float64).copy()
        if jitter_sigma > 0.0:
            rng = np.random.default_rng(seed)
            work_src = perturb_landmarks_gaussian(work_src, jitter_sigma, rng)

        scale_vec = np.array([crop_w / 112.0, crop_h / 112.0], dtype=np.float64)
        base_dst = ARCFACE_5PT_112 * scale_vec
        center = np.array([crop_w * 0.5, crop_h * 0.5], dtype=np.float64)
        eff_zoom = zoom_factor * (0.90 if transform_mode == "similarity_zoom_out_10pct" else 1.0)
        dst = center + eff_zoom * (base_dst - center) + np.array([tx, ty], dtype=np.float64)

        t0 = time.perf_counter()
        if transform_mode in {"similarity_umeyama_4dof", "similarity_zoom_out_10pct"}:
            mat_2x3 = _estimate_similarity_2d_umeyama(work_src, dst)
        elif transform_mode == "affine_least_squares_6dof":
            mat_2x3 = estimate_affine_least_squares_6dof(work_src, dst)
        else:
            raise WorkbenchError(f"Unsupported transform_mode='{transform_mode}'.", code="INVALID_PARAM")

        warped = cv2.warpAffine(
            bgr,
            mat_2x3,
            (crop_w, crop_h),
            flags=INTERPOLATION_FLAGS[interpolation],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        if abs(extra_rot) > 1e-6:
            warped = perturb_crop_roll(warped, extra_rot)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Project source landmarks into aligned crop space
        ones = np.ones((5, 1), dtype=np.float64)
        transformed_pts = (np.hstack([work_src, ones]) @ mat_2x3.T)
        residual_px = np.linalg.norm(transformed_pts - dst, axis=1)

        # Back-project aligned crop corners into original image coordinates (exact rotated quad)
        inv_mat = cv2.invertAffineTransform(mat_2x3)
        corners_h = np.array(
            [[0.0, 0.0, 1.0], [crop_w, 0.0, 1.0], [crop_w, crop_h, 1.0], [0.0, crop_h, 1.0]],
            dtype=np.float64,
        )
        quad_orig = (inv_mat @ corners_h.T).T

        # Decompose 2x2 linear part A
        lin_a = mat_2x3[:, :2]
        u_s, s_vals, _ = np.linalg.svd(lin_a)
        rot_rad = math.atan2(mat_2x3[1, 0], mat_2x3[0, 0])
        anisotropy = float(s_vals[0] / max(s_vals[1], 1e-9)) - 1.0
        io_px = float(np.linalg.norm(work_src[1] - work_src[0]))

        return warped, {
            "transform_mode": transform_mode,
            "interpolation": interpolation,
            "crop_size_wh": [crop_w, crop_h],
            "source_landmarks_orig_px": np.round(src_pts_5x2, 3).tolist(),
            "perturbed_source_landmarks_orig_px": np.round(work_src, 3).tolist(),
            "destination_template_crop_px": np.round(dst, 3).tolist(),
            "transformed_landmarks_crop_px": np.round(transformed_pts, 3).tolist(),
            "landmark_fit_residual_px": np.round(residual_px, 4).tolist(),
            "mean_fit_residual_px": round(float(np.mean(residual_px)), 4),
            "affine_matrix_2x3": np.round(mat_2x3, 6).tolist(),
            "crop_boundary_quad_orig_px": np.round(quad_orig, 2).tolist(),
            "decomposition": {
                "mean_scale": round(float(np.mean(s_vals)), 5),
                "rotation_deg": round(math.degrees(rot_rad), 3),
                "translation_xy_px": [round(float(mat_2x3[0, 2]), 3), round(float(mat_2x3[1, 2]), 3)],
                "anisotropy_ratio": round(anisotropy, 5),
                "inter_ocular_orig_px": round(io_px, 3),
            },
            "execution_time_ms": round(elapsed_ms, 4),
            "crop_data_url": encode_bgr_png_data_url(warped),
        }

    def run_alignment_lab(
        self,
        image_ref: str,
        face_index: int = 0,
        landmark_source: str = "yunet_2023mar",
        compare_both_detectors: bool = True,
        compare_transform_mode: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run interactive Landmark & Alignment Lab analysis and dual-landmark/transform comparison."""
        align_params = dict(params or {})
        img_meta = self.datasets.inspect_image(image_ref)
        path = self.datasets.resolve_image_ref(image_ref)
        bgr, _, _, _ = decode_bgr_image(path)

        dets_primary, _, _ = self._execute_detector_on_bgr(bgr, landmark_source, {})
        if not dets_primary:
            raise WorkbenchError(f"No faces detected by '{landmark_source}' on '{image_ref}'.", code="NO_FACES_DETECTED")
        f_idx = max(0, min(int(face_index), len(dets_primary) - 1))
        face_primary = dets_primary[f_idx]
        pts_primary = face_primary.landmarks.as_array().astype(np.float64)

        crop_primary, primary_res = self._compute_single_alignment(bgr, pts_primary, align_params)
        primary_res["landmark_source"] = landmark_source
        primary_res["selected_face_index"] = f_idx
        primary_res["bounding_box"] = face_primary.bbox.to_dict()
        primary_res["confidence"] = round(float(face_primary.confidence), 5)

        secondary_res: dict[str, Any] | None = None
        displacement_summary: dict[str, Any] | None = None
        diff_url: str | None = None
        crop_mae: float | None = None

        other_det = "scrfd_500m_research" if landmark_source == "yunet_2023mar" else "yunet_2023mar"
        if compare_both_detectors:
            dets_other, _, _ = self._execute_detector_on_bgr(bgr, other_det, {})
            best_other: DetectedFace | None = None
            best_iou = 0.0
            for d_o in dets_other:
                v = compute_iou(face_primary.bbox, d_o.bbox)
                if v > best_iou:
                    best_iou = v
                    best_other = d_o
            if best_other is not None and best_iou >= 0.25:
                pts_other = best_other.landmarks.as_array().astype(np.float64)
                crop_sec, secondary_res = self._compute_single_alignment(bgr, pts_other, align_params)
                secondary_res["landmark_source"] = other_det
                secondary_res["bounding_box"] = best_other.bbox.to_dict()
                secondary_res["confidence"] = round(float(best_other.confidence), 5)

                per_pt_disp = np.linalg.norm(pts_primary - pts_other, axis=1)
                displacement_summary = {
                    "compared_sources": [landmark_source, other_det],
                    "bbox_iou": round(best_iou, 4),
                    "landmark_names": list(CANONICAL_LANDMARK_NAMES),
                    "per_point_displacement_px": np.round(per_pt_disp, 3).tolist(),
                    "mean_displacement_px": round(float(np.mean(per_pt_disp)), 3),
                    "max_displacement_px": round(float(np.max(per_pt_disp)), 3),
                }
                heat, crop_mae = make_crop_diff_heatmap_bgr(crop_primary, crop_sec)
                diff_url = encode_bgr_png_data_url(heat)
        elif compare_transform_mode:
            alt_params = {**align_params, "transform_mode": compare_transform_mode}
            crop_sec, secondary_res = self._compute_single_alignment(bgr, pts_primary, alt_params)
            secondary_res["landmark_source"] = f"{landmark_source} ({compare_transform_mode})"
            secondary_res["bounding_box"] = face_primary.bbox.to_dict()
            heat, crop_mae = make_crop_diff_heatmap_bgr(crop_primary, crop_sec)
            diff_url = encode_bgr_png_data_url(heat)

        return {
            "image_meta": img_meta,
            "all_detected_faces": [
                {
                    "face_index": i,
                    "bounding_box": d.bbox.to_dict(),
                    "confidence": round(float(d.confidence), 5),
                }
                for i, d in enumerate(dets_primary)
            ],
            "canonical_landmark_names": list(CANONICAL_LANDMARK_NAMES),
            "primary_alignment": primary_res,
            "secondary_alignment": secondary_res,
            "landmark_displacement": displacement_summary,
            "crop_difference_data_url": diff_url,
            "crop_mean_abs_pixel_diff": crop_mae,
            "geometry_note": "Overlays depict geometric 5-point landmark correspondences (source -> canonical template), never recognition embedding similarity.",
        }

    # ------------------------------------------------------------------
    # HELPER: RECOGNITION EXTRACTION WITH PREPROCESSING OVERRIDE SUPPORT
    # ------------------------------------------------------------------
    def _extract_embeddings_for_crops(
        self,
        crops_bgr: list[np.ndarray],
        recognizer_id: str,
        rec_params: dict[str, Any],
    ) -> tuple[EmbeddingBatchOutput, str | None, dict[str, Any]]:
        runtime = rec_params.get("runtime")
        device = rec_params.get("device")
        precision = rec_params.get("precision")
        batch_policy = str(rec_params.get("batch_policy", "guarded_fixed_padded"))
        flip_tta = bool(rec_params.get("flip_tta", False))
        l2_norm = bool(rec_params.get("l2_normalize", True))
        allow_override = bool(rec_params.get("allow_experimental_preprocess_override", False))
        override_color = str(rec_params.get("override_color_order", "RGB")).upper()

        ent = self.models.get_recognizer(recognizer_id, runtime=runtime, device=device, precision=precision)
        base_emb: FaceEmbedder = ent.instance
        dyn_supported = bool(getattr(base_emb, "dynamic_batch_supported", True))
        default_bs = 8 if dyn_supported else 1
        batch_size = int(rec_params.get("batch_size", default_bs))

        if not dyn_supported:
            if batch_size != 1:
                raise WorkbenchError(
                    f"Model '{recognizer_id}' has a static `[1, 3, 112, 112]` graph and rejects `batch_size={batch_size}`.",
                    code="STATIC_BATCH_VIOLATION",
                )
            active_emb: FaceEmbedder = base_emb
        else:
            if batch_policy == "guarded_fixed_padded":
                active_emb = GuardedFixedBatchEmbedder(base_emb, fixed_batch_size=batch_size)
            else:
                if hasattr(base_emb, "batch_size"):
                    setattr(base_emb, "batch_size", batch_size)
                active_emb = base_emb

        prep_warning: str | None = None
        work_crops = crops_bgr
        if allow_override and override_color == "BGR":
            # Canonical models swap BGR->RGB inside preprocess; pre-swapping feeds BGR channels into RGB network
            work_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
            prep_warning = (
                "EXPERIMENTAL OVERRIDE ACTIVE: Input color channels were swapped (BGR fed to RGB network). "
                "Resulting embeddings and similarity scores are intentionally degraded for ablation analysis."
            )

        with self._infer_lock:
            if flip_tta:
                out = extract_with_flip_tta(active_emb, work_crops)
                if not l2_norm:
                    out = EmbeddingBatchOutput(
                        raw_embeddings=out.raw_embeddings,
                        embeddings=out.raw_embeddings.copy(),
                        l2_normalized=False,
                        embedding_dim=out.embedding_dim,
                        model_id=out.model_id,
                        model_version=out.model_version,
                        runtime=out.runtime,
                        device=out.device,
                        precision=out.precision,
                        is_synthetic_fixture=False,
                        preprocess_time_ms=out.preprocess_time_ms,
                        inference_time_ms=out.inference_time_ms,
                        postprocess_time_ms=out.postprocess_time_ms,
                    )
            else:
                out = active_emb.extract(work_crops, l2_normalize=l2_norm)

        meta = {
            "recognizer_id": recognizer_id,
            "model_version": out.model_version,
            "model_sha256": ent.model_sha256,
            "runtime": out.runtime,
            "device": out.device,
            "precision": out.precision,
            "embedding_dim": out.embedding_dim,
            "l2_normalized": out.l2_normalized,
            "flip_tta": flip_tta,
            "batch_size": 1 if not bool(getattr(base_emb, "dynamic_batch_supported", True)) else batch_size,
            "batch_policy": "static_b1" if not bool(getattr(base_emb, "dynamic_batch_supported", True)) else batch_policy,
        }
        return out, prep_warning, meta

    # ------------------------------------------------------------------
    # STAGE 3: FACE QUALITY ASSESSMENT LAB
    # ------------------------------------------------------------------
    def run_quality_lab(
        self,
        image_refs: list[str],
        detector_id: str = "yunet_2023mar",
        recognizer_id: str = "adaface_ir50_webface4m",
        quality_params: dict[str, Any] | None = None,
        degradation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate P1 classical + P3 extended quality indicators, ranking, filtering, and template pooling."""
        if not image_refs:
            raise WorkbenchError("At least one `image_ref` is required.", code="EMPTY_INPUT")
        q_cfg = dict(quality_params or {})
        deg = dict(degradation or {})

        assessor = FaceQualityAssessor(
            warning_thresholds={
                k: float(q_cfg[k])
                for k in (
                    "min_face_width_px",
                    "min_sharpness_laplacian_var",
                    "min_brightness_mean",
                    "max_brightness_mean",
                    "min_contrast_std",
                )
                if k in q_cfg
            }
        )
        aligner = FaceAligner()

        records: list[dict[str, Any]] = []
        crops_for_rec: list[np.ndarray] = []
        clean_crops_for_rec: list[np.ndarray] = []

        blur_s = float(deg.get("blur_sigma", 0.0))
        eff_res = int(deg.get("effective_res", 112))
        gamma_v = float(deg.get("gamma", 1.0))
        jpeg_q = int(deg.get("jpeg_quality", 100))
        has_deg = (blur_s > 0.0) or (eff_res < 112) or (abs(gamma_v - 1.0) > 1e-3) or (jpeg_q < 100)

        for ref in image_refs[:24]:
            path = self.datasets.resolve_image_ref(ref)
            bgr, stem_id, _, _ = decode_bgr_image(path)
            dets, _, _ = self._execute_detector_on_bgr(bgr, detector_id, {})
            for f_idx, det in enumerate(dets):
                aligned = aligner.align(bgr, det.landmarks, source_bbox=det.bbox)
                clean_crop = aligned.aligned_bgr
                eval_crop = clean_crop.copy()
                if blur_s > 0.0:
                    eval_crop = apply_gaussian_blur(eval_crop, blur_s)
                if eff_res < 112:
                    eval_crop = apply_downscale_upscale(eval_crop, eff_res)
                if abs(gamma_v - 1.0) > 1e-3:
                    eval_crop = apply_gamma(eval_crop, gamma_v)
                if jpeg_q < 100:
                    eval_crop = apply_jpeg_compression(eval_crop, jpeg_q)

                p1_q = assessor.assess(bgr, det.bbox, aligned_crop_bgr=eval_crop)
                crops_for_rec.append(eval_crop)
                clean_crops_for_rec.append(clean_crop)
                records.append(
                    {
                        "image_ref": ref,
                        "face_instance_id": f"{stem_id}::face_{f_idx:03d}",
                        "det_confidence": float(det.confidence),
                        "landmarks_arr": det.landmarks.as_array(),
                        "bbox": det.bbox.to_dict(),
                        "p1_classical": p1_q.to_dict(),
                        "eval_crop": eval_crop,
                    }
                )

        if not records:
            return {"items": [], "ranking": [], "template_aggregation_preview": None}

        emb_out, _, rec_meta = self._extract_embeddings_for_crops(crops_for_rec, recognizer_id, {})
        raw_norms = np.linalg.norm(emb_out.raw_embeddings.astype(np.float64), axis=1)

        self_clean_cosines: np.ndarray | None = None
        if has_deg:
            clean_out, _, _ = self._extract_embeddings_for_crops(clean_crops_for_rec, recognizer_id, {})
            self_clean_cosines = np.sum(emb_out.embeddings * clean_out.embeddings, axis=1)

        primary_metric = str(q_cfg.get("primary_quality_metric", "learned_embedding_l2_norm"))
        reject_frac = float(q_cfg.get("reject_quantile", 0.05))

        items_out: list[dict[str, Any]] = []
        classical_weights: list[float] = []
        for idx, rec in enumerate(records):
            r_norm = float(raw_norms[idx])
            ext = compute_extended_quality_features(
                rec["eval_crop"],
                landmarks_5x2=rec["landmarks_arr"],
                det_confidence=rec["det_confidence"],
                raw_embedding_norm=r_norm,
            )
            classical_weights.append(ext["classical_composite_score"])

            comp = ext["classical_composite_score"]
            if comp >= 0.72 and len(rec["p1_classical"]["warnings"]) == 0:
                q_cat = "HIGH"
            elif comp >= 0.50:
                q_cat = "MEDIUM"
            else:
                q_cat = "LOW"

            items_out.append(
                {
                    "index": idx,
                    "image_ref": rec["image_ref"],
                    "face_instance_id": rec["face_instance_id"],
                    "bounding_box": rec["bbox"],
                    "landmarks_points": [[round(float(x), 3), round(float(y), 3)] for x, y in rec["landmarks_arr"].tolist()],
                    "confidence": round(float(rec["det_confidence"]), 5),
                    "crop_data_url": encode_bgr_png_data_url(rec["eval_crop"]),
                    "classical_p1": rec["p1_classical"],
                    "extended_p3": ext,
                    "quality_category": q_cat,
                    "clean_vs_degraded_self_cosine": round(float(self_clean_cosines[idx]), 5)
                    if self_clean_cosines is not None
                    else None,
                    "unimplemented_metrics_disclosure": {
                        "occlusion_classifier": "NOT_IMPLEMENTED (never fabricated)",
                    },
                    "measurement_semantics": {
                        "raw_pixel_measurements": [
                            "sharpness_laplacian_var",
                            "sharpness_tenengrad",
                            "mean_luminance",
                            "contrast_std",
                            "exposure_score",
                            "jpeg_blockiness_ratio",
                            "inter_ocular_px",
                            "yaw_asymmetry_ratio",
                            "roll_angle_deg",
                        ],
                        "model_dependent_uncalibrated_proxy": ["learned_embedding_l2_norm"],
                    },
                }
            )

        metric_vals = np.array([float(it["extended_p3"].get(primary_metric, 0.0)) for it in items_out], dtype=np.float64)
        cutoff = float(np.quantile(metric_vals, reject_frac)) if reject_frac > 0.0 and len(metric_vals) > 1 else -1e9
        for it, mv in zip(items_out, metric_vals):
            it["active_metric_name"] = primary_metric
            it["active_metric_value"] = round(float(mv), 5)
            it["retained_after_filter"] = bool(mv >= cutoff)

        ranked = sorted(items_out, key=lambda x: x["active_metric_value"], reverse=True)

        # Multi-shot template aggregation weights preview across loaded shots
        raw_mat = emb_out.raw_embeddings
        centered = (raw_norms - np.mean(raw_norms)) / max(float(np.std(raw_norms)), 1e-6)
        w_norm = np.exp(np.clip(centered, -4.0, 4.0))
        w_norm = (w_norm / np.sum(w_norm)).tolist()
        cw_arr = np.maximum(np.asarray(classical_weights, dtype=np.float64), 1e-6)
        w_class = (cw_arr / np.sum(cw_arr)).tolist()

        tmpl_norm = aggregate_subject_template(raw_mat, mode="learned_norm_weighted")
        tmpl_unif = aggregate_subject_template(raw_mat, mode="uniform_mean")
        tmpl_clas = aggregate_subject_template(raw_mat, mode="classical_quality_weighted", external_weights=cw_arr)

        return {
            "recognizer_meta": rec_meta,
            "degradation_applied": {"enabled": has_deg, **deg},
            "filter_summary": {
                "metric": primary_metric,
                "reject_quantile": reject_frac,
                "cutoff_value": round(cutoff, 5) if cutoff > -1e8 else None,
                "total_faces": len(items_out),
                "retained_faces": sum(1 for it in items_out if it["retained_after_filter"]),
                "rejected_faces": sum(1 for it in items_out if not it["retained_after_filter"]),
            },
            "items": items_out,
            "ranking_ids": [r["face_instance_id"] for r in ranked],
            "template_aggregation_preview": {
                "shot_count": len(items_out),
                "uniform_weights": [round(1.0 / len(items_out), 4)] * len(items_out),
                "classical_quality_weights": [round(float(x), 4) for x in w_class],
                "learned_norm_weights": [round(float(x), 4) for x in w_norm],
                "template_cosine_norm_vs_uniform": round(float(np.dot(tmpl_norm, tmpl_unif)), 6),
                "template_cosine_classical_vs_uniform": round(float(np.dot(tmpl_clas, tmpl_unif)), 6),
            },
        }

    # ------------------------------------------------------------------
    # STAGE 4: 1:1 RECOGNITION & VERIFICATION LAB
    # ------------------------------------------------------------------
    def _lookup_calibration_context(
        self,
        recognizer_id: str,
        metric: str,
        flip_tta: bool,
        threshold: float,
    ) -> dict[str, Any]:
        model_map = {
            "adaface_ir50_webface4m": "adaface_ir50_fp16",
            "mobilefacenet_w600k_research": "mobilefacenet_w600k_fp32",
        }
        m_key = model_map.get(recognizer_id)
        if not m_key or not self._p3_calib_lookup:
            return {
                "calibrated_reference_available": False,
                "calibration_source": "User-Selected Threshold (No precomputed LFW calibration table for this model/mode)",
            }

        if flip_tta and metric == "cosine":
            block = self._p3_calib_lookup.get("horizontal_flip_tta", {}).get(m_key, {}).get("flip_tta_calib_test", {})
            src_name = f"LFW Folds 1–5 Identity-Disjoint Calibration ({m_key}, Flip-TTA, cosine)"
        else:
            block = self._p3_calib_lookup.get("distance_metrics", {}).get(metric, {}).get(m_key, {})
            src_name = f"LFW Folds 1–5 Identity-Disjoint Calibration ({m_key}, Single-Pass, {metric})"

        if not block:
            return {
                "calibrated_reference_available": False,
                "calibration_source": "User-Selected Threshold",
            }

        return {
            "calibrated_reference_available": True,
            "calibration_source": src_name,
            "calibrated_acc_threshold": block.get("calibrated_acc_threshold"),
            "test_eer_pct": block.get("test_eer_pct"),
            "test_roc_auc": block.get("test_roc_auc"),
            "operating_points": block.get("calibrated_operating_points", {}),
            "selected_threshold": round(float(threshold), 6),
        }

    def run_verification_1to1(
        self,
        image_a_ref: str,
        image_b_ref: str,
        face_index_a: int = 0,
        face_index_b: int = 0,
        detector_id: str = "yunet_2023mar",
        detector_params: dict[str, Any] | None = None,
        alignment_params: dict[str, Any] | None = None,
        recognizer_id: str = "adaface_ir50_webface4m",
        recognizer_params: dict[str, Any] | None = None,
        metric: str = "cosine",
        threshold: float = 0.2169,
    ) -> dict[str, Any]:
        """Run full 1:1 verification between a selected face in Image A and a selected face in Image B."""
        det_p = dict(detector_params or {})
        align_p = dict(alignment_params or {})
        rec_p = dict(recognizer_params or {})

        meta_a = self.datasets.inspect_image(image_a_ref)
        meta_b = self.datasets.inspect_image(image_b_ref)
        bgr_a, _, _, dec_a_ms = decode_bgr_image(self.datasets.resolve_image_ref(image_a_ref))
        bgr_b, _, _, dec_b_ms = decode_bgr_image(self.datasets.resolve_image_ref(image_b_ref))

        dets_a, prof_a, _ = self._execute_detector_on_bgr(bgr_a, detector_id, det_p)
        dets_b, prof_b, _ = self._execute_detector_on_bgr(bgr_b, detector_id, det_p)
        if not dets_a:
            raise WorkbenchError(f"Zero faces detected in Image A ('{image_a_ref}').", code="NO_FACE_IMAGE_A")
        if not dets_b:
            raise WorkbenchError(f"Zero faces detected in Image B ('{image_b_ref}').", code="NO_FACE_IMAGE_B")

        idx_a = max(0, min(int(face_index_a), len(dets_a) - 1))
        idx_b = max(0, min(int(face_index_b), len(dets_b) - 1))
        fa = dets_a[idx_a]
        fb = dets_b[idx_b]

        crop_a, al_a = self._compute_single_alignment(bgr_a, fa.landmarks.as_array(), align_p)
        crop_b, al_b = self._compute_single_alignment(bgr_b, fb.landmarks.as_array(), align_p)

        assessor = FaceQualityAssessor()
        qa_p1 = assessor.assess(bgr_a, fa.bbox, aligned_crop_bgr=crop_a)
        qb_p1 = assessor.assess(bgr_b, fb.bbox, aligned_crop_bgr=crop_b)

        emb_out, prep_warn, rec_meta = self._extract_embeddings_for_crops([crop_a, crop_b], recognizer_id, rec_p)
        raw_a = emb_out.raw_embeddings[0:1]
        raw_b = emb_out.raw_embeddings[1:2]
        norm_a = l2_normalize_embeddings(raw_a)[0]
        norm_b = l2_normalize_embeddings(raw_b)[0]

        norm_val_a = float(np.linalg.norm(raw_a[0]))
        norm_val_b = float(np.linalg.norm(raw_b[0]))
        qa_ext = compute_extended_quality_features(crop_a, fa.landmarks.as_array(), fa.confidence, norm_val_a)
        qb_ext = compute_extended_quality_features(crop_b, fb.landmarks.as_array(), fb.confidence, norm_val_b)

        all_metrics = {
            "cosine": round(float(compute_pair_scores(raw_a, raw_b, "cosine")[0]), 6),
            "normalized_l2": round(float(compute_pair_scores(raw_a, raw_b, "normalized_l2")[0]), 6),
            "unnormalized_l2": round(float(compute_pair_scores(raw_a, raw_b, "unnormalized_l2")[0]), 6),
            "normalized_l1": round(float(compute_pair_scores(raw_a, raw_b, "normalized_l1")[0]), 6),
            "positive_unit_l2_distance": round(float(np.linalg.norm(norm_a - norm_b)), 6),
            "positive_raw_l2_distance": round(float(np.linalg.norm(raw_a[0] - raw_b[0])), 6),
        }
        if metric not in {"cosine", "normalized_l2", "unnormalized_l2", "normalized_l1"}:
            raise WorkbenchError(f"Unsupported metric='{metric}'.", code="INVALID_METRIC")

        active_score = all_metrics[metric]
        tau = float(threshold)
        is_match = bool(active_score >= tau)

        heat_bgr, crop_mae = make_crop_diff_heatmap_bgr(crop_a, crop_b)
        lm_disp_crop = np.linalg.norm(
            np.asarray(al_a["transformed_landmarks_crop_px"]) - np.asarray(al_b["transformed_landmarks_crop_px"]),
            axis=1,
        )

        calib_ctx = self._lookup_calibration_context(
            recognizer_id=recognizer_id,
            metric=metric,
            flip_tta=bool(rec_p.get("flip_tta", False)),
            threshold=tau,
        )

        return {
            "image_a": {
                "meta": meta_a,
                "selected_face_index": idx_a,
                "all_faces": [
                    {"face_index": i, "bounding_box": d.bbox.to_dict(), "confidence": round(float(d.confidence), 5)}
                    for i, d in enumerate(dets_a)
                ],
                "face": {
                    "face_instance_id": f"{meta_a['image_id']}::face_{idx_a:03d}",
                    "bounding_box": fa.bbox.to_dict(),
                    "confidence": round(float(fa.confidence), 6),
                    "landmarks": fa.landmarks.to_dict(),
                    "alignment": al_a,
                    "quality_p1": qa_p1.to_dict(),
                    "quality_extended": qa_ext,
                    "raw_embedding_norm": round(norm_val_a, 5),
                    "unit_embedding_norm": round(float(np.linalg.norm(emb_out.embeddings[0])), 5),
                    "embedding_preview_first12": [round(float(x), 5) for x in emb_out.embeddings[0, :12]],
                },
            },
            "image_b": {
                "meta": meta_b,
                "selected_face_index": idx_b,
                "all_faces": [
                    {"face_index": i, "bounding_box": d.bbox.to_dict(), "confidence": round(float(d.confidence), 5)}
                    for i, d in enumerate(dets_b)
                ],
                "face": {
                    "face_instance_id": f"{meta_b['image_id']}::face_{idx_b:03d}",
                    "bounding_box": fb.bbox.to_dict(),
                    "confidence": round(float(fb.confidence), 6),
                    "landmarks": fb.landmarks.to_dict(),
                    "alignment": al_b,
                    "quality_p1": qb_p1.to_dict(),
                    "quality_extended": qb_ext,
                    "raw_embedding_norm": round(norm_val_b, 5),
                    "unit_embedding_norm": round(float(np.linalg.norm(emb_out.embeddings[1])), 5),
                    "embedding_preview_first12": [round(float(x), 5) for x in emb_out.embeddings[1, :12]],
                },
            },
            "recognizer_meta": rec_meta,
            "preprocessing_warning": prep_warn,
            "metrics": all_metrics,
            "decision": {
                "metric": metric,
                "score": active_score,
                "threshold": round(tau, 6),
                "margin_from_threshold": round(active_score - tau, 6),
                "verdict": "MATCH (Above Threshold)" if is_match else "NON-MATCH (Below Threshold)",
                "is_match": is_match,
                "score_semantics_disclosure": (
                    "Raw geometric similarity/distance score on embedding vectors — NOT a calibrated match probability."
                ),
                "calibration_context": calib_ctx,
            },
            "visual_comparison": {
                "crop_diff_heatmap_data_url": encode_bgr_png_data_url(heat_bgr),
                "crop_mean_abs_diff": crop_mae,
                "aligned_landmark_displacement_px": np.round(lm_disp_crop, 3).tolist(),
                "correspondence_disclosure": (
                    "Correspondence lines connect detected 5-point anatomical landmarks (`YuNet`/`SCRFD`). "
                    "Global CNN embeddings do not produce local keypoint matches; no local descriptors are fabricated."
                ),
            },
            "stage_latency_ms": {
                "decode_ms": round(dec_a_ms + dec_b_ms, 3),
                "detection_ms": round(float(prof_a.get("total_detect_ms", 0.0)) + float(prof_b.get("total_detect_ms", 0.0)), 3),
                "alignment_ms": round(al_a["execution_time_ms"] + al_b["execution_time_ms"], 3),
                "recognition_preprocess_ms": round(emb_out.preprocess_time_ms, 3),
                "recognition_inference_ms": round(emb_out.inference_time_ms, 3),
                "recognition_postprocess_ms": round(emb_out.postprocess_time_ms, 3),
            },
        }

    # ------------------------------------------------------------------
    # STAGE 5: 1:N CANDIDATE SEARCH & RETRIEVAL BENCHMARK
    # ------------------------------------------------------------------
    def _get_or_build_gallery_embeddings(
        self,
        gallery_source: str,
        gallery_limit: int,
        detector_id: str,
        recognizer_id: str,
        rec_params: dict[str, Any],
        custom_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        runtime = str(rec_params.get("runtime", "onnxruntime"))
        device = str(rec_params.get("device", "cuda"))
        precision = str(rec_params.get("precision", "fp16" if recognizer_id == "adaface_ir50_webface4m" else "fp32"))
        flip_tta = bool(rec_params.get("flip_tta", False))
        cap = max(4, min(int(gallery_limit), 600))

        cache_key = (
            f"{gallery_source}::{cap}::{detector_id}::{recognizer_id}::{runtime}::{device}::{precision}::{flip_tta}"
            if gallery_source != "custom_refs"
            else f"custom::{hash(tuple(custom_refs or []))}::{detector_id}::{recognizer_id}::{precision}"
        )
        with self._gallery_lock:
            if cache_key in self._gallery_cache:
                return self._gallery_cache[cache_key]

        if gallery_source == "nasa_p11_verification":
            ds = self.datasets.get_dataset_items("nasa_p11_verification", kind="images", limit=100)["items"]
            refs = [(it["image_ref"], it.get("subject_tag_eval_only", "unknown")) for it in ds[:cap]]
        elif gallery_source == "lfw_sample":
            ds = self.datasets.get_dataset_items(
                "lfw_research_split", kind="images", split_filter="test_folds_6_10", limit=200
            )["items"]
            # If cap > 200, fetch additional pages
            offset = 200
            while len(ds) < cap:
                more = self.datasets.get_dataset_items(
                    "lfw_research_split", kind="images", split_filter="test_folds_6_10", offset=offset, limit=200
                )["items"]
                if not more:
                    break
                ds.extend(more)
                offset += 200
            refs = [(it["image_ref"], it.get("subject_tag_eval_only", "lfw_template")) for it in ds[:cap]]
        elif gallery_source == "custom_refs" and custom_refs:
            refs = [(r, "session_template") for r in custom_refs[:cap]]
        else:
            raise WorkbenchError(f"Unsupported gallery_source='{gallery_source}'.", code="INVALID_GALLERY")

        aligner = FaceAligner()
        crops: list[np.ndarray] = []
        entries: list[dict[str, Any]] = []
        for idx, (ref, tag) in enumerate(refs):
            p = self.datasets.resolve_image_ref(ref)
            bgr, stem, _, _ = decode_bgr_image(p)
            dets, _, _ = self._execute_detector_on_bgr(bgr, detector_id, {})
            if not dets:
                continue
            al = aligner.align(bgr, dets[0].landmarks, source_bbox=dets[0].bbox)
            crops.append(al.aligned_bgr)
            entries.append(
                {
                    "template_id": f"TMPL-{idx:04d} ({stem})",
                    "anonymized_template_tag": f"eval_tag::{tag}",
                    "image_ref": ref,
                    "crop_data_url": encode_bgr_png_data_url(al.aligned_bgr),
                }
            )

        if not crops:
            raise WorkbenchError("Gallery extraction produced 0 valid face templates.", code="EMPTY_GALLERY")

        emb_out, _, rec_meta = self._extract_embeddings_for_crops(
            crops,
            recognizer_id,
            {"runtime": runtime, "device": device, "precision": precision, "flip_tta": flip_tta, "l2_normalize": True},
        )
        bundle = {
            "vectors": np.ascontiguousarray(emb_out.embeddings, dtype=np.float32),
            "entries": entries,
            "recognizer_meta": rec_meta,
        }
        with self._gallery_lock:
            if len(self._gallery_cache) >= 6:
                oldest = next(iter(self._gallery_cache))
                self._gallery_cache.pop(oldest, None)
            self._gallery_cache[cache_key] = bundle
        return bundle

    @staticmethod
    def _build_index(
        index_id: str,
        vectors: np.ndarray,
        params: dict[str, Any],
    ) -> tuple[Any, float]:
        n, d = vectors.shape
        t0 = time.perf_counter()
        if index_id == "exact_flat_index":
            backend = str(params.get("backend", "numpy_cpu"))
            metric = str(params.get("metric", "ip"))
            idx: Any = ExactFlatIndex(dim=d, metric=metric, backend=backend)
            idx.add(vectors)
        elif index_id == "ivf_flat_index":
            nlist = min(int(params.get("nlist", 64)), n)
            train_max = int(params.get("train_sample_max", 25000))
            idx = IVFFlatIndex(dim=d, nlist=max(1, nlist), seed=42)
            idx.train_and_add(vectors, train_sample_max=train_max)
        elif index_id == "ivf_pq_index":
            nlist = min(int(params.get("nlist", 64)), n)
            m_sub = int(params.get("m_subspaces", 32))
            if d % m_sub != 0:
                m_sub = 16 if d % 16 == 0 else 8
            nbits = int(params.get("nbits", 8))
            k_c = min(256, 1 << max(2, min(8, nbits)))
            idx = IVFPQIndex(dim=d, nlist=max(1, nlist), m_subspaces=m_sub, k_centroids=k_c, seed=42)
            idx.train_and_add(vectors, train_sample_max=20000)
        elif index_id == "hnsw_index":
            if n > 15000:
                raise WorkbenchError(
                    f"Pure-Python `HNSWIndex` is capped at `N <= 15,000` (`got N={n}`) to prevent laptop CPU/RAM lockup.",
                    code="HNSW_GALLERY_CAP_EXCEEDED",
                )
            m_links = int(params.get("m", 16))
            efc = int(params.get("ef_construction", 80))
            idx = HNSWIndex(dim=d, m=m_links, ef_construction=efc, seed=42)
            idx.build(vectors)
        else:
            raise WorkbenchError(f"Unknown retrieval index_id='{index_id}'.", code="UNKNOWN_INDEX")
        build_ms = (time.perf_counter() - t0) * 1000.0
        return idx, build_ms

    @staticmethod
    def _query_index(idx: Any, index_id: str, queries: np.ndarray, top_k: int, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        if index_id == "exact_flat_index":
            scores, ids = idx.search(queries, top_k=top_k)
        elif index_id in {"ivf_flat_index", "ivf_pq_index"}:
            nprobe = int(params.get("nprobe", 16))
            scores, ids = idx.search(queries, top_k=top_k, nprobe=nprobe)
        elif index_id == "hnsw_index":
            efs = int(params.get("ef_search", 64))
            scores, ids = idx.search(queries, top_k=top_k, ef_search=efs)
        else:
            raise WorkbenchError(f"Unknown index_id='{index_id}'.", code="UNKNOWN_INDEX")
        q_ms = (time.perf_counter() - t0) * 1000.0
        return scores, ids, q_ms

    def run_identification_search(
        self,
        probe_image_ref: str,
        probe_face_index: int = 0,
        gallery_source: str = "nasa_p11_verification",
        gallery_limit: int = 120,
        custom_refs: list[str] | None = None,
        detector_id: str = "yunet_2023mar",
        recognizer_id: str = "adaface_ir50_webface4m",
        recognizer_params: dict[str, Any] | None = None,
        index_id: str = "exact_flat_index",
        index_params: dict[str, Any] | None = None,
        top_k: int = 10,
        threshold: float = 0.2169,
    ) -> dict[str, Any]:
        """Search a probe face against an enrolled research gallery and compare against ExactFlat ranks."""
        rec_p = dict(recognizer_params or {})
        idx_p = dict(index_params or {})

        probe_path = self.datasets.resolve_image_ref(probe_image_ref)
        probe_bgr, probe_stem, _, _ = decode_bgr_image(probe_path)
        dets, _, _ = self._execute_detector_on_bgr(probe_bgr, detector_id, {})
        if not dets:
            raise WorkbenchError(f"Zero faces detected in probe image '{probe_image_ref}'.", code="NO_PROBE_FACE")
        p_idx = max(0, min(int(probe_face_index), len(dets) - 1))
        p_face = dets[p_idx]
        aligner = FaceAligner()
        p_aligned = aligner.align(probe_bgr, p_face.landmarks, source_bbox=p_face.bbox)

        gallery = self._get_or_build_gallery_embeddings(
            gallery_source=gallery_source,
            gallery_limit=gallery_limit,
            detector_id=detector_id,
            recognizer_id=recognizer_id,
            rec_params=rec_p,
            custom_refs=custom_refs,
        )
        p_emb_out, _, _ = self._extract_embeddings_for_crops(
            [p_aligned.aligned_bgr], recognizer_id, {**rec_p, "l2_normalize": True}
        )
        q_vec = p_emb_out.embeddings[0:1]

        gal_vecs = gallery["vectors"]
        eff_k = min(max(1, int(top_k)), gal_vecs.shape[0])

        # Exact reference ranking
        exact_idx = ExactFlatIndex(dim=gal_vecs.shape[1], metric="ip", backend="numpy_cpu")
        exact_idx.add(gal_vecs)
        exact_scores, exact_ids = exact_idx.search(q_vec, top_k=gal_vecs.shape[0])
        exact_rank_of_id = {int(gid): rank + 1 for rank, gid in enumerate(exact_ids[0].tolist())}
        exact_cos_of_id = {int(gid): float(sc) for gid, sc in zip(exact_ids[0].tolist(), exact_scores[0].tolist())}

        # Selected index search
        ann_idx, build_ms = self._build_index(index_id, gal_vecs, idx_p)
        scores, ids, search_ms = self._query_index(ann_idx, index_id, q_vec, eff_k, idx_p)

        candidates: list[dict[str, Any]] = []
        returned_set: set[int] = set()
        for rank_0, (gid_raw, idx_score) in enumerate(zip(ids[0].tolist(), scores[0].tolist())):
            gid = int(gid_raw)
            if gid < 0:
                continue
            returned_set.add(gid)
            entry = gallery["entries"][gid]
            exact_cos = exact_cos_of_id[gid]
            ex_rank = exact_rank_of_id[gid]
            candidates.append(
                {
                    "rank": rank_0 + 1,
                    "exact_rank": ex_rank,
                    "rank_shift": (rank_0 + 1) - ex_rank,
                    "gallery_index": gid,
                    "template_id": entry["template_id"],
                    "anonymized_template_tag": entry["anonymized_template_tag"],
                    "image_ref": entry["image_ref"],
                    "crop_data_url": entry["crop_data_url"],
                    "index_score": round(float(idx_score), 6),
                    "exact_cosine": round(exact_cos, 6),
                    "above_threshold": bool(exact_cos >= float(threshold)),
                }
            )

        missed_exact_top_k = [
            {
                "exact_rank": r + 1,
                "template_id": gallery["entries"][int(exact_ids[0, r])]["template_id"],
                "image_ref": gallery["entries"][int(exact_ids[0, r])]["image_ref"],
                "exact_cosine": round(float(exact_scores[0, r]), 6),
            }
            for r in range(eff_k)
            if int(exact_ids[0, r]) not in returned_set
        ]

        return {
            "probe_face": {
                "image_ref": probe_image_ref,
                "selected_face_index": p_idx,
                "bounding_box": p_face.bbox.to_dict(),
                "crop_data_url": encode_bgr_png_data_url(p_aligned.aligned_bgr),
            },
            "gallery_summary": {
                "gallery_source": gallery_source,
                "gallery_size": int(gal_vecs.shape[0]),
                "embedding_dim": int(gal_vecs.shape[1]),
                "recognizer_meta": gallery["recognizer_meta"],
            },
            "index_summary": {
                "index_id": index_id,
                "params": idx_p,
                "build_time_ms": round(build_ms, 3),
                "search_latency_ms": round(search_ms, 4),
                "memory_bytes": ann_idx.memory_bytes(),
                "memory_mib": round(ann_idx.memory_bytes() / (1024.0 * 1024.0), 3),
            },
            "threshold": float(threshold),
            "candidates": candidates,
            "missed_exact_topk_candidates": missed_exact_top_k,
        }

    def run_retrieval_benchmark(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Controlled 1:N retrieval scaling & Exact-vs-ANN comparison (`10K`, `100K`, `1M`)."""
        index_id = str(config.get("index_id", "ivf_pq_index"))
        index_params = dict(config.get("index_params", {}))
        gallery_size = int(config.get("gallery_size", 10000))
        num_queries = min(max(10, int(config.get("num_queries", 200))), 1000)
        embedding_dim = int(config.get("embedding_dim", 512))
        real_anchor_limit = min(int(config.get("real_anchor_limit", 120)), gallery_size)
        seed = int(config.get("seed", 808))

        if gallery_size > 1_000_000:
            raise WorkbenchError("Controlled laptop benchmark caps `gallery_size <= 1,000,000`.", code="GALLERY_TOO_LARGE")
        if index_id == "hnsw_index" and gallery_size > 15_000:
            raise WorkbenchError(
                "Pure-Python `HNSWIndex` is capped at `gallery_size <= 15,000` (`~142s` build at 10K). Use `ExactFlat`, `IVF-Flat`, or `IVF-PQ` for `100K–1M`.",
                code="HNSW_GALLERY_CAP_EXCEEDED",
            )

        if progress_cb:
            progress_cb(10.0, f"Assembling {gallery_size:,}-vector {embedding_dim}-D gallery...")

        rng = np.random.default_rng(seed)
        real_vecs = np.empty((0, embedding_dim), dtype=np.float32)
        if embedding_dim == 512 and real_anchor_limit > 0:
            try:
                gal = self._get_or_build_gallery_embeddings(
                    gallery_source="lfw_sample",
                    gallery_limit=real_anchor_limit,
                    detector_id="yunet_2023mar",
                    recognizer_id="adaface_ir50_webface4m",
                    rec_params={},
                )
                real_vecs = gal["vectors"]
            except Exception:  # noqa: BLE001
                real_vecs = np.empty((0, embedding_dim), dtype=np.float32)

        n_real = int(real_vecs.shape[0])
        n_syn = max(0, gallery_size - n_real)
        if n_syn > 0:
            # Generate normalized unit-hypersphere distractors in chunks to minimize peak RAM
            syn = np.empty((n_syn, embedding_dim), dtype=np.float32)
            chunk = 100_000
            for s in range(0, n_syn, chunk):
                if cancel_check and cancel_check():
                    raise WorkbenchError("Retrieval benchmark cancelled.", code="JOB_CANCELLED")
                e = min(n_syn, s + chunk)
                blk = rng.standard_normal((e - s, embedding_dim), dtype=np.float32)
                norms = np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-12)
                syn[s:e] = blk / norms
            gallery_mat = np.vstack([real_vecs, syn]) if n_real > 0 else syn
        else:
            gallery_mat = real_vecs[:gallery_size].copy()

        # Construct noisy queries around sampled gallery targets so true NN is realistic and non-trivial
        q_target_ids = rng.choice(gallery_mat.shape[0], size=min(num_queries, gallery_mat.shape[0]), replace=False)
        noise = rng.standard_normal((len(q_target_ids), embedding_dim), dtype=np.float32) * 0.025
        queries = l2_normalize_embeddings(gallery_mat[q_target_ids] + noise)

        if progress_cb:
            progress_cb(35.0, "Computing ExactFlat ground-truth nearest neighbors...")

        exact_backend = "torch_cuda_fp16" if (gallery_size >= 100_000 and self.models.get_hardware_snapshot()["gpu"]["available"]) else "numpy_cpu"
        exact_idx = ExactFlatIndex(dim=embedding_dim, metric="ip", backend=exact_backend)
        t_ex_b0 = time.perf_counter()
        exact_idx.add(gallery_mat)
        exact_build_ms = (time.perf_counter() - t_ex_b0) * 1000.0

        t_ex_q0 = time.perf_counter()
        exact_scores, exact_ids = exact_idx.search(queries, top_k=10)
        exact_batch_ms = (time.perf_counter() - t_ex_q0) * 1000.0

        if cancel_check and cancel_check():
            raise WorkbenchError("Retrieval benchmark cancelled.", code="JOB_CANCELLED")
        if progress_cb:
            progress_cb(60.0, f"Building `{index_id}` on {gallery_size:,} vectors...")

        ann_idx, ann_build_ms = self._build_index(index_id, gallery_mat, index_params)

        if progress_cb:
            progress_cb(82.0, f"Measuring single-query and batched latency on {len(queries)} queries...")

        single_latencies: list[float] = []
        n_single_probe = min(len(queries), 40)
        for qi in range(n_single_probe):
            _, _, l_ms = self._query_index(ann_idx, index_id, queries[qi : qi + 1], 10, index_params)
            single_latencies.append(l_ms)

        ann_scores, ann_ids, batched_ms = self._query_index(ann_idx, index_id, queries, 10, index_params)
        recall_stats = compute_recall_at_k(exact_ids[:, 0], ann_ids, k_values=(1, 5, 10))

        sample_comparisons: list[dict[str, Any]] = []
        for qi in range(min(15, len(queries))):
            gt_id = int(exact_ids[qi, 0])
            pred_row = ann_ids[qi].tolist()
            found_pos = pred_row.index(gt_id) + 1 if gt_id in pred_row else None
            sample_comparisons.append(
                {
                    "query_index": qi,
                    "exact_top1_id": gt_id,
                    "exact_top1_cosine": round(float(exact_scores[qi, 0]), 5),
                    "ann_top1_id": int(ann_ids[qi, 0]),
                    "ann_top1_score": round(float(ann_scores[qi, 0]), 5),
                    "exact_top1_rank_in_ann": found_pos,
                    "rank_change": (found_pos - 1) if found_pos is not None else None,
                    "missed_in_top10": found_pos is None,
                }
            )

        mem_bytes = ann_idx.memory_bytes()
        fp32_raw_bytes = int(gallery_size * embedding_dim * 4)
        code_only_b = ann_idx.code_bytes_only() if hasattr(ann_idx, "code_bytes_only") else mem_bytes

        if progress_cb:
            progress_cb(100.0, "Retrieval benchmark complete.")

        return {
            "index_id": index_id,
            "index_params": index_params,
            "gallery_size": gallery_size,
            "real_embeddings_count": n_real,
            "synthetic_distractors_count": n_syn,
            "query_count": int(len(queries)),
            "embedding_dim": embedding_dim,
            "synthetic_disclaimer": (
                "Contains unit-hypersphere synthetic distractors for memory/latency/ANN recall scaling ONLY — "
                "do NOT interpret as biometric recognition accuracy."
                if n_syn > 0
                else "100% real extracted face embeddings."
            ),
            "metrics": {
                "build_time_ms": round(ann_build_ms, 2),
                "exact_reference_build_ms": round(exact_build_ms, 2),
                "exact_reference_batch_ms": round(exact_batch_ms, 2),
                "index_memory_bytes": mem_bytes,
                "index_memory_mib": round(mem_bytes / (1024.0 * 1024.0), 3),
                "code_compression_vs_fp32": round(fp32_raw_bytes / max(code_only_b, 1), 2),
                "single_query_p50_ms": round(float(np.percentile(single_latencies, 50)), 4),
                "single_query_p95_ms": round(float(np.percentile(single_latencies, 95)), 4),
                "batched_qps": round((len(queries) * 1000.0) / max(batched_ms, 1e-4), 1),
                "recall_at_1": recall_stats["recall_at_1"],
                "recall_at_5": recall_stats["recall_at_5"],
                "recall_at_10": recall_stats["recall_at_10"],
                "rank_distribution": recall_stats["rank_distribution"],
            },
            "exact_vs_ann_query_samples": sample_comparisons,
            "hardware_snapshot": self.models.get_hardware_snapshot(),
        }

    # ------------------------------------------------------------------
    # STAGE 6: END-TO-END PIPELINE BUILDER & DATASET VERIFICATION RUNS
    # ------------------------------------------------------------------
    def run_pipeline_experiment(
        self,
        pipeline_cfg: dict[str, Any],
        progress_cb: Callable[[float, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Validate and execute a configurable multi-stage pipeline over images or verification pairs."""
        val = validate_pipeline_configuration(pipeline_cfg, self.registry)
        if not val.valid:
            raise WorkbenchError(
                "Pipeline validation failed: " + "; ".join(val.errors),
                code="INVALID_PIPELINE_CONFIG",
                details=val.to_dict(),
            )

        dataset_id = str(pipeline_cfg.get("dataset_id", "nasa_p11_multiscale"))
        pair_limit = min(max(10, int(pipeline_cfg.get("pair_limit", 120))), 600)

        det_cfg = pipeline_cfg.get("detection", {})
        det_id = str(det_cfg.get("algorithm_id", "yunet_2023mar"))
        det_params = dict(det_cfg.get("params", {}))

        align_cfg = pipeline_cfg.get("alignment", {})
        align_params = dict(align_cfg.get("params", {}))

        qual_cfg = pipeline_cfg.get("quality", {"enabled": True})
        qual_enabled = bool(qual_cfg.get("enabled", True))
        qual_params = dict(qual_cfg.get("params", {}))

        rec_cfg = pipeline_cfg.get("recognition", {"enabled": True})
        rec_enabled = bool(rec_cfg.get("enabled", True))
        rec_id = str(rec_cfg.get("algorithm_id", "adaface_ir50_webface4m"))
        rec_params = dict(rec_cfg.get("params", {}))

        norm_enabled = bool(pipeline_cfg.get("normalization", {}).get("enabled", True))
        rec_params["l2_normalize"] = norm_enabled

        match_cfg = pipeline_cfg.get("matching", {"metric": "cosine", "threshold": 0.2169})
        metric_name = str(match_cfg.get("metric", "cosine"))
        threshold = float(match_cfg.get("threshold", 0.2169))

        ret_cfg = pipeline_cfg.get("retrieval", {"enabled": False})
        ret_enabled = bool(ret_cfg.get("enabled", False))

        stage_graph: list[dict[str, Any]] = []
        t_pipe0 = time.perf_counter()

        # Branch A: Pairwise Verification Dataset (`nasa_p11_verification` or `lfw_research_split`)
        if dataset_id in {"nasa_p11_verification", "lfw_research_split"} and rec_enabled:
            if dataset_id == "nasa_p11_verification":
                all_pairs = self.datasets.get_dataset_items("nasa_p11_verification", kind="pairs", limit=200)["items"]
                calib_pairs = all_pairs[:60]
                test_pairs = all_pairs
            else:
                half = max(4, pair_limit // 2)
                q_pos = max(1, half // 2)
                q_neg = max(1, half - q_pos)
                all_lfw = self.datasets._get_lfw_pairs()
                calib_pos = [p for p in all_lfw if p["split"] == "calibration_folds_1_5" and p["is_same_subject"]]
                calib_neg = [p for p in all_lfw if p["split"] == "calibration_folds_1_5" and not p["is_same_subject"]]
                test_pos = [p for p in all_lfw if p["split"] == "test_folds_6_10" and p["is_same_subject"]]
                test_neg = [p for p in all_lfw if p["split"] == "test_folds_6_10" and not p["is_same_subject"]]
                calib_pairs = calib_pos[:q_pos] + calib_neg[:q_neg]
                test_pairs = test_pos[:q_pos] + test_neg[:q_neg]

            unique_refs = sorted(
                {p["image_a"] for p in calib_pairs + test_pairs} | {p["image_b"] for p in calib_pairs + test_pairs}
            )
            decode_ms_sum, det_ms_sum, align_ms_sum, qual_ms_sum = 0.0, 0.0, 0.0, 0.0
            crops_list: list[np.ndarray] = []
            ref_to_idx: dict[str, int] = {}
            assessor = FaceQualityAssessor()

            for idx, ref in enumerate(unique_refs):
                if cancel_check and cancel_check():
                    raise WorkbenchError("Pipeline execution cancelled.", code="JOB_CANCELLED")
                if progress_cb and idx % 8 == 0:
                    progress_cb(min(65.0, (idx / max(1, len(unique_refs))) * 65.0), f"Detecting & aligning image {idx+1}/{len(unique_refs)}...")
                bgr, _, _, d_ms = decode_bgr_image(self.datasets.resolve_image_ref(ref))
                decode_ms_sum += d_ms
                dets, prof, _ = self._execute_detector_on_bgr(bgr, det_id, det_params)
                det_ms_sum += float(prof.get("total_detect_ms", 0.0))
                if dets:
                    crop, al_info = self._compute_single_alignment(bgr, dets[0].landmarks.as_array(), align_params)
                    align_ms_sum += al_info["execution_time_ms"]
                    if qual_enabled:
                        q_res = assessor.assess(bgr, dets[0].bbox, aligned_crop_bgr=crop)
                        qual_ms_sum += q_res.execution_time_ms
                else:
                    # Center fallback crop if detector misses extreme frame
                    cw = int(align_params.get("crop_width", 112))
                    ch = int(align_params.get("crop_height", 112))
                    crop = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_LINEAR)
                ref_to_idx[ref] = len(crops_list)
                crops_list.append(crop)

            if progress_cb:
                progress_cb(72.0, f"Extracting `{rec_id}` embeddings across {len(crops_list)} crops...")

            emb_out, prep_warn, rec_meta = self._extract_embeddings_for_crops(crops_list, rec_id, rec_params)
            raw_all = emb_out.raw_embeddings

            def _score_pairs(p_list: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
                ia = [ref_to_idx[p["image_a"]] for p in p_list]
                ib = [ref_to_idx[p["image_b"]] for p in p_list]
                sc = compute_pair_scores(raw_all[ia], raw_all[ib], metric=metric_name)
                lb = np.array([bool(p["is_same_subject"]) for p in p_list], dtype=bool)
                return sc, lb

            t_m0 = time.perf_counter()
            c_scores, c_labels = _score_pairs(calib_pairs)
            t_scores, t_labels = _score_pairs(test_pairs)
            ver_eval = evaluate_calibrated_verification(c_scores, c_labels, t_scores, t_labels)
            match_ms = (time.perf_counter() - t_m0) * 1000.0

            # Build compact 40-point ROC / DET / FAR-vs-threshold curves for plotting
            grid = np.linspace(float(t_scores.min()) - 1e-5, float(t_scores.max()) + 1e-5, 45)
            gen_s = t_scores[t_labels]
            imp_s = t_scores[~t_labels]
            roc_pts = []
            for th_v in grid:
                far_v = float(np.mean(imp_s >= th_v)) if imp_s.size else 0.0
                tar_v = float(np.mean(gen_s >= th_v)) if gen_s.size else 0.0
                roc_pts.append(
                    {
                        "threshold": round(float(th_v), 5),
                        "far": round(far_v, 5),
                        "tar": round(tar_v, 5),
                        "frr": round(1.0 - tar_v, 5),
                    }
                )

            ret_summary = None
            if ret_enabled:
                r_idx_id = str(ret_cfg.get("algorithm_id", "exact_flat_index"))
                r_idx, r_b_ms = self._build_index(r_idx_id, emb_out.embeddings, dict(ret_cfg.get("params", {})))
                _, _, r_q_ms = self._query_index(r_idx, r_idx_id, emb_out.embeddings[: min(20, len(crops_list))], 5, dict(ret_cfg.get("params", {})))
                ret_summary = {"index_id": r_idx_id, "build_ms": round(r_b_ms, 2), "query_ms": round(r_q_ms, 3)}

            total_ms = (time.perf_counter() - t_pipe0) * 1000.0
            stage_graph = [
                {"stage_id": "image_input", "status": "completed", "time_ms": round(decode_ms_sum, 2), "input_shape": f"{len(unique_refs)} files", "output_shape": "BGR uint8 HWC", "face_count": len(unique_refs)},
                {"stage_id": "face_detection", "status": "completed", "time_ms": round(det_ms_sum, 2), "algorithm_id": det_id, "input_shape": "BGR HWC", "output_shape": f"{len(crops_list)} boxes+5pt", "face_count": len(crops_list)},
                {"stage_id": "alignment", "status": "completed", "time_ms": round(align_ms_sum, 2), "algorithm_id": "five_point_canonical_aligner", "input_shape": "5x2 landmarks", "output_shape": f"[{len(crops_list)}, 112, 112, 3]", "face_count": len(crops_list)},
                {"stage_id": "quality_assessment", "status": "completed" if qual_enabled else "skipped", "time_ms": round(qual_ms_sum, 2), "input_shape": "[N, 112, 112, 3]", "output_shape": "QualityAssessment", "face_count": len(crops_list)},
                {"stage_id": "face_recognition", "status": "completed", "time_ms": round(emb_out.preprocess_time_ms + emb_out.inference_time_ms, 2), "algorithm_id": rec_id, "input_shape": f"[{len(crops_list)}, 3, 112, 112]", "output_shape": f"[{len(crops_list)}, {emb_out.embedding_dim}]", "face_count": len(crops_list)},
                {"stage_id": "embedding_normalization", "status": "completed" if norm_enabled else "skipped", "time_ms": round(emb_out.postprocess_time_ms, 3), "input_shape": f"[{len(crops_list)}, {emb_out.embedding_dim}]", "output_shape": f"unit S^{emb_out.embedding_dim-1}" if norm_enabled else "raw unnormalized", "face_count": len(crops_list)},
                {"stage_id": "matching_retrieval", "status": "completed", "time_ms": round(match_ms, 2), "input_shape": f"{len(test_pairs)} pairs", "output_shape": metric_name, "details": ret_summary},
                {"stage_id": "evaluation", "status": "completed", "time_ms": 1.0, "output_shape": f"ROC AUC={ver_eval['test_roc_auc']}"},
            ]

            manual_acc = float(np.mean((t_scores >= threshold) == t_labels)) * 100.0
            emp_op_test = compute_empirical_operating_point(gen_s, imp_s, threshold, higher_is_more_similar=True)
            c_gen_s = c_scores[c_labels]
            c_imp_s = c_scores[~c_labels]

            def _pack_partition(
                part_id: str,
                part_label: str,
                p_items: list[dict[str, Any]],
                s_vec: np.ndarray,
                l_vec: np.ndarray,
                *,
                is_calibration: bool,
                is_held_out_test: bool,
                is_smoke_test: bool,
                identity_note: str,
                default_tau: float,
                tau_origin: str,
            ) -> dict[str, Any]:
                g_list = [round(float(v), 6) for v in s_vec[l_vec].tolist()]
                i_list = [round(float(v), 6) for v in s_vec[~l_vec].tolist()]
                pair_refs = [
                    {
                        "pair_index": idx_p,
                        "pair_id": str(p.get("pair_id", f"{part_id}_{idx_p:04d}")),
                        "is_same_subject": bool(lb),
                        "image_a": str(p["image_a"]),
                        "image_b": str(p["image_b"]),
                    }
                    for idx_p, (p, lb) in enumerate(zip(p_items, l_vec.tolist()))
                ]
                pair_scores_aligned = [round(float(v), 6) for v in s_vec.tolist()]
                return {
                    "partition_id": part_id,
                    "split_name": part_label,
                    "is_calibration": is_calibration,
                    "is_held_out_test": is_held_out_test,
                    "is_smoke_test": is_smoke_test,
                    "identity_partition_info": identity_note,
                    "metric": metric_name,
                    "higher_is_more_similar": True,
                    "decision_rule": "score >= threshold -> MATCH; score < threshold -> NON-MATCH",
                    "default_threshold": round(float(default_tau), 6),
                    "default_threshold_origin": tau_origin,
                    "genuine_count": len(g_list),
                    "impostor_count": len(i_list),
                    "total_pairs": len(pair_scores_aligned),
                    "genuine_scores": g_list,
                    "impostor_scores": i_list,
                    "pair_scores": pair_scores_aligned,
                    "pair_refs": pair_refs,
                }

            calib_op_1e3 = ver_eval["calibrated_operating_points"]["target_far_0.001"]["calibrated_threshold"]
            calib_op_1e2 = ver_eval["calibrated_operating_points"]["target_far_0.01"]["calibrated_threshold"]
            calib_acc_tau = ver_eval["calibrated_acc_threshold"]

            ds_meta_obj = next((d for d in self.datasets.list_datasets() if d.get("dataset_id") == dataset_id), {})
            if dataset_id == "lfw_research_split":
                dataset_split_desc = (
                    f"Identity-Disjoint Stratified LFW: Calib Folds 1–5 ({len(calib_pairs)} pairs, 2,086 split IDs) "
                    f"-> Held-Out Test Folds 6–10 ({len(test_pairs)} pairs, 2,195 split IDs, 0 subject overlap)"
                )
                partitions_payload = {
                    "held_out_test": _pack_partition(
                        "held_out_test",
                        f"LFW Held-Out Test Folds 6–10 ({len(test_pairs)} Stratified Pairs · 2,195 Split Subjects · 0 Calib Overlap)",
                        test_pairs,
                        t_scores,
                        t_labels,
                        is_calibration=False,
                        is_held_out_test=True,
                        is_smoke_test=False,
                        identity_note="Held-Out Test Partition (Folds 6–10: 2,195 subjects in full LFW split; 0 shared identities with Calibration Folds 1–5).",
                        default_tau=calib_acc_tau,
                        tau_origin="Fixed pre-established threshold calibrated exclusively on Calibration Folds 1–5",
                    ),
                    "calibration": _pack_partition(
                        "calibration",
                        f"LFW Calibration Folds 1–5 ({len(calib_pairs)} Stratified Pairs · 2,086 Split Subjects — Calibration Only)",
                        calib_pairs,
                        c_scores,
                        c_labels,
                        is_calibration=True,
                        is_held_out_test=False,
                        is_smoke_test=False,
                        identity_note="Calibration Partition ONLY (Folds 1–5: 2,086 subjects; used to select operating thresholds before held-out evaluation).",
                        default_tau=calib_acc_tau,
                        tau_origin="Empirical best-accuracy threshold on Calibration Folds 1–5",
                    ),
                }
                default_partition_key = "held_out_test"
                eval_protocol_label = "Identity-Disjoint 5-Fold Calibration (Folds 1–5) -> 5-Fold Held-Out Test (Folds 6–10)"
            else:
                dataset_split_desc = f"NASA P1.1 120-Pair Integration Smoke Test ({len(test_pairs)} pairs — Non-Benchmark)"
                partitions_payload = {
                    "smoke_test": _pack_partition(
                        "smoke_test",
                        f"NASA P1.1 Integration Smoke Split ({len(test_pairs)} Pairs — Non-Benchmark Smoke Verification)",
                        test_pairs,
                        t_scores,
                        t_labels,
                        is_calibration=False,
                        is_held_out_test=False,
                        is_smoke_test=True,
                        identity_note="Small public-domain astronaut integration smoke set (16 images / 120 pairs) — NOT a general biometric benchmark.",
                        default_tau=threshold,
                        tau_origin="Configured pipeline decision threshold",
                    )
                }
                default_partition_key = "smoke_test"
                eval_protocol_label = "Integration Smoke Verification (Non-Benchmark)"

            score_distributions_artifact = {
                "schema_version": "wb3.v1",
                "dataset_id": dataset_id,
                "dataset_name": ds_meta_obj.get("name", dataset_id),
                "dataset_manifest_sha256": ds_meta_obj.get("manifest_sha256"),
                "evaluation_protocol": eval_protocol_label,
                "detector_id": det_id,
                "detector_sha256": self.registry[det_id].model_sha256 if det_id in self.registry else None,
                "recognizer_id": rec_id,
                "recognizer_sha256": rec_meta.get("model_sha256"),
                "metric": metric_name,
                "higher_is_more_similar": True,
                "decision_rule": "score >= threshold -> MATCH; score < threshold -> NON-MATCH",
                "configured_threshold": round(float(threshold), 6),
                "pre_established_thresholds": {
                    "configured_manual": round(float(threshold), 6),
                    "calibrated_best_acc": calib_acc_tau,
                    "calibrated_far_1e3": calib_op_1e3,
                    "calibrated_far_1e2": calib_op_1e2,
                },
                "default_partition": default_partition_key,
                "partitions": partitions_payload,
            }

            metrics_out = {
                "num_unique_images": len(unique_refs),
                "num_test_pairs": len(test_pairs),
                "genuine_pairs": emp_op_test["genuine_count"],
                "impostor_pairs": emp_op_test["impostor_count"],
                "decision_threshold": emp_op_test["decision_threshold"],
                "tp_at_manual_thresh": emp_op_test["tp"],
                "fn_at_manual_thresh": emp_op_test["fn"],
                "fp_at_manual_thresh": emp_op_test["fp"],
                "tn_at_manual_thresh": emp_op_test["tn"],
                "tar_at_manual_thresh": emp_op_test["tar"],
                "far_at_manual_thresh": emp_op_test["far"],
                "frr_at_manual_thresh": emp_op_test["frr"],
                "tnr_at_manual_thresh": emp_op_test["tnr"],
                "calibrated_acc_threshold": calib_acc_tau,
                "calibrated_threshold_far_1e3": calib_op_1e3,
                "test_roc_auc": ver_eval["test_roc_auc"],
                "test_eer_pct": ver_eval["test_eer_pct"],
                "test_accuracy_at_calib_thresh_pct": ver_eval["test_accuracy_at_calib_thresh_pct"],
                "test_accuracy_at_manual_thresh_pct": round(manual_acc, 4),
                "test_tar_at_calib_far_1e3": ver_eval["calibrated_operating_points"]["target_far_0.001"]["test_realized_tar"],
                "test_realized_far_at_calib_far_1e3": ver_eval["calibrated_operating_points"]["target_far_0.001"]["test_realized_far"],
                "test_realized_far_at_calib_1e3": ver_eval["calibrated_operating_points"]["target_far_0.001"]["test_realized_far"],
                "total_pipeline_ms": round(total_ms, 2),
                "faces_per_sec": round(len(crops_list) * 1000.0 / max(total_ms, 1e-3), 2),
            }
            if progress_cb:
                progress_cb(100.0, "Verification pipeline run complete.")
            return {
                "validation": val.to_dict(),
                "dataset_id": dataset_id,
                "dataset_split": dataset_split_desc,
                "recognizer_meta": rec_meta,
                "preprocessing_warning": prep_warn,
                "metrics": metrics_out,
                "curves": {
                    "roc_det_sweep": roc_pts,
                    "score_distributions": score_distributions_artifact,
                    "empirical_operating_point": emp_op_test,
                },
                "stage_graph": stage_graph,
                "hardware_snapshot": self.models.get_hardware_snapshot(),
            }

        # Branch B: Detection + Multi-Face Pipeline Dataset (`nasa_p11_multiscale` / `nasa_p1_detection` / `nasa_dev_samples`)
        ds_items = self.datasets.get_dataset_items(dataset_id, kind="images", limit=100)["items"]
        decode_ms_sum, det_ms_sum, align_ms_sum, qual_ms_sum = 0.0, 0.0, 0.0, 0.0
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        gt_any = False
        all_crops: list[np.ndarray] = []
        assessor = FaceQualityAssessor()

        for idx, it in enumerate(ds_items):
            if cancel_check and cancel_check():
                raise WorkbenchError("Pipeline execution cancelled.", code="JOB_CANCELLED")
            if progress_cb:
                progress_cb((idx / max(1, len(ds_items))) * 70.0, f"Processing image {idx+1}/{len(ds_items)}...")
            p = self.datasets.resolve_image_ref(it["image_ref"])
            bgr, _, _, d_ms = decode_bgr_image(p)
            decode_ms_sum += d_ms
            dets, prof, _ = self._execute_detector_on_bgr(bgr, det_id, det_params)
            det_ms_sum += float(prof.get("total_detect_ms", 0.0))

            gt_info = self.datasets.get_ground_truth_for_path(p)
            if gt_info:
                gt_any = True
                ev = evaluate_detections_with_boxes(dets, gt_info["boxes"], 0.50)
                tot_tp += ev["tp"]
                tot_fp += ev["fp"]
                tot_fn += ev["fn"]

            for det in dets:
                crop, al_info = self._compute_single_alignment(bgr, det.landmarks.as_array(), align_params)
                align_ms_sum += al_info["execution_time_ms"]
                if qual_enabled:
                    q_r = assessor.assess(bgr, det.bbox, aligned_crop_bgr=crop)
                    qual_ms_sum += q_r.execution_time_ms
                all_crops.append(crop)

        rec_status = "skipped"
        rec_ms = 0.0
        norm_ms = 0.0
        emb_dim_out: int | None = None
        rec_meta_out: dict[str, Any] = {}
        if rec_enabled and all_crops:
            if self.registry[rec_id].availability_status != "AVAILABLE":
                rec_status = "BLOCKED_MISSING_REFERENCE_RESNET50_WEIGHTS"
            else:
                emb_out, _, rec_meta_out = self._extract_embeddings_for_crops(all_crops, rec_id, rec_params)
                rec_status = "completed"
                rec_ms = emb_out.preprocess_time_ms + emb_out.inference_time_ms
                norm_ms = emb_out.postprocess_time_ms
                emb_dim_out = emb_out.embedding_dim

        total_ms = (time.perf_counter() - t_pipe0) * 1000.0
        prec = (tot_tp / (tot_tp + tot_fp)) if (gt_any and (tot_tp + tot_fp) > 0) else None
        rec_v = (tot_tp / (tot_tp + tot_fn)) if (gt_any and (tot_tp + tot_fn) > 0) else None
        f1 = (2.0 * prec * rec_v / (prec + rec_v)) if (prec is not None and rec_v is not None and (prec + rec_v) > 0) else None

        stage_graph = [
            {"stage_id": "image_input", "status": "completed", "time_ms": round(decode_ms_sum, 2), "input_shape": f"{len(ds_items)} images", "output_shape": "BGR HWC", "face_count": len(ds_items)},
            {"stage_id": "face_detection", "status": "completed", "time_ms": round(det_ms_sum, 2), "algorithm_id": det_id, "input_shape": "BGR HWC", "output_shape": f"{len(all_crops)} faces", "face_count": len(all_crops)},
            {"stage_id": "alignment", "status": "completed", "time_ms": round(align_ms_sum, 2), "algorithm_id": "five_point_canonical_aligner", "input_shape": "5x2 landmarks", "output_shape": f"[{len(all_crops)}, 112, 112, 3]", "face_count": len(all_crops)},
            {"stage_id": "quality_assessment", "status": "completed" if qual_enabled else "skipped", "time_ms": round(qual_ms_sum, 2), "input_shape": "crops", "output_shape": "QualityAssessment", "face_count": len(all_crops)},
            {"stage_id": "face_recognition", "status": rec_status, "time_ms": round(rec_ms, 2), "algorithm_id": rec_id if rec_enabled else None, "input_shape": f"[{len(all_crops)}, 3, 112, 112]", "output_shape": f"[{len(all_crops)}, {emb_dim_out}]" if emb_dim_out else rec_status, "face_count": len(all_crops)},
            {"stage_id": "embedding_normalization", "status": "completed" if (rec_status == "completed" and norm_enabled) else "skipped", "time_ms": round(norm_ms, 3), "output_shape": "unit L2" if norm_enabled else "raw", "face_count": len(all_crops)},
            {"stage_id": "evaluation", "status": "completed", "time_ms": 0.5, "output_shape": f"F1={round(f1, 4) if f1 is not None else 'N/A'}"},
        ]

        if progress_cb:
            progress_cb(100.0, "Pipeline run complete.")

        return {
            "validation": val.to_dict(),
            "dataset_id": dataset_id,
            "recognizer_meta": rec_meta_out,
            "metrics": {
                "num_images": len(ds_items),
                "detected_faces": len(all_crops),
                "precision": round(prec, 4) if prec is not None else None,
                "recall": round(rec_v, 4) if rec_v is not None else None,
                "f1": round(f1, 4) if f1 is not None else None,
                "detection_ms": round(det_ms_sum, 2),
                "recognition_ms": round(rec_ms, 2),
                "total_pipeline_ms": round(total_ms, 2),
                "images_per_sec": round(len(ds_items) * 1000.0 / max(total_ms, 1e-3), 2),
                "faces_per_sec": round(len(all_crops) * 1000.0 / max(total_ms, 1e-3), 2),
            },
            "curves": {},
            "stage_graph": stage_graph,
            "hardware_snapshot": self.models.get_hardware_snapshot(),
        }
