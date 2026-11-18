"""Milestone P3 — Controlled Algorithmic Modifications across Detection, Alignment, Quality & Matching.

Every variant isolates a single algorithmic parameter or mathematical formulation so controlled
comparisons can be executed against the preserved P1/P1.1/P2 baselines without modifying any
historical source files.
"""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from src.alignment.face_aligner import _estimate_similarity_2d_umeyama
from src.common.types import (
    CANONICAL_LANDMARK_NAMES,
    BoundingBox,
    DetectedFace,
    DetectionBatchResult,
    FacialLandmarks,
)
from src.detection.base_detector import validate_bgr_image
from src.detection.scrfd_detector import SCRFDDetector, _distance2bbox, _distance2kps, _greedy_nms
from src.recognition.base_embedder import EmbeddingBatchOutput, FaceEmbedder, l2_normalize_embeddings


INTERPOLATION_FLAGS: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}

ARCFACE_5PT_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def gaussian_soft_nms(
    dets_xyxys: np.ndarray,
    sigma: float = 0.50,
    score_thresh: float = 0.30,
) -> tuple[list[int], np.ndarray]:
    """Gaussian Soft-NMS (Bodla et al., ICCV 2017): decays overlapping scores smoothly via
    `s_i <- s_i * exp(-IoU(M, b_i)^2 / sigma)` instead of hard step suppression.
    """
    n = dets_xyxys.shape[0]
    if n == 0:
        return [], np.empty((0,), dtype=np.float32)

    boxes = dets_xyxys[:, :4].astype(np.float64).copy()
    scores = dets_xyxys[:, 4].astype(np.float64).copy()
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0] + 1.0) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1] + 1.0
    )
    indices = np.arange(n, dtype=np.int64)

    keep: list[int] = []
    out_scores: list[float] = []

    for i in range(n):
        max_pos = i + int(np.argmax(scores[i:]))
        if scores[max_pos] < score_thresh:
            break

        # Swap max_pos into slot i
        boxes[[i, max_pos]] = boxes[[max_pos, i]]
        scores[[i, max_pos]] = scores[[max_pos, i]]
        areas[[i, max_pos]] = areas[[max_pos, i]]
        indices[[i, max_pos]] = indices[[max_pos, i]]

        keep.append(int(indices[i]))
        out_scores.append(float(scores[i]))

        if i == n - 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[i + 1 :, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[i + 1 :, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i + 1 :, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[i + 1 :, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[i + 1 :] - inter, 1e-9)
        scores[i + 1 :] *= np.exp(-(iou * iou) / max(sigma, 1e-6))

    return keep, np.asarray(out_scores, dtype=np.float32)


class ResearchSCRFDWrapper:
    """Non-invasive research wrapper over `SCRFDDetector` isolating canvas geometry,
    coordinate inverse-scaling (`P2-AUD-08`), confidence/NMS thresholds, and Soft-NMS.
    """

    def __init__(self, base_scrfd: SCRFDDetector) -> None:
        self.base = base_scrfd

    def detect_variant(
        self,
        image_bgr: np.ndarray,
        max_dim: int = 1280,
        canvas_mode: str = "square_padded",
        inverse_scale_mode: str = "single_scalar",
        score_threshold: float = 0.50,
        nms_threshold: float = 0.40,
        nms_mode: str = "hard_greedy",
        soft_nms_sigma: float = 0.50,
        allow_upscale: bool = False,
    ) -> tuple[DetectionBatchResult, dict[str, Any]]:
        orig_w, orig_h = validate_bgr_image(image_bgr)
        if canvas_mode not in {"square_padded", "stride32_tight"}:
            raise ValueError(f"Unsupported canvas_mode='{canvas_mode}'")
        if inverse_scale_mode not in {"single_scalar", "exact_dual_xy"}:
            raise ValueError(f"Unsupported inverse_scale_mode='{inverse_scale_mode}'")
        if nms_mode not in {"hard_greedy", "gaussian_soft_nms"}:
            raise ValueError(f"Unsupported nms_mode='{nms_mode}'")

        t_total_0 = time.perf_counter()
        t_res_0 = time.perf_counter()

        longest = max(orig_w, orig_h)
        if not allow_upscale and longest <= max_dim:
            new_w, new_h = orig_w, orig_h
            resized = image_bgr
        else:
            scale = float(max_dim) / float(longest)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)

        if canvas_mode == "square_padded":
            canvas_w = int(math.ceil(max_dim / 32.0) * 32)
            canvas_h = canvas_w
        else:
            canvas_w = max(32, int(math.ceil(new_w / 32.0) * 32))
            canvas_h = max(32, int(math.ceil(new_h / 32.0) * 32))

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized
        resize_ms = (time.perf_counter() - t_res_0) * 1000.0

        t_pre_0 = time.perf_counter()
        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / self.base._input_std,
            (canvas_w, canvas_h),
            (self.base._input_mean, self.base._input_mean, self.base._input_mean),
            swapRB=(self.base.color_order == "RGB"),
        )
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        preprocess_ms = (time.perf_counter() - t_pre_0) * 1000.0

        t_inf_0 = time.perf_counter()
        net_outs = self.base._run_raw_blob(blob)
        inference_ms = (time.perf_counter() - t_inf_0) * 1000.0

        t_post_0 = time.perf_counter()
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        for idx, stride in enumerate(self.base._feat_strides):
            if self.base._batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.base._fmc][0] * stride
                kps_preds = net_outs[idx + self.base._fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.base._fmc] * stride
                kps_preds = net_outs[idx + self.base._fmc * 2] * stride

            scores_1d = scores.reshape(-1)
            pos_inds = np.where(scores_1d >= score_threshold)[0]
            if pos_inds.size == 0:
                continue

            feat_h = canvas_h // stride
            feat_w = canvas_w // stride
            centers = self.base._get_anchor_centers(feat_h, feat_w, stride)
            pos_centers = centers[pos_inds]

            scores_list.append(scores_1d[pos_inds])
            bboxes_list.append(_distance2bbox(pos_centers, bbox_preds[pos_inds]))
            kpss_list.append(_distance2kps(pos_centers, kps_preds[pos_inds]).reshape((-1, 5, 2)))

        detections: list[DetectedFace] = []
        if inverse_scale_mode == "single_scalar":
            scale_x = float(new_h) / float(orig_h)
            scale_y = scale_x
        else:
            scale_x = float(new_w) / float(orig_w)
            scale_y = float(new_h) / float(orig_h)

        if scores_list:
            all_scores = np.concatenate(scores_list, axis=0)
            all_bboxes = np.concatenate(bboxes_list, axis=0)
            all_kpss = np.concatenate(kpss_list, axis=0)

            all_bboxes[:, [0, 2]] /= scale_x
            all_bboxes[:, [1, 3]] /= scale_y
            all_kpss[:, :, 0] /= scale_x
            all_kpss[:, :, 1] /= scale_y

            order = np.argsort(-all_scores, kind="stable")
            if self.base.top_k > 0 and order.size > self.base.top_k:
                order = order[: self.base.top_k]

            pre_det = np.column_stack([all_bboxes[order], all_scores[order]]).astype(np.float32, copy=False)
            ordered_kpss = all_kpss[order]

            if nms_mode == "hard_greedy":
                keep = _greedy_nms(pre_det, nms_threshold)
                kept_det = pre_det[keep]
                kept_scores = kept_det[:, 4]
                kept_kpss = ordered_kpss[keep]
            else:
                keep, decayed_scores = gaussian_soft_nms(
                    pre_det, sigma=soft_nms_sigma, score_thresh=score_threshold
                )
                kept_det = pre_det[keep].copy()
                kept_det[:, 4] = decayed_scores
                kept_scores = decayed_scores
                kept_kpss = ordered_kpss[keep]

            for row, sc_val, kp5 in zip(kept_det, kept_scores, kept_kpss):
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
                        confidence=float(sc_val),
                    )
                )

        post_ms = (time.perf_counter() - t_post_0) * 1000.0
        total_ms = (time.perf_counter() - t_total_0) * 1000.0
        res = DetectionBatchResult(
            detections=detections,
            execution_time_ms=total_ms,
            original_size_wh=(orig_w, orig_h),
            detector_input_size_wh=(canvas_w, canvas_h),
            scale_xy=(1.0 / scale_x, 1.0 / scale_y),
            model_id=self.base.model_id,
            model_version=self.base.model_version,
        )
        prof = {
            "resize_and_pad_ms": resize_ms,
            "preprocess_blob_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": post_ms,
            "total_detect_ms": total_ms,
            "canvas_wh": (canvas_w, canvas_h),
            "active_wh": (new_w, new_h),
            "padded_area_ratio": round(float(canvas_w * canvas_h) / float(max(1, new_w * new_h)), 4),
        }
        return res, prof


def estimate_affine_least_squares_6dof(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Full 6-DoF 2D affine least-squares warp (`[x, y, 1] @ M^T = [u, v]`) across 5 landmarks."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    ones = np.ones((src.shape[0], 1), dtype=np.float64)
    design = np.hstack([src, ones])  # (5, 3)
    sol, _, _, _ = np.linalg.lstsq(design, dst, rcond=None)  # (3, 2)
    return sol.T.astype(np.float64)  # (2, 3)


def align_face_research(
    image_bgr: np.ndarray,
    landmarks_5x2: np.ndarray,
    transform_mode: str = "similarity_umeyama_4dof",
    interpolation: str = "linear",
    output_size_wh: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Align a BGR image using configurable geometric transformation and interpolation kernel."""
    if interpolation not in INTERPOLATION_FLAGS:
        raise ValueError(f"Unsupported interpolation='{interpolation}'")
    pts = np.asarray(landmarks_5x2, dtype=np.float64)
    out_w, out_h = int(output_size_wh[0]), int(output_size_wh[1])
    scale_vec = np.array([out_w / 112.0, out_h / 112.0], dtype=np.float64)
    dst = ARCFACE_5PT_112 * scale_vec

    if transform_mode == "similarity_umeyama_4dof":
        mat = _estimate_similarity_2d_umeyama(pts, dst)
    elif transform_mode == "affine_least_squares_6dof":
        mat = estimate_affine_least_squares_6dof(pts, dst)
    elif transform_mode == "similarity_zoom_out_10pct":
        center = np.array([out_w * 0.5, out_h * 0.5], dtype=np.float64)
        dst_zoomed = center + 0.90 * (dst - center)
        mat = _estimate_similarity_2d_umeyama(pts, dst_zoomed)
    else:
        raise ValueError(f"Unsupported transform_mode='{transform_mode}'")

    return cv2.warpAffine(
        image_bgr,
        mat,
        (out_w, out_h),
        flags=INTERPOLATION_FLAGS[interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def perturb_landmarks_gaussian(
    landmarks_5x2: np.ndarray,
    sigma_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pts = np.asarray(landmarks_5x2, dtype=np.float64)
    if sigma_px <= 0.0:
        return pts.copy()
    return pts + rng.normal(loc=0.0, scale=float(sigma_px), size=pts.shape)


def perturb_crop_roll(crop_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply controlled in-plane roll mis-alignment of `angle_deg` around crop center."""
    if abs(angle_deg) < 1e-6:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), float(angle_deg), 1.0)
    return cv2.warpAffine(
        crop_bgr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )


def apply_gaussian_blur(crop_bgr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return crop_bgr
    ksize = int(2 * round(3.0 * sigma) + 1)
    ksize = max(3, ksize | 1)
    return cv2.GaussianBlur(crop_bgr, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))


def apply_downscale_upscale(crop_bgr: np.ndarray, eff_res: int) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if eff_res >= min(h, w):
        return crop_bgr
    small = cv2.resize(crop_bgr, (eff_res, eff_res), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_gamma(crop_bgr: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-6:
        return crop_bgr
    lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** float(gamma)) * 255.0, 0.0, 255.0).astype(
        np.uint8
    )
    return cv2.LUT(crop_bgr, lut)


def apply_jpeg_compression(crop_bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for quality={quality}")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if dec is None:
        raise RuntimeError("cv2.imdecode returned None")
    return dec


def compute_extended_quality_features(
    crop_bgr: np.ndarray,
    landmarks_5x2: np.ndarray | None = None,
    det_confidence: float = 1.0,
    raw_embedding_norm: float | None = None,
) -> dict[str, float]:
    """Compute 8 independent face-quality indicators covering sharpness, resolution, pose,
    illumination, compression blockiness, detector confidence, and learned embedding norm.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))

    mean_lum = float(np.mean(gray_f))
    contrast_std = float(np.std(gray_f))
    clipped_frac = float(np.mean((gray <= 15) | (gray >= 240)))
    exposure_score = max(0.0, 1.0 - abs(mean_lum - 127.5) / 127.5) * (1.0 - clipped_frac)

    # 8x8 DCT boundary vs interior gradient ratio (JPEG blockiness indicator)
    h_diffs = np.abs(gray_f[:, 1:] - gray_f[:, :-1])
    v_diffs = np.abs(gray_f[1:, :] - gray_f[:-1, :])
    h_bnd_cols = [c for c in range(7, h_diffs.shape[1], 8)]
    v_bnd_rows = [r for r in range(7, v_diffs.shape[0], 8)]
    bnd_mean = 0.5 * (
        float(np.mean(h_diffs[:, h_bnd_cols])) + float(np.mean(v_diffs[v_bnd_rows, :]))
    )
    overall_grad = max(0.5 * (float(np.mean(h_diffs)) + float(np.mean(v_diffs))), 1e-6)
    jpeg_blockiness = bnd_mean / overall_grad

    if landmarks_5x2 is not None:
        pts = np.asarray(landmarks_5x2, dtype=np.float64)
        io_dist = float(np.linalg.norm(pts[1] - pts[0]))
        d_left = float(np.linalg.norm(pts[2] - pts[0]))
        d_right = float(np.linalg.norm(pts[2] - pts[1]))
        yaw_asym = abs(d_left - d_right) / max(io_dist, 1e-6)
        roll_deg = abs(math.degrees(math.atan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])))
    else:
        io_dist = 35.2
        yaw_asym = 0.0
        roll_deg = 0.0

    # Composite classical quality score in [0, 1]
    sharp_norm = min(1.0, math.log1p(lap_var) / math.log1p(600.0))
    res_norm = min(1.0, io_dist / 45.0)
    pose_norm = max(0.0, 1.0 - yaw_asym)
    classical_composite = 0.45 * sharp_norm + 0.25 * exposure_score + 0.15 * res_norm + 0.15 * pose_norm

    return {
        "sharpness_laplacian_var": round(lap_var, 4),
        "sharpness_tenengrad": round(tenengrad, 4),
        "inter_ocular_px": round(io_dist, 4),
        "yaw_asymmetry_ratio": round(yaw_asym, 6),
        "roll_angle_deg": round(roll_deg, 4),
        "mean_luminance": round(mean_lum, 4),
        "contrast_std": round(contrast_std, 4),
        "exposure_score": round(exposure_score, 6),
        "jpeg_blockiness_ratio": round(jpeg_blockiness, 6),
        "detector_confidence": round(float(det_confidence), 6),
        "classical_composite_score": round(classical_composite, 6),
        "learned_embedding_l2_norm": round(float(raw_embedding_norm), 6)
        if raw_embedding_norm is not None
        else 0.0,
    }


def compute_pair_scores(
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute pairwise higher-is-more-similar scores under configurable geometry."""
    norm_a = l2_normalize_embeddings(raw_a)
    norm_b = l2_normalize_embeddings(raw_b)

    if metric == "cosine":
        return np.sum(norm_a * norm_b, axis=1).astype(np.float64)
    if metric == "normalized_l2":
        return (-np.linalg.norm(norm_a - norm_b, axis=1)).astype(np.float64)
    if metric == "unnormalized_l2":
        return (-np.linalg.norm(raw_a.astype(np.float64) - raw_b.astype(np.float64), axis=1)).astype(
            np.float64
        )
    if metric == "normalized_l1":
        return (-np.sum(np.abs(norm_a - norm_b), axis=1)).astype(np.float64)
    raise ValueError(f"Unsupported similarity metric='{metric}'")


def evaluate_calibrated_verification(
    calib_scores: np.ndarray,
    calib_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    target_fars: tuple[float, ...] = (0.001, 0.01, 0.05),
) -> dict[str, Any]:
    """Strictly calibrate operating thresholds on `calib` and evaluate realized TAR/FAR/FRR on
    identity-disjoint `test`. Also computes threshold-free ROC AUC and test EER.
    """
    cs = np.asarray(calib_scores, dtype=np.float64)
    cl = np.asarray(calib_labels, dtype=bool)
    ts = np.asarray(test_scores, dtype=np.float64)
    tl = np.asarray(test_labels, dtype=bool)

    s_min = float(min(cs.min(), ts.min())) - 1e-6
    s_max = float(max(cs.max(), ts.max())) + 1e-6
    grid = np.linspace(s_min, s_max, 4001)

    # 1. Calibrate best accuracy threshold exclusively on Calibration Split
    calib_preds = cs[:, None] >= grid[None, :]
    calib_accs = np.mean(calib_preds == cl[:, None], axis=0)
    best_idx = int(np.argmax(calib_accs))
    tau_acc = float(grid[best_idx])

    test_pred_acc = ts >= tau_acc
    test_gen = ts[tl]
    test_imp = ts[~tl]
    calib_imp = cs[~cl]

    test_acc = float(np.mean(test_pred_acc == tl))
    test_tar_at_tau_acc = float(np.mean(test_gen >= tau_acc))
    test_far_at_tau_acc = float(np.mean(test_imp >= tau_acc))
    test_frr_at_tau_acc = 1.0 - test_tar_at_tau_acc

    # 2. Threshold-free ROC AUC on Held-Out Test Split (exact Mann-Whitney U / rank statistic)
    combined = np.concatenate([test_gen, test_imp])
    # Average ranks for ties via scipy-equivalent sorting
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)
    # Tie adjustment
    sorted_vals = combined[order]
    diff_mask = np.concatenate([[True], sorted_vals[1:] != sorted_vals[:-1], [True]])
    starts = np.where(diff_mask)[0]
    for s_i, e_i in zip(starts[:-1], starts[1:]):
        if e_i - s_i > 1:
            avg_r = 0.5 * (s_i + 1 + e_i)
            ranks[order[s_i:e_i]] = avg_r
    n_pos = float(test_gen.size)
    n_neg = float(test_imp.size)
    u_stat = np.sum(ranks[: int(n_pos)]) - n_pos * (n_pos + 1.0) * 0.5
    roc_auc = float(u_stat / max(n_pos * n_neg, 1.0))

    # 3. Test DET curve & Equal Error Rate (EER)
    test_fars = np.mean(test_imp[:, None] >= grid[None, :], axis=0)
    test_frrs = np.mean(test_gen[:, None] < grid[None, :], axis=0)
    eer_idx = int(np.argmin(np.abs(test_fars - test_frrs)))
    test_eer = 0.5 * float(test_fars[eer_idx] + test_frrs[eer_idx])

    # 4. Out-of-sample calibrated TAR @ Target FAR
    calibrated_operating_points: dict[str, Any] = {}
    sorted_calib_imp = np.sort(calib_imp)
    n_c_imp = sorted_calib_imp.size
    for tfar in target_fars:
        max_allowed_fp = int(math.floor(tfar * n_c_imp))
        if max_allowed_fp <= 0:
            tau_far = float(sorted_calib_imp[-1]) + 1e-7
        elif max_allowed_fp >= n_c_imp:
            tau_far = float(sorted_calib_imp[0]) - 1e-7
        else:
            # Threshold strictly above the (N - max_allowed_fp)-th sorted impostor score
            tau_far = float(sorted_calib_imp[n_c_imp - max_allowed_fp - 1]) + 1e-7

        realized_calib_far = float(np.mean(calib_imp >= tau_far))
        realized_test_far = float(np.mean(test_imp >= tau_far))
        realized_test_tar = float(np.mean(test_gen >= tau_far))
        calibrated_operating_points[f"target_far_{tfar:g}"] = {
            "calibrated_threshold": round(tau_far, 6),
            "calib_far": round(realized_calib_far, 6),
            "test_realized_far": round(realized_test_far, 6),
            "test_realized_tar": round(realized_test_tar, 6),
            "test_realized_frr": round(1.0 - realized_test_tar, 6),
            "test_false_accepts_count": int(np.sum(test_imp >= tau_far)),
        }

    return {
        "calib_best_accuracy_pct": round(float(calib_accs[best_idx]) * 100.0, 4),
        "calibrated_acc_threshold": round(tau_acc, 6),
        "test_accuracy_at_calib_thresh_pct": round(test_acc * 100.0, 4),
        "test_tar_at_calib_acc_thresh_pct": round(test_tar_at_tau_acc * 100.0, 4),
        "test_far_at_calib_acc_thresh_pct": round(test_far_at_tau_acc * 100.0, 4),
        "test_frr_at_calib_acc_thresh_pct": round(test_frr_at_tau_acc * 100.0, 4),
        "test_roc_auc": round(roc_auc, 6),
        "test_eer_pct": round(test_eer * 100.0, 4),
        "calibrated_operating_points": calibrated_operating_points,
    }


def extract_with_flip_tta(embedder: FaceEmbedder, aligned_crops_bgr: list[np.ndarray]) -> EmbeddingBatchOutput:
    """Horizontal-flip Test-Time Augmentation (`z_tta = z(x) + z(hflip(x))`, then L2-normalize)."""
    t0 = time.perf_counter()
    orig_out = embedder.extract(aligned_crops_bgr, l2_normalize=False)
    flipped_crops = [cv2.flip(c, 1) for c in aligned_crops_bgr]
    flip_out = embedder.extract(flipped_crops, l2_normalize=False)
    summed_raw = orig_out.raw_embeddings + flip_out.raw_embeddings
    normed = l2_normalize_embeddings(summed_raw)
    post_ms = (time.perf_counter() - t0) * 1000.0 - (
        orig_out.preprocess_time_ms
        + orig_out.inference_time_ms
        + flip_out.preprocess_time_ms
        + flip_out.inference_time_ms
    )
    return EmbeddingBatchOutput(
        raw_embeddings=summed_raw,
        embeddings=normed,
        l2_normalized=True,
        embedding_dim=orig_out.embedding_dim,
        model_id=orig_out.model_id,
        model_version=orig_out.model_version,
        runtime=orig_out.runtime,
        device=orig_out.device,
        precision=orig_out.precision,
        is_synthetic_fixture=False,
        preprocess_time_ms=orig_out.preprocess_time_ms + flip_out.preprocess_time_ms,
        inference_time_ms=orig_out.inference_time_ms + flip_out.inference_time_ms,
        postprocess_time_ms=max(0.0, post_ms),
    )


def aggregate_subject_template(
    raw_embeddings: np.ndarray,
    mode: str = "uniform_mean",
    external_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate `K` multi-image face embeddings `(K, D)` into a single unit-L2 `D`-dim template."""
    unit_embs = l2_normalize_embeddings(raw_embeddings)
    if mode == "single_first":
        return unit_embs[0]
    if mode == "uniform_mean":
        mean_v = np.mean(unit_embs, axis=0, keepdims=True)
        return l2_normalize_embeddings(mean_v)[0]
    if mode == "learned_norm_weighted":
        norms = np.linalg.norm(raw_embeddings.astype(np.float64), axis=1)
        # Softmax / power-scaled norm weights emphasize clean high-norm captures over blurred shots
        centered = (norms - np.mean(norms)) / max(float(np.std(norms)), 1e-6)
        w = np.exp(np.clip(centered, -4.0, 4.0))
        w = (w / np.sum(w)).astype(np.float32)
        weighted = np.sum(unit_embs * w[:, None], axis=0, keepdims=True)
        return l2_normalize_embeddings(weighted)[0]
    if mode == "classical_quality_weighted":
        if external_weights is None:
            raise ValueError("external_weights required for mode='classical_quality_weighted'")
        w = np.maximum(np.asarray(external_weights, dtype=np.float64), 1e-6)
        w = (w / np.sum(w)).astype(np.float32)
        weighted = np.sum(unit_embs * w[:, None], axis=0, keepdims=True)
        return l2_normalize_embeddings(weighted)[0]
    raise ValueError(f"Unsupported aggregation mode='{mode}'")


class GuardedFixedBatchEmbedder(FaceEmbedder):
    """P3 clean fixed-batch wrapper resolving `P2-AUD-06`:
    1. Rejects models with `dynamic_batch_supported=False` (such as static `[1,3,112,112]` `SFace`).
    2. Never mutates `base_embedder.batch_size` permanently (restores state around each chunk call).
    """

    def __init__(self, base_embedder: FaceEmbedder, fixed_batch_size: int = 8) -> None:
        if base_embedder.is_synthetic_fixture:
            raise ValueError("GuardedFixedBatchEmbedder cannot wrap a synthetic fixture.")
        if not bool(getattr(base_embedder, "dynamic_batch_supported", True)):
            raise ValueError(
                f"Model '{base_embedder.model_id}' has dynamic_batch_supported=False and cannot "
                "be wrapped by GuardedFixedBatchEmbedder."
            )
        if fixed_batch_size < 1:
            raise ValueError(f"fixed_batch_size must be >= 1; got {fixed_batch_size}")
        self._base = base_embedder
        self.fixed_batch_size = int(fixed_batch_size)

    @property
    def model_id(self) -> str:
        return self._base.model_id

    @property
    def model_version(self) -> str:
        return self._base.model_version

    @property
    def embedding_dim(self) -> int:
        return self._base.embedding_dim

    @property
    def is_synthetic_fixture(self) -> bool:
        return False

    @property
    def initialization_time_ms(self) -> float:
        return self._base.initialization_time_ms

    def preprocess(self, aligned_crops_bgr: list[np.ndarray]) -> np.ndarray:
        return self._base.preprocess(aligned_crops_bgr)

    def _run_exact_fixed_chunk(self, fixed_chunk: np.ndarray) -> np.ndarray:
        has_bs = hasattr(self._base, "batch_size")
        prev_bs = getattr(self._base, "batch_size", None)
        try:
            if has_bs:
                setattr(self._base, "batch_size", self.fixed_batch_size)
            return self._base.infer_preprocessed(fixed_chunk)
        finally:
            if has_bs:
                setattr(self._base, "batch_size", prev_bs)

    def infer_preprocessed(self, batch_nchw: np.ndarray) -> np.ndarray:
        if batch_nchw.ndim != 4:
            raise ValueError(f"Expected 4D NCHW array; got {batch_nchw.shape}")
        n = batch_nchw.shape[0]
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        bs = self.fixed_batch_size
        outputs: list[np.ndarray] = []
        for start in range(0, n, bs):
            chunk = batch_nchw[start : start + bs]
            k = chunk.shape[0]
            if k < bs:
                pad_shape = (bs - k, *chunk.shape[1:])
                padded = np.concatenate([chunk, np.zeros(pad_shape, dtype=chunk.dtype)], axis=0)
                out_full = self._run_exact_fixed_chunk(padded)
                outputs.append(out_full[:k].copy())
            else:
                outputs.append(self._run_exact_fixed_chunk(chunk))
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def extract(
        self,
        aligned_crops_bgr: list[np.ndarray],
        l2_normalize: bool | None = None,
    ) -> EmbeddingBatchOutput:
        default_norm = getattr(self._base, "default_l2_normalize", True)
        do_norm = default_norm if l2_normalize is None else bool(l2_normalize)

        t_pre = time.perf_counter()
        batch_nchw = self.preprocess(aligned_crops_bgr)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

        t_inf = time.perf_counter()
        raw_embs = self.infer_preprocessed(batch_nchw)
        inference_ms = (time.perf_counter() - t_inf) * 1000.0

        t_post = time.perf_counter()
        final_embs = l2_normalize_embeddings(raw_embs) if do_norm else raw_embs.copy()
        postprocess_ms = (time.perf_counter() - t_post) * 1000.0

        return EmbeddingBatchOutput(
            raw_embeddings=raw_embs,
            embeddings=final_embs,
            l2_normalized=do_norm,
            embedding_dim=self.embedding_dim,
            model_id=self.model_id,
            model_version=self.model_version,
            runtime=str(getattr(self._base, "runtime", "onnxruntime")),
            device=str(getattr(self._base, "device", "unknown")),
            precision=str(getattr(self._base, "precision", "fp32")),
            is_synthetic_fixture=False,
            preprocess_time_ms=preprocess_ms,
            inference_time_ms=inference_ms,
            postprocess_time_ms=postprocess_ms,
        )
