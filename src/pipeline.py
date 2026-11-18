"""End-to-end Milestone P1 Facial Recognition Pipeline (YuNet + ResNet50 Adapter).

Pipeline Architecture:
  Image Input -> Image Decoding -> YuNet Face Detection -> Bounding Boxes + Landmarks
  -> 5-Point Face Alignment -> Classical Quality Assessment -> ResNet50 Preprocessing
  -> ResNet50 Feature Extraction -> Embedding L2 Normalization -> Face-Instance Results

Strict Governance Constraints:
- Detection (YuNet) and Recognition (ResNet50) remain strictly separated.
- Multiple faces per image return independent `FaceInstanceResult` entries.
- No 1:N identification, clustering, or person identity resolution is performed.
- If the optional custom ResNet50 weights are unavailable locally, synthetic or
  substitute weights are NEVER used; face instances explicitly report
  `embedding=None` and `embedding_status="BLOCKED_MISSING_REFERENCE_RESNET50_WEIGHTS"`
  (or raise `MissingModelArtifactError` when `strict_recognition_artifact=True`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.alignment.face_aligner import FaceAligner
from src.common.config_loader import (
    compute_sha256,
    load_yaml_config,
    resolve_project_path,
)
from src.common.types import FaceInstanceResult, ImagePipelineResult
from src.detection.base_detector import FaceDetector, InvalidImageError, validate_bgr_image
from src.detection.yunet_detector import YuNetDetector
from src.quality.face_quality import FaceQualityAssessor
from src.recognition.base_embedder import (
    FaceEmbedder,
    MissingModelArtifactError,
    l2_normalize_embeddings,
)
from src.recognition.resnet50_embedder import (
    ReferenceResNet50Embedder,
    ResNet50Preprocessor,
)


def decode_bgr_image(image_source: str | Path | np.ndarray) -> tuple[np.ndarray, str, str | None, float]:
    """Decode an image from disk (supporting Unicode Windows paths) or validate an ndarray."""
    t0 = time.perf_counter()
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        if raw_bytes.size == 0:
            raise InvalidImageError(f"Image file is empty (0 bytes): {path}")
        decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if decoded is None:
            raise InvalidImageError(f"Failed to decode corrupt or unsupported image file: {path}")
        validate_bgr_image(decoded)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        return decoded, path.stem, str(path), decode_ms

    if isinstance(image_source, np.ndarray):
        validate_bgr_image(image_source)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        return image_source, "in_memory_image", None, decode_ms

    raise InvalidImageError(f"Unsupported image_source type: {type(image_source).__name__}")


class FaceRecognitionPipeline:
    """Modular single-image facial detection, alignment, quality, and embedding pipeline."""

    def __init__(
        self,
        config: str | Path | dict[str, Any] = "configs/models.yaml",
        detector: FaceDetector | None = None,
        aligner: FaceAligner | None = None,
        quality_assessor: FaceQualityAssessor | None = None,
        embedder: FaceEmbedder | None = None,
        strict_recognition_artifact: bool | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            self._config_path: str | Path = config
            self._cfg = load_yaml_config(config)
        elif isinstance(config, dict):
            self._config_path = "configs/models.yaml"
            self._cfg = config
        else:
            raise TypeError("config must be a YAML file path or configuration dictionary.")

        pipe_cfg: dict[str, Any] = self._cfg.get("pipeline", {})
        det_key = str(pipe_cfg.get("detector_model_key", "yunet_2023mar"))
        rec_key = str(pipe_cfg.get("recognizer_model_key", "reference_resnet50_baseline"))
        self.l2_normalize_enabled = bool(pipe_cfg.get("l2_normalize_embeddings", True))
        self.strict_recognition_artifact = bool(
            strict_recognition_artifact
            if strict_recognition_artifact is not None
            else pipe_cfg.get("strict_recognition_artifact", False)
        )

        det_model_cfg = self._cfg.get("models", {}).get(det_key, {})
        det_params = det_model_cfg.get("detection_params", {})
        self.detector: FaceDetector = detector or YuNetDetector(
            model_id=det_key,
            model_path=det_model_cfg.get("model_path"),
            score_threshold=det_params.get("score_threshold"),
            nms_threshold=det_params.get("nms_threshold"),
            top_k=det_params.get("top_k"),
            resize_mode=det_params.get("resize_mode"),
            max_input_dim=det_params.get("max_input_dim"),
            fixed_input_size_wh=(
                tuple(det_params["fixed_input_size"])
                if "fixed_input_size" in det_params
                else None
            ),
            config_path=self._config_path,
        )
        self.aligner: FaceAligner = aligner or FaceAligner(config_path=self._config_path)
        self.quality_assessor: FaceQualityAssessor = quality_assessor or FaceQualityAssessor(
            config_path=self._config_path
        )

        rec_model_cfg = self._cfg["models"][rec_key]
        self._recognizer_cfg = rec_model_cfg
        self._recognizer_key = rec_key
        self._fallback_preprocessor = ResNet50Preprocessor.from_model_config(
            rec_model_cfg, precision_override="fp32"
        )

        self.embedder: FaceEmbedder | None = None
        self._recognizer_missing_reason: str | None = None

        if embedder is not None:
            if embedder.is_synthetic_fixture:
                raise ValueError(
                    "SyntheticResNet50FixtureEmbedder cannot be attached to FaceRecognitionPipeline. "
                    "Synthetic outputs must never be represented as actual facial embeddings."
                )
            self.embedder = embedder
        else:
            try:
                self.embedder = ReferenceResNet50Embedder(
                    model_id=rec_key, config_path=self._cfg
                )
            except MissingModelArtifactError as exc:
                if self.strict_recognition_artifact:
                    raise
                self._recognizer_missing_reason = str(exc)

        self._has_processed_once = False
        self._detector_provenance = self._build_detector_provenance(det_key)
        self._recognizer_provenance = self._build_recognizer_provenance(rec_key)

    def _build_detector_provenance(self, det_key: str) -> dict[str, Any]:
        det_cfg = self._cfg.get("models", {}).get(det_key, {})
        det_path = resolve_project_path(det_cfg.get("model_path", "")) if det_cfg.get("model_path") else None
        sha = compute_sha256(det_path) if (det_path and det_path.is_file()) else None
        return {
            "model_id": self.detector.model_id,
            "model_name": det_cfg.get("model_name", self.detector.model_id),
            "model_version": self.detector.model_version,
            "runtime": det_cfg.get("runtime", "opencv_dnn"),
            "sha256": sha,
            "status": "ACTIVE",
        }

    def _build_recognizer_provenance(self, rec_key: str) -> dict[str, Any]:
        rec_cfg = self._cfg.get("models", {}).get(rec_key, {})
        if self.embedder is not None:
            emb_sha = getattr(self.embedder, "model_sha256", None)
            emb_path = getattr(self.embedder, "model_path", None)
            if emb_sha is None:
                rec_path = (
                    Path(emb_path)
                    if emb_path is not None
                    else (
                        resolve_project_path(rec_cfg.get("model_path", ""))
                        if rec_cfg.get("model_path")
                        else None
                    )
                )
                emb_sha = compute_sha256(rec_path) if (rec_path and rec_path.is_file()) else None
            return {
                "model_id": self.embedder.model_id,
                "model_name": rec_cfg.get("model_name", self.embedder.model_id),
                "model_version": self.embedder.model_version,
                "architecture": getattr(
                    self.embedder, "architecture", rec_cfg.get("architecture", "resnet50")
                ),
                "runtime": getattr(self.embedder, "runtime", rec_cfg.get("runtime", "onnxruntime")),
                "device": getattr(self.embedder, "device", "unknown"),
                "precision": getattr(self.embedder, "precision", "fp32"),
                "sha256": emb_sha,
                "status": "ACTIVE",
            }
        return {
            "model_id": rec_key,
            "model_name": rec_cfg.get("model_name", rec_key),
            "model_version": rec_cfg.get("model_version", "pending_internal_delivery"),
            "runtime": rec_cfg.get("runtime", "onnxruntime"),
            "sha256": None,
            "status": "BLOCKED_MISSING_REFERENCE_RESNET50_WEIGHTS",
            "reason": self._recognizer_missing_reason,
        }

    def process_image(
        self,
        image_source: str | Path | np.ndarray,
        image_id: str | None = None,
    ) -> ImagePipelineResult:
        """Process a single image end-to-end and return independent face-instance results."""
        is_cold_start = not self._has_processed_once
        t_pipeline_start = time.perf_counter()

        # 1. Image decoding
        image_bgr, inferred_id, resolved_path_str, decode_ms = decode_bgr_image(image_source)
        final_image_id = image_id or inferred_id
        img_h, img_w, img_c = image_bgr.shape

        # 2. YuNet face detection
        det_batch = self.detector.detect(image_bgr)
        detection_ms = det_batch.execution_time_ms
        num_detected = len(det_batch.detections)

        # 3. Face alignment & 4. Image quality assessment
        aligned_records = []
        quality_records = []
        alignment_total_ms = 0.0
        quality_total_ms = 0.0

        for det in det_batch.detections:
            aligned_face = self.aligner.align(image_bgr, det.landmarks, source_bbox=det.bbox)
            alignment_total_ms += aligned_face.execution_time_ms
            aligned_records.append(aligned_face)

            q_res = self.quality_assessor.assess(
                source_image_bgr=image_bgr,
                bbox=det.bbox,
                aligned_crop_bgr=aligned_face.aligned_bgr,
            )
            quality_total_ms += q_res.execution_time_ms
            quality_records.append(q_res)

        # 5. ResNet50 preprocessing, 6. ResNet50 inference, 7. Embedding normalization
        crops_bgr = [rec.aligned_bgr for rec in aligned_records]
        raw_embs: np.ndarray | None = None
        final_embs: np.ndarray | None = None
        emb_dim: int | None = None
        rec_preprocess_ms: float = 0.0
        rec_inference_ms: float | None = 0.0 if self.embedder is not None else None
        emb_norm_ms: float | None = 0.0 if self.embedder is not None else None
        emb_status: str = "NO_FACES_DETECTED"

        if num_detected > 0:
            if self.embedder is not None:
                t_pre = time.perf_counter()
                batch_nchw = self.embedder.preprocess(crops_bgr)
                rec_preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

                t_inf = time.perf_counter()
                raw_embs = self.embedder.infer_preprocessed(batch_nchw)
                rec_inference_ms = (time.perf_counter() - t_inf) * 1000.0

                t_norm = time.perf_counter()
                final_embs = (
                    l2_normalize_embeddings(raw_embs)
                    if self.l2_normalize_enabled
                    else raw_embs.copy()
                )
                emb_norm_ms = (time.perf_counter() - t_norm) * 1000.0
                emb_dim = int(final_embs.shape[1])
                emb_status = "COMPUTED"
            else:
                # Still execute real preprocessing validation on aligned crops
                t_pre = time.perf_counter()
                _ = self._fallback_preprocessor.preprocess(crops_bgr)
                rec_preprocess_ms = (time.perf_counter() - t_pre) * 1000.0
                rec_inference_ms = None
                emb_norm_ms = None
                emb_dim = None
                emb_status = "BLOCKED_MISSING_REFERENCE_RESNET50_WEIGHTS"

        total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
        self._has_processed_once = True

        models_meta = {
            "detector": self._detector_provenance,
            "recognizer": self._recognizer_provenance,
        }

        face_results: list[FaceInstanceResult] = []
        for idx, (det, aligned_face, q_res) in enumerate(
            zip(det_batch.detections, aligned_records, quality_records)
        ):
            per_face_pre = rec_preprocess_ms / num_detected if num_detected > 0 else 0.0
            per_face_inf = (
                (rec_inference_ms / num_detected)
                if (rec_inference_ms is not None and num_detected > 0)
                else None
            )
            per_face_norm = (
                (emb_norm_ms / num_detected)
                if (emb_norm_ms is not None and num_detected > 0)
                else None
            )

            face_results.append(
                FaceInstanceResult(
                    image_id=final_image_id,
                    face_instance_id=f"{final_image_id}::face_{idx:03d}",
                    bounding_box=det.bbox.to_dict(),
                    landmarks=det.landmarks.to_dict(),
                    detection_confidence=round(float(det.confidence), 6),
                    alignment={
                        "output_size_wh": list(aligned_face.output_size_wh),
                        "crop_bounds_xyxy": list(aligned_face.crop_bounds_xyxy),
                        "inter_ocular_px": round(float(aligned_face.inter_ocular_px), 3),
                    },
                    quality=q_res.to_dict(),
                    raw_embedding=raw_embs[idx].tolist() if raw_embs is not None else None,
                    embedding=final_embs[idx].tolist() if final_embs is not None else None,
                    embedding_dim=emb_dim,
                    l2_normalized=bool(self.l2_normalize_enabled and final_embs is not None),
                    embedding_status=emb_status,
                    models=models_meta,
                    stage_latency_ms={
                        "alignment_ms": round(aligned_face.execution_time_ms, 4),
                        "quality_ms": round(q_res.execution_time_ms, 4),
                        "recognition_preprocess_ms": round(per_face_pre, 4),
                        "recognition_inference_ms": (
                            round(per_face_inf, 4) if per_face_inf is not None else None
                        ),
                        "embedding_normalization_ms": (
                            round(per_face_norm, 4) if per_face_norm is not None else None
                        ),
                    },
                )
            )

        stage_latencies: dict[str, float | None] = {
            "image_decode_ms": round(decode_ms, 4),
            "detection_ms": round(detection_ms, 4),
            "alignment_ms": round(alignment_total_ms, 4),
            "quality_ms": round(quality_total_ms, 4),
            "recognition_preprocess_ms": round(rec_preprocess_ms, 4),
            "recognition_inference_ms": (
                round(rec_inference_ms, 4) if rec_inference_ms is not None else None
            ),
            "embedding_normalization_ms": (
                round(emb_norm_ms, 4) if emb_norm_ms is not None else None
            ),
            "total_ms": round(total_ms, 4),
        }

        return ImagePipelineResult(
            image_id=final_image_id,
            image_path=resolved_path_str,
            image_dimensions={"width": int(img_w), "height": int(img_h), "channels": int(img_c)},
            num_faces=num_detected,
            faces=face_results,
            models=models_meta,
            stage_latency_ms=stage_latencies,
            cold_start=is_cold_start,
        )
