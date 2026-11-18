"""Milestone P2 — Candidate Face Recognition Embedders & Fixed-Batch Scheduler.

Implements verified `FaceEmbedder` adapters without assuming shared preprocessing:
  1. `SFaceEmbedder` (`sface_2021dec` — `models/recognition/face_recognition_sface_2021dec.onnx`):
     - Architecture: MobileFaceNet trained with SFace Loss (`128`-D output, `38,696,353` bytes).
     - Verified Preprocessing: `112x112` 5-point similarity aligned crop, **`RGB`** (`swapRB=True`),
       raw `float32 [0.0, 255.0]` (`mean=[0,0,0], std=[1,1,1]`), because `(x - 127.5) * 0.0078125`
       is embedded inside the ONNX graph's first two scalar nodes (`_minusscalar0`, `_mulscalar0`).
       Verified `cosine = 1.0000000`, `max_abs_diff = 3.46e-06` vs. `cv2.FaceRecognizerSF.feature()`.
     - Graph Constraint: Static input shape `[1, 3, 112, 112]`; multi-face batches execute sequential
       `B = 1` forward passes on `CPUExecutionProvider`, `CUDAExecutionProvider` (`io_binding`), or
       `opencv_dnn` (`cv2.FaceRecognizerSF`).
  2. `MobileFaceNetEmbedder` (`mobilefacenet_w600k_research` — `mobilefacenet_w600k_mbf.onnx`):
     - Architecture: InsightFace MobileFaceNet (`512`-D output, `3,400,448` params, `13,616,099` bytes).
     - Verified Preprocessing (`arcface_onnx.py`): `112x112` 5-point ArcFace aligned crop, **`RGB`**
       (`swapRB=True`), `(rgb_uint8 - 127.5) / 127.5 -> [-1.0, 1.0]`.
     - Supports native dynamic batching (`['None', 3, 112, 112]`) across `B = 1, 4, 8, 16` on
       `CPUExecutionProvider` and `CUDAExecutionProvider` (`io_binding`).
  3. `FixedBatchPaddedEmbedder`:
     - Non-invasive `FaceEmbedder` decorator for Workstream C's controlled batch-scheduling experiment:
       pads partial tail batches (`1 <= k < fixed_batch_size`) with zeros up to `fixed_batch_size`
       (e.g., `B = 8`), executes constant-shape GPU inference to eliminate ONNX Runtime `last_x_dims`
       cuDNN descriptor rebuilds, and slices `[:k]` cleanly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config_loader import compute_sha256, load_yaml_config, resolve_project_path
from src.common.onnx_utils import ensure_cuda_dll_paths
from src.recognition.base_embedder import (
    EmbeddingBatchOutput,
    FaceEmbedder,
    MissingModelArtifactError,
    l2_normalize_embeddings,
)
from src.recognition.resnet50_embedder import ResNet50Preprocessor


class ONNXCandidateEmbedder(FaceEmbedder):
    """Config-driven ONNX Runtime (and optional OpenCV SFace) recognition adapter."""

    def __init__(
        self,
        model_id: str,
        model_path: str | Path | None = None,
        runtime: str | None = None,
        device: str | None = None,
        precision: str | None = None,
        batch_size: int | None = None,
        l2_normalize: bool | None = None,
        config_path: str | Path | dict[str, Any] = "configs/p2_models.yaml",
    ) -> None:
        cfg = config_path if isinstance(config_path, dict) else load_yaml_config(config_path)
        if model_id not in cfg.get("models", {}):
            raise KeyError(f"Unknown recognition model_id='{model_id}' in P2 configuration")
        model_cfg = cfg["models"][model_id]

        self._model_id = model_id
        self._model_name = str(model_cfg.get("model_name", model_id))
        self._model_version = str(model_cfg.get("model_version", "unknown"))
        self._architecture = str(model_cfg.get("architecture", model_id))
        self._licensing_status = dict(model_cfg.get("licensing_status", {}))

        in_dims = model_cfg.get("input_dimensions", [1, 3, 112, 112])
        self._expected_channels = int(in_dims[1])
        self._expected_height = int(in_dims[2])
        self._expected_width = int(in_dims[3])
        self._expected_embedding_dim = int(model_cfg["embedding_dim"])
        self._embedding_dim = self._expected_embedding_dim
        self._dynamic_batch_supported = bool(model_cfg.get("dynamic_batch_supported", True))

        supported_runtimes = [str(r).lower() for r in model_cfg.get("supported_runtimes", ["onnxruntime"])]
        self._runtime = str(runtime or model_cfg.get("runtime", "onnxruntime")).lower()
        if self._runtime not in supported_runtimes:
            raise ValueError(
                f"Unsupported runtime='{self._runtime}' for '{model_id}'; supported={supported_runtimes}"
            )

        supported_precisions = [str(p).lower() for p in model_cfg.get("supported_precisions", ["fp32"])]
        self._precision = str(precision or model_cfg.get("precision", "fp32")).lower()
        if self._precision not in supported_precisions:
            raise ValueError(
                f"Unsupported precision='{self._precision}' for '{model_id}'; "
                f"supported={supported_precisions} (upstream distributes FP32 ONNX only)"
            )

        ensure_cuda_dll_paths()
        import onnxruntime as ort

        req_dev = str(device or model_cfg.get("device", "cuda")).lower()
        if self._runtime == "opencv_dnn":
            if req_dev.startswith("cuda"):
                raise ValueError("opencv_dnn FaceRecognizerSF in PyPI opencv-python supports CPU only.")
            self._device = "cpu"
        elif req_dev.startswith("cuda"):
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                fallback = str(model_cfg.get("fallback_device", "cpu")).lower()
                if fallback == "cpu":
                    self._device = "cpu"
                else:
                    raise RuntimeError("CUDAExecutionProvider not available in ONNX Runtime.")
            else:
                self._device = "cuda:0"
        elif req_dev == "cpu":
            self._device = "cpu"
        else:
            raise ValueError(f"Unsupported device='{req_dev}'")

        self.batch_size = int(batch_size or model_cfg.get("batch_size", 8))
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}")

        norm_cfg = model_cfg.get("normalization", {})
        self.default_l2_normalize = bool(
            l2_normalize if l2_normalize is not None else norm_cfg.get("l2_normalize", True)
        )
        self.preprocessor = ResNet50Preprocessor.from_model_config(
            model_cfg, precision_override=self._precision
        )

        resolved_path = resolve_project_path(model_path or model_cfg["model_path"])
        if not resolved_path.is_file():
            raise MissingModelArtifactError(f"Model artifact not found at '{resolved_path}'")
        self._model_path = resolved_path
        self._model_sha256 = compute_sha256(resolved_path)

        self._ort_session: Any = None
        self._ort_input_name: str = ""
        self._ort_output_name: str = ""
        self._cv_sface: Any = None

        t0 = time.perf_counter()
        if self._runtime == "opencv_dnn":
            self._cv_sface = cv2.FaceRecognizerSF.create(str(resolved_path), "")
        else:
            sess_opts = ort.SessionOptions()
            # Suppress noisy legacy MXNet initializer-in-input warnings on SFace
            sess_opts.log_severity_level = 3
            if self._device.startswith("cuda"):
                ep_list: list[Any] = [
                    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                    "CPUExecutionProvider",
                ]
            else:
                ep_list = ["CPUExecutionProvider"]
            self._ort_session = ort.InferenceSession(
                str(resolved_path), sess_options=sess_opts, providers=ep_list
            )
            if self._device.startswith("cuda") and "CUDAExecutionProvider" not in self._ort_session.get_providers():
                raise RuntimeError(
                    f"CUDA requested for '{model_id}', but active providers are {self._ort_session.get_providers()}"
                )
            in_meta = self._ort_session.get_inputs()[0]
            out_meta = self._ort_session.get_outputs()[0]
            self._ort_input_name = in_meta.name
            self._ort_output_name = out_meta.name
            if isinstance(in_meta.shape[0], int) and in_meta.shape[0] == 1:
                self._dynamic_batch_supported = False
        self._init_time_ms = (time.perf_counter() - t0) * 1000.0

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
    def architecture(self) -> str:
        return self._architecture

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def runtime(self) -> str:
        return self._runtime

    @property
    def device(self) -> str:
        return self._device

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def dynamic_batch_supported(self) -> bool:
        return self._dynamic_batch_supported

    @property
    def active_providers(self) -> list[str]:
        if self._ort_session is not None:
            return list(self._ort_session.get_providers())
        return ["opencv_dnn_cpu"]

    @property
    def is_synthetic_fixture(self) -> bool:
        return False

    @property
    def initialization_time_ms(self) -> float:
        return self._init_time_ms

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "model_name": self._model_name,
            "model_version": self._model_version,
            "architecture": self._architecture,
            "runtime": self._runtime,
            "device": self._device,
            "precision": self._precision,
            "dynamic_batch_supported": self._dynamic_batch_supported,
            "model_path": str(self._model_path),
            "sha256": self._model_sha256,
            "embedding_dim": self._embedding_dim,
            "licensing_status": dict(self._licensing_status),
        }

    def preprocess(self, aligned_crops_bgr: list[np.ndarray]) -> np.ndarray:
        return self.preprocessor.preprocess(aligned_crops_bgr)

    def _run_ort_chunk(self, chunk: np.ndarray) -> np.ndarray:
        import onnxruntime as ort

        ort_chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        if self._device.startswith("cuda"):
            io_b = self._ort_session.io_binding()
            ort_in = ort.OrtValue.ortvalue_from_numpy(ort_chunk, "cuda", 0)
            io_b.bind_ortvalue_input(self._ort_input_name, ort_in)
            io_b.bind_output(self._ort_output_name, "cuda", 0)
            self._ort_session.run_with_iobinding(io_b)
            io_b.synchronize_outputs()
            raw = io_b.get_outputs()[0].numpy()
        else:
            raw = self._ort_session.run([self._ort_output_name], {self._ort_input_name: ort_chunk})[0]
        return np.asarray(raw, dtype=np.float32)

    def infer_preprocessed(self, batch_nchw: np.ndarray) -> np.ndarray:
        if not isinstance(batch_nchw, np.ndarray) or batch_nchw.ndim != 4:
            raise ValueError(f"Expected 4D NCHW array; got {getattr(batch_nchw, 'shape', None)}")
        n, c, h, w = batch_nchw.shape
        if (c, h, w) != (self._expected_channels, self._expected_height, self._expected_width):
            raise ValueError(
                f"Expected tensor shape (N, {self._expected_channels}, {self._expected_height}, "
                f"{self._expected_width}); got {batch_nchw.shape}"
            )
        if n == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        outputs: list[np.ndarray] = []
        if self._runtime == "opencv_dnn":
            # Reconstruct BGR uint8 crop for cv2.FaceRecognizerSF.feature() parity path
            for idx in range(n):
                rgb_chw = np.clip(batch_nchw[idx], 0.0, 255.0).astype(np.uint8)
                rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
                bgr_hwc = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
                feat = self._cv_sface.feature(bgr_hwc)
                outputs.append(np.asarray(feat, dtype=np.float32))
        else:
            eff_bs = 1 if not self._dynamic_batch_supported else self.batch_size
            for start in range(0, n, eff_bs):
                chunk = batch_nchw[start : start + eff_bs]
                outputs.append(self._run_ort_chunk(chunk))

        combined = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if combined.shape != (n, self._expected_embedding_dim):
            raise ValueError(
                f"Output shape mismatch for '{self._model_id}': expected {(n, self._expected_embedding_dim)}, "
                f"got {combined.shape}"
            )
        return combined

    def extract(
        self,
        aligned_crops_bgr: list[np.ndarray],
        l2_normalize: bool | None = None,
    ) -> EmbeddingBatchOutput:
        do_norm = self.default_l2_normalize if l2_normalize is None else bool(l2_normalize)

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
            embedding_dim=self._embedding_dim,
            model_id=self._model_id,
            model_version=self._model_version,
            runtime=self._runtime,
            device=self._device,
            precision=self._precision,
            is_synthetic_fixture=False,
            preprocess_time_ms=preprocess_ms,
            inference_time_ms=inference_ms,
            postprocess_time_ms=postprocess_ms,
        )


class SFaceEmbedder(ONNXCandidateEmbedder):
    """OpenCV Zoo `SFace` (`2021dec`, `128`-D) recognition embedder."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_id", "sface_2021dec")
        super().__init__(**kwargs)


class MobileFaceNetEmbedder(ONNXCandidateEmbedder):
    """InsightFace `MobileFaceNet W600K` (`buffalo_s`, `512`-D) recognition embedder."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_id", "mobilefacenet_w600k_research")
        super().__init__(**kwargs)


class FixedBatchPaddedEmbedder(FaceEmbedder):
    """Fixed-batch padding decorator for controlled GPU dynamic-shape vs. fixed-shape experiments.

    Pads any partial chunk `1 <= k < fixed_batch_size` up to `fixed_batch_size` so every forward
    call into ONNX Runtime `CUDAExecutionProvider` has identical tensor shape `(fixed_batch_size, C, H, W)`,
    eliminating ORT's 1-entry `last_x_dims` cuDNN descriptor rebuild on variable-face images.
    """

    def __init__(self, base_embedder: FaceEmbedder, fixed_batch_size: int = 8) -> None:
        if base_embedder.is_synthetic_fixture:
            raise ValueError("FixedBatchPaddedEmbedder cannot wrap a synthetic fixture.")
        if fixed_batch_size < 1:
            raise ValueError(f"fixed_batch_size must be >= 1; got {fixed_batch_size}")
        self._base = base_embedder
        self.fixed_batch_size = int(fixed_batch_size)
        if hasattr(self._base, "batch_size"):
            self._base.batch_size = self.fixed_batch_size  # type: ignore[attr-defined]

    @property
    def model_id(self) -> str:
        return self._base.model_id

    @property
    def model_version(self) -> str:
        return self._base.model_version

    @property
    def architecture(self) -> str:
        return str(getattr(self._base, "architecture", self._base.model_id))

    @property
    def model_path(self) -> Path | None:
        return getattr(self._base, "model_path", None)

    @property
    def model_sha256(self) -> str | None:
        return getattr(self._base, "model_sha256", None)

    @property
    def runtime(self) -> str:
        return str(getattr(self._base, "runtime", "onnxruntime"))

    @property
    def device(self) -> str:
        return str(getattr(self._base, "device", "unknown"))

    @property
    def precision(self) -> str:
        return str(getattr(self._base, "precision", "fp32"))

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
                out_full = self._base.infer_preprocessed(padded)
                outputs.append(out_full[:k])
            else:
                outputs.append(self._base.infer_preprocessed(chunk))
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
            runtime=self.runtime,
            device=self.device,
            precision=self.precision,
            is_synthetic_fixture=False,
            preprocess_time_ms=preprocess_ms,
            inference_time_ms=inference_ms,
            postprocess_time_ms=postprocess_ms,
        )
