"""ResNet50-family facial feature extraction adapter and decoupled preprocessor (Milestones P1 & P1.1).

Supports both:
- Publicly documented 50-layer residual face recognition models (`architecture: "ir_50"` / `"ir_se_50"`,
  such as official `AdaFace IR-50 WebFace4M` in PyTorch `.pt` and exported FP32/FP16 `.onnx`).
- Legacy reference ResNet50 adapter contract (`reference_resnet50_baseline`; raises `MissingModelArtifactError`
  when absent — never fabricates or silently substitutes ImageNet weights).

Governance Rules Enforced:
1. Never fabricates missing weights or substitutes ImageNet classification weights.
2. Uses model-specific preprocessing loaded from `configs/models.yaml` (`color_order`, `scale`,
   `mean`, `std`, `input_dimensions`) rather than assuming identical ResNet50 preprocessing.
3. Strictly separates preprocessing, batched inference (CPU/CUDA, FP32/FP16), and L2 normalization.
4. Validates input crop shapes, NCHW tensor dimensions, and output embedding dimensions, and
   records full SHA-256 artifact provenance.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.common.config_loader import (
    compute_sha256,
    load_yaml_config,
    resolve_project_path,
)
from src.common.onnx_utils import ensure_cuda_dll_paths
from src.detection.base_detector import InvalidImageError, validate_bgr_image
from src.recognition.adaface_ir50 import load_adaface_ir50_checkpoint
from src.recognition.base_embedder import (
    EmbeddingBatchOutput,
    FaceEmbedder,
    MissingModelArtifactError,
    l2_normalize_embeddings,
)


class ResNet50Preprocessor:
    """Configurable model-specific face crop preprocessor (never assumes ImageNet statistics)."""

    def __init__(
        self,
        input_size_wh: tuple[int, int] = (112, 112),
        color_order: str = "RGB",
        scale: float = 1.0,
        mean: tuple[float, float, float] = (127.5, 127.5, 127.5),
        std: tuple[float, float, float] = (127.5, 127.5, 127.5),
        precision: str = "fp32",
    ) -> None:
        self.input_size_wh = (int(input_size_wh[0]), int(input_size_wh[1]))
        self.color_order = color_order.upper()
        if self.color_order not in {"RGB", "BGR"}:
            raise ValueError(f"Unsupported color_order='{color_order}'; expected 'RGB' or 'BGR'.")

        self.scale = np.float32(scale)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if np.any(self.std == 0):
            raise ValueError("Normalization std values must be non-zero.")

        self.precision = precision.lower()
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported precision='{precision}'; expected 'fp32' or 'fp16'.")

    @classmethod
    def from_model_config(
        cls,
        model_cfg: dict[str, Any],
        precision_override: str | None = None,
    ) -> ResNet50Preprocessor:
        in_dims = model_cfg.get("input_dimensions", [1, 3, 112, 112])
        height, width = int(in_dims[2]), int(in_dims[3])
        norm = model_cfg.get("normalization", {})
        precision = precision_override or str(model_cfg.get("precision", "fp32"))
        return cls(
            input_size_wh=(width, height),
            color_order=str(norm.get("color_order", "RGB")),
            scale=float(norm.get("scale", 1.0)),
            mean=tuple(float(x) for x in norm.get("mean", [127.5, 127.5, 127.5])),  # type: ignore[arg-type]
            std=tuple(float(x) for x in norm.get("std", [127.5, 127.5, 127.5])),  # type: ignore[arg-type]
            precision=precision,
        )

    def preprocess(self, aligned_crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Validate and transform a list of BGR uint8 crops into a contiguous NCHW array."""
        if not isinstance(aligned_crops_bgr, list):
            raise InvalidImageError("aligned_crops_bgr must be a list of numpy.ndarray crops.")

        target_w, target_h = self.input_size_wh
        out_dtype = np.float16 if self.precision == "fp16" else np.float32

        if len(aligned_crops_bgr) == 0:
            return np.empty((0, 3, target_h, target_w), dtype=out_dtype)

        tensors: list[np.ndarray] = []
        for idx, crop in enumerate(aligned_crops_bgr):
            try:
                cw, ch = validate_bgr_image(crop)
            except InvalidImageError as exc:
                raise InvalidImageError(f"Invalid crop at index {idx}: {exc}") from exc

            work = (
                cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                if (cw, ch) != (target_w, target_h)
                else crop
            )
            if self.color_order == "RGB":
                work = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

            normed = ((work.astype(np.float32) * self.scale) - self.mean) / self.std
            chw = np.transpose(normed, (2, 0, 1))
            tensors.append(chw.astype(out_dtype, copy=False))

        return np.ascontiguousarray(np.stack(tensors, axis=0), dtype=out_dtype)


class ReferenceResNet50Embedder(FaceEmbedder):
    """Unified 50-layer residual face recognition embedder (PyTorch & ONNX Runtime, CPU & CUDA).

    Loads the configured pretrained checkpoint (e.g., `adaface_ir50_webface4m`) or ONNX graph
    once at initialization, executes batched inference in FP32 or FP16, validates output
    embedding dimensionality, and exposes full SHA-256 provenance.
    """

    def __init__(
        self,
        model_id: str = "adaface_ir50_webface4m",
        model_path: str | Path | None = None,
        runtime: str | None = None,
        device: str | None = None,
        precision: str | None = None,
        batch_size: int | None = None,
        l2_normalize: bool | None = None,
        pytorch_module: nn.Module | None = None,
        config_path: str | Path | dict[str, Any] = "configs/models.yaml",
    ) -> None:
        cfg_all = config_path if isinstance(config_path, dict) else load_yaml_config(config_path)
        if model_id not in cfg_all.get("models", {}):
            raise KeyError(f"Model '{model_id}' not found in configuration")

        model_cfg: dict[str, Any] = cfg_all["models"][model_id]
        self._model_id = model_id
        self._model_name = str(model_cfg.get("model_name", model_id))
        self._model_version = str(model_cfg.get("model_version", "unknown"))
        self._architecture = str(model_cfg.get("architecture", "resnet50"))
        self._expected_embedding_dim = int(model_cfg.get("embedding_dim", 512))
        self._embedding_dim = self._expected_embedding_dim
        self._licensing_status: dict[str, Any] = dict(model_cfg.get("licensing_status", {}))

        in_dims = model_cfg.get("input_dimensions", [1, 3, 112, 112])
        self._expected_channels = int(in_dims[1])
        self._expected_height = int(in_dims[2])
        self._expected_width = int(in_dims[3])

        self._runtime = str(runtime or model_cfg.get("runtime", "onnxruntime")).lower()
        if self._runtime not in {"onnxruntime", "pytorch"}:
            raise ValueError(f"Unsupported runtime='{self._runtime}'")

        requested_device = str(device or model_cfg.get("device", "cuda")).lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            fallback = str(model_cfg.get("fallback_device", "cpu")).lower()
            self._device = fallback
        else:
            self._device = "cuda:0" if requested_device == "cuda" else requested_device

        self._precision = str(precision or model_cfg.get("precision", "fp32")).lower()
        if self._precision not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported precision='{self._precision}'")
        if self._precision == "fp16" and self._device == "cpu":
            raise ValueError("FP16 inference is only supported on CUDA devices, not CPU.")

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

        # Resolve artifact path based on runtime & precision
        if model_path is not None:
            resolved_path = resolve_project_path(model_path)
        elif self._runtime == "pytorch":
            pt_rel = model_cfg.get("pytorch_checkpoint_path") or model_cfg["model_path"]
            resolved_path = resolve_project_path(pt_rel)
        elif self._runtime == "onnxruntime" and self._precision == "fp16":
            fp16_rel = model_cfg.get("onnx_fp16_path") or model_cfg["model_path"]
            resolved_path = resolve_project_path(fp16_rel)
        else:
            resolved_path = resolve_project_path(model_cfg["model_path"])

        self._model_path = resolved_path
        if not resolved_path.is_file():
            raise MissingModelArtifactError(
                f"Recognition model artifact for '{model_id}' not found at '{resolved_path}'. "
                "Per engineering governance, ImageNet or unverified substitute weights "
                "must NOT be substituted."
            )

        self._model_sha256 = compute_sha256(resolved_path)
        self._parameter_count: int | None = None
        self._ort_session: Any = None
        self._ort_input_name: str = ""
        self._ort_input_dtype = np.float32
        self._ort_output_name: str = ""
        self._torch_model: nn.Module | None = None

        t0 = time.perf_counter()
        if self._runtime == "onnxruntime":
            self._init_onnxruntime_session(resolved_path)
        else:
            self._init_pytorch_model(resolved_path, pytorch_module)
        self._init_time_ms = (time.perf_counter() - t0) * 1000.0

    def _init_onnxruntime_session(self, onnx_path: Path) -> None:
        ensure_cuda_dll_paths()
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        provider = (
            "CUDAExecutionProvider" if self._device.startswith("cuda") else "CPUExecutionProvider"
        )
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(f"Requested ONNX Runtime provider '{provider}' not in {available}")
        if provider == "CUDAExecutionProvider":
            sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            ep_list: list[Any] = [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})
            ]
        else:
            ep_list = ["CPUExecutionProvider"]

        self._ort_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=ep_list,
        )
        in_meta = self._ort_session.get_inputs()[0]
        out_meta = self._ort_session.get_outputs()[0]
        self._ort_input_name = in_meta.name
        self._ort_input_dtype = (
            np.float16 if "float16" in str(in_meta.type).lower() else np.float32
        )
        self._ort_output_name = out_meta.name
        out_shape = out_meta.shape
        if len(out_shape) >= 2 and isinstance(out_shape[-1], int) and out_shape[-1] > 0:
            actual_dim = int(out_shape[-1])
            if actual_dim != self._expected_embedding_dim:
                raise ValueError(
                    f"ONNX output dimension mismatch for '{self._model_id}': "
                    f"expected {self._expected_embedding_dim}, got {actual_dim}"
                )
            self._embedding_dim = actual_dim

    def _init_pytorch_model(self, pt_path: Path, pytorch_module: nn.Module | None) -> None:
        dev = torch.device(self._device)
        if dev.type == "cuda":
            torch.cuda.init()

        arch_lower = self._architecture.lower().replace("-", "_")
        if pytorch_module is not None:
            state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            pytorch_module.load_state_dict(state)
            model: nn.Module = pytorch_module
        elif arch_lower in {"ir_50", "ir50", "adaface_ir50"}:
            model = load_adaface_ir50_checkpoint(
                pt_path,
                input_size=(self._expected_height, self._expected_width),
                output_dim=self._expected_embedding_dim,
                mode="ir",
                strict=True,
            )
        elif arch_lower in {"ir_se_50", "ir_se50", "irse50"}:
            model = load_adaface_ir50_checkpoint(
                pt_path,
                input_size=(self._expected_height, self._expected_width),
                output_dim=self._expected_embedding_dim,
                mode="ir_se",
                strict=True,
            )
        else:
            model = torch.jit.load(str(pt_path), map_location="cpu")

        self._parameter_count = sum(p.numel() for p in model.parameters())
        model.eval()
        model = model.to(dev)
        if self._precision == "fp16":
            model = model.half()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self._torch_model = model

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
    def parameter_count(self) -> int | None:
        return self._parameter_count

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
    def is_synthetic_fixture(self) -> bool:
        return False

    @property
    def initialization_time_ms(self) -> float:
        return self._init_time_ms

    @property
    def provenance(self) -> dict[str, Any]:
        """Structured provenance record for auditability and FaceInstanceResult attachment."""
        return {
            "model_id": self._model_id,
            "model_name": self._model_name,
            "model_version": self._model_version,
            "architecture": self._architecture,
            "runtime": self._runtime,
            "device": self._device,
            "precision": self._precision,
            "model_path": str(self._model_path),
            "sha256": self._model_sha256,
            "parameter_count": self._parameter_count,
            "embedding_dim": self._embedding_dim,
            "licensing_status": dict(self._licensing_status),
        }

    def preprocess(self, aligned_crops_bgr: list[np.ndarray]) -> np.ndarray:
        return self.preprocessor.preprocess(aligned_crops_bgr)

    def infer_preprocessed(self, batch_nchw: np.ndarray) -> np.ndarray:
        if not isinstance(batch_nchw, np.ndarray) or batch_nchw.ndim != 4:
            raise ValueError(f"Expected 4D NCHW array; got {getattr(batch_nchw, 'shape', None)}")

        n, c, h, w = batch_nchw.shape
        if (c, h, w) != (self._expected_channels, self._expected_height, self._expected_width):
            raise ValueError(
                f"Expected preprocessed tensor shape (N, {self._expected_channels}, "
                f"{self._expected_height}, {self._expected_width}); got {batch_nchw.shape}"
            )

        if n == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        outputs: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            chunk = batch_nchw[start : start + self.batch_size]
            if self._runtime == "onnxruntime":
                import onnxruntime as ort

                ort_chunk = np.ascontiguousarray(chunk, dtype=self._ort_input_dtype)
                if self._device.startswith("cuda"):
                    io_binding = self._ort_session.io_binding()
                    ort_in = ort.OrtValue.ortvalue_from_numpy(ort_chunk, "cuda", 0)
                    io_binding.bind_ortvalue_input(self._ort_input_name, ort_in)
                    io_binding.bind_output(self._ort_output_name, "cuda", 0)
                    self._ort_session.run_with_iobinding(io_binding)
                    io_binding.synchronize_outputs()
                    raw_out = io_binding.get_outputs()[0].numpy()
                else:
                    raw_out = self._ort_session.run(
                        [self._ort_output_name], {self._ort_input_name: ort_chunk}
                    )[0]
                outputs.append(np.asarray(raw_out, dtype=np.float32))
            else:
                assert self._torch_model is not None
                dev = torch.device(self._device)
                dtype = torch.float16 if self._precision == "fp16" else torch.float32
                chunk_contig = np.ascontiguousarray(chunk)
                with torch.inference_mode():
                    tensor_in = torch.from_numpy(chunk_contig).to(device=dev, dtype=dtype)
                    tensor_out = self._torch_model(tensor_in)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    outputs.append(tensor_out.detach().float().cpu().numpy())

        combined = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if combined.ndim != 2 or int(combined.shape[1]) != self._expected_embedding_dim:
            raise ValueError(
                f"Output embedding shape validation failed for '{self._model_id}': "
                f"expected ({n}, {self._expected_embedding_dim}), got {combined.shape}"
            )
        self._embedding_dim = int(combined.shape[1])
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


# Canonical P1.1 alias while preserving full P1 backward compatibility
ResNet50Embedder = ReferenceResNet50Embedder


class SyntheticResNet50FixtureEmbedder(FaceEmbedder):
    """Explicitly labelled synthetic fixture for testing adapter mechanics ONLY.

    WARNING: Uses deterministic synthetic weights (`torch.manual_seed`). Outputs are
    NEVER facial biometric embeddings and callers MUST pass `synthetic_input_confirmed=True`.
    """

    def __init__(
        self,
        device: str = "cpu",
        precision: str = "fp32",
        batch_size: int = 8,
        embedding_dim: int = 512,
        l2_normalize: bool = True,
        seed: int = 1234,
    ) -> None:
        self._device = "cuda:0" if device == "cuda" else device
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for SyntheticResNet50FixtureEmbedder, but unavailable.")

        self._precision = precision.lower()
        if self._precision not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported precision='{precision}'")
        if self._precision == "fp16" and self._device == "cpu":
            raise ValueError("FP16 synthetic fixture requires CUDA device.")

        self.batch_size = int(batch_size)
        self._embedding_dim = int(embedding_dim)
        self.default_l2_normalize = bool(l2_normalize)
        self.preprocessor = ResNet50Preprocessor(precision=self._precision)

        t0 = time.perf_counter()
        torch.manual_seed(seed)
        dev = torch.device(self._device)
        if dev.type == "cuda":
            torch.cuda.init()
        probe = (
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, self._embedding_dim),
            )
            .eval()
            .to(dev)
        )
        if self._precision == "fp16":
            probe = probe.half()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self._probe = probe
        self._init_time_ms = (time.perf_counter() - t0) * 1000.0

    @property
    def model_id(self) -> str:
        return "synthetic_resnet50_fixture_not_biometric"

    @property
    def model_version(self) -> str:
        return "synthetic-fixture-0.1"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def is_synthetic_fixture(self) -> bool:
        return True

    @property
    def initialization_time_ms(self) -> float:
        return self._init_time_ms

    def preprocess(self, aligned_crops_bgr: list[np.ndarray]) -> np.ndarray:
        return self.preprocessor.preprocess(aligned_crops_bgr)

    def infer_preprocessed(self, batch_nchw: np.ndarray) -> np.ndarray:
        if batch_nchw.ndim != 4:
            raise ValueError(f"Expected 4D NCHW input; got {batch_nchw.shape}")
        if batch_nchw.shape[0] == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        dev = torch.device(self._device)
        dtype = torch.float16 if self._precision == "fp16" else torch.float32
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, batch_nchw.shape[0], self.batch_size):
                sub = torch.from_numpy(np.ascontiguousarray(batch_nchw[i : i + self.batch_size])).to(
                    device=dev, dtype=dtype
                )
                out = self._probe(sub)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                chunks.append(out.detach().float().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def extract(
        self,
        aligned_crops_bgr: list[np.ndarray],
        l2_normalize: bool | None = None,
        *,
        synthetic_input_confirmed: bool = False,
    ) -> EmbeddingBatchOutput:
        if not synthetic_input_confirmed:
            raise RuntimeError(
                "SyntheticResNet50FixtureEmbedder cannot be invoked without "
                "synthetic_input_confirmed=True. Synthetic outputs must never be "
                "represented as actual facial embeddings."
            )
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
            model_id=self.model_id,
            model_version=self.model_version,
            runtime="pytorch_synthetic_fixture",
            device=self._device,
            precision=self._precision,
            is_synthetic_fixture=True,
            preprocess_time_ms=preprocess_ms,
            inference_time_ms=inference_ms,
            postprocess_time_ms=postprocess_ms,
        )
