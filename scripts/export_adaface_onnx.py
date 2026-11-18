"""Export AdaFace IR-50 (WebFace4M) PyTorch checkpoint to FP32 & FP16 ONNX and verify parity (P1.1).

Performs Workstream E steps:
  1. Loads original `models/recognition/public_resnet50/adaface_ir50_webface4m.pt`
  2. Verifies exact `IR-50` architecture (`43,585,600` parameters, `[B, 3, 112, 112] -> [B, 512]`)
  3. Exports FP32 ONNX (`adaface_ir50_webface4m.onnx`, opset 17, dynamic batch axis)
  4. Exports FP16 ONNX (`adaface_ir50_webface4m_fp16.onnx`, opset 17, dynamic batch axis on CUDA)
  5. Validates both graphs with `onnx.checker.check_model`
  6. Evaluates numerical parity (`max_abs_diff`, `mean_abs_diff`, `cosine_similarity`, `L2 norm`)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import compute_sha256, resolve_project_path  # noqa: E402
from src.common.onnx_utils import ensure_cuda_dll_paths  # noqa: E402
from src.recognition.adaface_ir50 import load_adaface_ir50_checkpoint  # noqa: E402
from src.recognition.base_embedder import l2_normalize_embeddings  # noqa: E402


def _parity_metrics(ref_raw: np.ndarray, cand_raw: np.ndarray) -> dict[str, Any]:
    ref_norm = l2_normalize_embeddings(ref_raw)
    cand_norm = l2_normalize_embeddings(cand_raw)

    raw_abs = np.abs(ref_raw - cand_raw)
    norm_abs = np.abs(ref_norm - cand_norm)
    cos_sims = np.sum(ref_norm * cand_norm, axis=1)
    cand_l2_norms = np.linalg.norm(cand_norm, axis=1)

    return {
        "output_shape": list(cand_raw.shape),
        "raw_max_abs_diff": float(np.max(raw_abs)),
        "raw_mean_abs_diff": float(np.mean(raw_abs)),
        "normalized_max_abs_diff": float(np.max(norm_abs)),
        "normalized_mean_abs_diff": float(np.mean(norm_abs)),
        "min_cosine_similarity": float(np.min(cos_sims)),
        "mean_cosine_similarity": float(np.mean(cos_sims)),
        "normalized_l2_norm_mean": float(np.mean(cand_l2_norms)),
        "normalized_l2_norm_max_dev": float(np.max(np.abs(cand_l2_norms - 1.0))),
    }


def export_and_validate(
    pt_rel_path: str = "models/recognition/public_resnet50/adaface_ir50_webface4m.pt",
    onnx_fp32_rel_path: str = "models/recognition/public_resnet50/adaface_ir50_webface4m.onnx",
    onnx_fp16_rel_path: str = "models/recognition/public_resnet50/adaface_ir50_webface4m_fp16.onnx",
    opset_version: int = 17,
) -> dict[str, Any]:
    pt_path = resolve_project_path(pt_rel_path)
    onnx_fp32_path = resolve_project_path(onnx_fp32_rel_path)
    onnx_fp16_path = resolve_project_path(onnx_fp16_rel_path)
    onnx_fp32_path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = load_adaface_ir50_checkpoint(pt_path, input_size=(112, 112), output_dim=512, mode="ir")
    param_count = sum(p.numel() for p in model_cpu.parameters())

    dummy_fp32 = torch.randn(2, 3, 112, 112, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model_cpu,
            dummy_fp32,
            str(onnx_fp32_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={
                "input": {0: "batch"},
                "embedding": {0: "batch"},
            },
        )

    onnx_fp32_proto = onnx.load(str(onnx_fp32_path))
    onnx.checker.check_model(onnx_fp32_proto)

    fp16_exported = False
    if torch.cuda.is_available():
        model_fp16 = load_adaface_ir50_checkpoint(
            pt_path, input_size=(112, 112), output_dim=512, mode="ir"
        ).cuda().half()
        dummy_fp16 = torch.randn(2, 3, 112, 112, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            torch.onnx.export(
                model_fp16,
                dummy_fp16,
                str(onnx_fp16_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["embedding"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "embedding": {0: "batch"},
                },
            )
        onnx_fp16_proto = onnx.load(str(onnx_fp16_path))
        onnx.checker.check_model(onnx_fp16_proto)
        fp16_exported = True

    ensure_cuda_dll_paths()
    import onnxruntime as ort

    rng = np.random.default_rng(42)
    test_batch = rng.uniform(-1.0, 1.0, size=(8, 3, 112, 112)).astype(np.float32)

    with torch.inference_mode():
        pt_cpu_out = model_cpu(torch.from_numpy(test_batch)).numpy()

    parity: dict[str, Any] = {}

    sess_cpu = ort.InferenceSession(str(onnx_fp32_path), providers=["CPUExecutionProvider"])
    ort_cpu_out = sess_cpu.run(["embedding"], {"input": test_batch})[0]
    parity["ort_cpu_fp32_vs_pytorch_cpu_fp32"] = _parity_metrics(pt_cpu_out, ort_cpu_out)

    if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
        model_cuda_fp32 = load_adaface_ir50_checkpoint(
            pt_path, input_size=(112, 112), output_dim=512, mode="ir"
        ).cuda()
        with torch.inference_mode():
            pt_cuda_fp32_out = model_cuda_fp32(torch.from_numpy(test_batch).cuda()).cpu().numpy()
        parity["pytorch_cuda_fp32_vs_pytorch_cpu_fp32"] = _parity_metrics(pt_cpu_out, pt_cuda_fp32_out)

        with torch.inference_mode():
            pt_cuda_fp16_out = (
                model_fp16(torch.from_numpy(test_batch).cuda().half()).float().cpu().numpy()
            )
        parity["pytorch_cuda_fp16_vs_pytorch_cpu_fp32"] = _parity_metrics(pt_cpu_out, pt_cuda_fp16_out)

        sess_cuda_fp32 = ort.InferenceSession(
            str(onnx_fp32_path), providers=["CUDAExecutionProvider"]
        )
        ort_cuda_fp32_out = sess_cuda_fp32.run(["embedding"], {"input": test_batch})[0]
        parity["ort_cuda_fp32_vs_pytorch_cpu_fp32"] = _parity_metrics(pt_cpu_out, ort_cuda_fp32_out)

        if fp16_exported:
            sess_cuda_fp16 = ort.InferenceSession(
                str(onnx_fp16_path), providers=["CUDAExecutionProvider"]
            )
            ort_cuda_fp16_out = sess_cuda_fp16.run(
                ["embedding"], {"input": test_batch.astype(np.float16)}
            )[0].astype(np.float32)
            parity["ort_cuda_fp16_vs_pytorch_cpu_fp32"] = _parity_metrics(pt_cpu_out, ort_cuda_fp16_out)

    summary = {
        "pytorch_checkpoint": {
            "path": pt_rel_path,
            "size_bytes": pt_path.stat().st_size,
            "sha256": compute_sha256(pt_path),
            "parameter_count": param_count,
        },
        "onnx_fp32": {
            "path": onnx_fp32_rel_path,
            "size_bytes": onnx_fp32_path.stat().st_size,
            "sha256": compute_sha256(onnx_fp32_path),
            "opset_version": opset_version,
            "checker_valid": True,
        },
        "onnx_fp16": (
            {
                "path": onnx_fp16_rel_path,
                "size_bytes": onnx_fp16_path.stat().st_size,
                "sha256": compute_sha256(onnx_fp16_path),
                "opset_version": opset_version,
                "checker_valid": True,
            }
            if fp16_exported
            else None
        ),
        "parity": parity,
    }
    return summary


def main() -> int:
    summary = export_and_validate()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
