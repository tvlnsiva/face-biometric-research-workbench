"""Automated tests for ONNX synthetic graph validation and ONNX Runtime CPU/CUDA execution."""

from __future__ import annotations

import numpy as np
import onnx
import pytest

from src.common.onnx_utils import (
    build_synthetic_onnx_model_bytes,
    ensure_cuda_dll_paths,
    run_ort_session_strict,
)

ensure_cuda_dll_paths()
import onnxruntime as ort  # noqa: E402


def test_onnx_synthetic_model_creation_and_validation() -> None:
    model_bytes = build_synthetic_onnx_model_bytes(
        channels=3, height=112, width=112, embedding_dim=128, seed=7
    )
    assert len(model_bytes) > 0
    model = onnx.load_from_string(model_bytes)
    onnx.checker.check_model(model, full_check=True)
    assert [n.op_type for n in model.graph.node] == [
        "Conv",
        "Relu",
        "GlobalAveragePool",
        "Flatten",
        "Gemm",
    ]


def test_onnxruntime_cpu_inference() -> None:
    model_bytes = build_synthetic_onnx_model_bytes()
    rng = np.random.default_rng(99)
    sample = rng.standard_normal((2, 3, 112, 112), dtype=np.float32)

    res = run_ort_session_strict(model_bytes, "CPUExecutionProvider", sample)
    assert res["output_device"] == "cpu"
    assert res["output_shape"] == [2, 128]
    assert res["output_finite"] is True


@pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="CUDAExecutionProvider is not available in this ONNX Runtime installation",
)
def test_onnxruntime_cuda_inference_executes_on_gpu() -> None:
    model_bytes = build_synthetic_onnx_model_bytes()
    rng = np.random.default_rng(99)
    sample = rng.standard_normal((2, 3, 112, 112), dtype=np.float32)

    cpu_res = run_ort_session_strict(model_bytes, "CPUExecutionProvider", sample)
    cuda_res = run_ort_session_strict(model_bytes, "CUDAExecutionProvider", sample)

    assert "CUDAExecutionProvider" in cuda_res["active_providers"]
    assert cuda_res["output_device"].lower() == "cuda"
    assert cuda_res["output_shape"] == [2, 128]
    assert cuda_res["output_finite"] is True
    assert abs(cuda_res["output_norm"] - cpu_res["output_norm"]) < 1e-3
