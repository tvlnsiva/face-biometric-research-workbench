"""PyTorch-to-ONNX export and numerical parity verification harness (Phase 6).

Preserves a PyTorch reference `nn.Module`, exports to ONNX with dynamic batch axis,
validates graph structure via `onnx.checker.check_model`, and measures numerical
differences (`max_abs_diff`, `mean_abs_diff`, `min_cosine_similarity`) across
ONNX Runtime CPU and CUDA execution providers (with CPU fallback strictly disabled).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn as nn

from src.common.onnx_utils import ensure_cuda_dll_paths
from src.recognition.base_embedder import l2_normalize_embeddings


def export_pytorch_embedder_to_onnx(
    pytorch_model: nn.Module,
    output_onnx_path: str | Path,
    input_shape_chw: tuple[int, int, int] = (3, 112, 112),
    opset_version: int = 17,
) -> Path:
    """Export a reference PyTorch embedding `nn.Module` to ONNX and validate structure."""
    out_path = Path(output_onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cpu_ref = pytorch_model.eval().to("cpu").float()
    dummy_input = torch.randn(1, *input_shape_chw, dtype=torch.float32, device="cpu")

    with torch.inference_mode():
        torch.onnx.export(
            cpu_ref,
            dummy_input,
            str(out_path),
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

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model, full_check=True)
    return out_path


def compare_pytorch_and_onnx_outputs(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    sample_inputs_nchw: np.ndarray,
    provider: str = "CPUExecutionProvider",
) -> dict[str, Any]:
    """Compare reference PyTorch FP32 outputs against ONNX Runtime outputs."""
    ensure_cuda_dll_paths()
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"Requested provider '{provider}' not in {available}")

    x_np = np.ascontiguousarray(sample_inputs_nchw, dtype=np.float32)
    cpu_ref = pytorch_model.eval().to("cpu").float()
    with torch.inference_mode():
        pt_out = cpu_ref(torch.from_numpy(x_np)).detach().cpu().numpy().astype(np.float32)

    sess_options = ort.SessionOptions()
    if provider == "CUDAExecutionProvider":
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=[provider])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    if provider == "CUDAExecutionProvider":
        io_binding = session.io_binding()
        gpu_in = ort.OrtValue.ortvalue_from_numpy(x_np, "cuda", 0)
        io_binding.bind_ortvalue_input(in_name, gpu_in)
        io_binding.bind_output(out_name, "cuda", 0)
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        ort_dev = io_binding.get_outputs()[0].device_name()
        ort_out = io_binding.get_outputs()[0].numpy().astype(np.float32)
    else:
        ort_dev = "cpu"
        ort_out = session.run([out_name], {in_name: x_np})[0].astype(np.float32)

    abs_diff = np.abs(pt_out - ort_out)
    pt_norm = l2_normalize_embeddings(pt_out)
    ort_norm = l2_normalize_embeddings(ort_out)
    cosine_sims = np.sum(pt_norm * ort_norm, axis=1)

    return {
        "provider": provider,
        "executed_on_device": ort_dev,
        "batch_size": int(x_np.shape[0]),
        "embedding_dim": int(pt_out.shape[1]),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "min_cosine_similarity": float(np.min(cosine_sims)),
        "mean_cosine_similarity": float(np.mean(cosine_sims)),
    }
