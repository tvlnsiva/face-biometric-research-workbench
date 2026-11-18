"""ONNX synthetic model construction and Windows CUDA/cuDNN runtime helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def ensure_cuda_dll_paths() -> list[str]:
    """Register bundled PyTorch CUDA/cuDNN DLL directories on Windows if present.

    On Windows, `onnxruntime-gpu` dynamically loads CUDA 12 and cuDNN 9 DLLs.
    When system-wide cuDNN is not installed globally, PyTorch's isolated wheel
    directory (`.venv/Lib/site-packages/torch/lib`) provides compatible CUDA 12.8
    and cuDNN 9 shared libraries without modifying host system directories.
    """
    added: list[str] = []
    if sys.platform != "win32":
        return added

    candidate_dirs: list[Path] = []
    venv_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if venv_torch_lib.is_dir():
        candidate_dirs.append(venv_torch_lib)

    cuda_v12_8_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    if cuda_v12_8_bin.is_dir():
        candidate_dirs.append(cuda_v12_8_bin)

    current_path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for directory in candidate_dirs:
        dir_str = str(directory)
        if dir_str not in current_path_entries:
            os.environ["PATH"] = dir_str + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dir_str)
                added.append(dir_str)
            except OSError:
                pass
    return added


def build_synthetic_onnx_model_bytes(
    channels: int = 3,
    height: int = 112,
    width: int = 112,
    embedding_dim: int = 128,
    seed: int = 42,
) -> bytes:
    """Create and validate a deterministic synthetic ONNX model in memory.

    Graph structure:
      input [batch, C, H, W] -> Conv(3x3, stride=2, 16 filters) -> Relu
      -> GlobalAveragePool -> Flatten -> Gemm(16 -> embedding_dim) -> output [batch, embedding_dim]
    """
    import onnx
    from onnx import TensorProto, checker, helper, numpy_helper

    rng = np.random.default_rng(seed)
    conv_w = rng.standard_normal((16, channels, 3, 3), dtype=np.float32) * np.float32(0.05)
    conv_b = np.zeros((16,), dtype=np.float32)
    gemm_w = rng.standard_normal((embedding_dim, 16), dtype=np.float32) * np.float32(0.05)
    gemm_b = np.zeros((embedding_dim,), dtype=np.float32)

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", channels, height, width]
    )
    output_info = helper.make_tensor_value_info(
        "embedding", TensorProto.FLOAT, ["batch", embedding_dim]
    )

    initializers = [
        numpy_helper.from_array(conv_w, name="conv_w"),
        numpy_helper.from_array(conv_b, name="conv_b"),
        numpy_helper.from_array(gemm_w, name="gemm_w"),
        numpy_helper.from_array(gemm_b, name="gemm_b"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            inputs=["input", "conv_w", "conv_b"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", inputs=["conv_out"], outputs=["relu_out"]),
        helper.make_node("GlobalAveragePool", inputs=["relu_out"], outputs=["gap_out"]),
        helper.make_node("Flatten", inputs=["gap_out"], outputs=["flat_out"], axis=1),
        helper.make_node(
            "Gemm",
            inputs=["flat_out", "gemm_w", "gemm_b"],
            outputs=["embedding"],
            transB=1,
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="workbench_p0_synthetic_probe",
        inputs=[input_info],
        outputs=[output_info],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="workbench_env_validator",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    checker.check_model(model, full_check=True)
    return model.SerializeToString()


def run_ort_session_strict(
    model_bytes: bytes,
    provider: str,
    input_array: np.ndarray,
    intra_op_threads: int = 4,
    inter_op_threads: int = 1,
) -> dict[str, Any]:
    """Run an ONNX Runtime session strictly on the requested provider without silent fallback."""
    ensure_cuda_dll_paths()
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"Requested provider '{provider}' not in available providers: {available}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_op_threads
    sess_options.inter_op_num_threads = inter_op_threads
    if provider == "CUDAExecutionProvider":
        # Strictly forbid silent fallback of nodes to CPUExecutionProvider
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    session = ort.InferenceSession(
        model_bytes,
        sess_options=sess_options,
        providers=[provider],
    )
    active_providers = session.get_providers()

    if provider == "CUDAExecutionProvider":
        io_binding = session.io_binding()
        gpu_input = ort.OrtValue.ortvalue_from_numpy(input_array, "cuda", 0)
        io_binding.bind_ortvalue_input("input", gpu_input)
        io_binding.bind_output("embedding", "cuda", 0)
        session.run_with_iobinding(io_binding)
        ort_outputs = io_binding.get_outputs()
        output_device = ort_outputs[0].device_name()
        output_np = ort_outputs[0].numpy()
    else:
        outputs = session.run(["embedding"], {"input": input_array})
        output_device = "cpu"
        output_np = outputs[0]

    return {
        "requested_provider": provider,
        "active_providers": active_providers,
        "output_device": output_device,
        "output_shape": list(output_np.shape),
        "output_finite": bool(np.all(np.isfinite(output_np))),
        "output_norm": float(np.linalg.norm(output_np)),
    }
