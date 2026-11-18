"""Face Biometric Research Workbench — Environment Validation Script (Milestone P0).

Executes end-to-end validation of Python, virtual environment isolation, pinned
package imports/versions, PyTorch CPU & CUDA execution, ONNX graph creation &
checker validation, ONNX Runtime CPU & CUDA execution (with strict CPU-fallback
prohibition and CUDA IO-binding verification), and OpenCV + YuNet model loading.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is importable when invoked as `python scripts/validate_environment.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import (  # noqa: E402
    compute_sha256,
    load_model_manifest,
    load_yaml_config,
    resolve_project_path,
)
from src.common.onnx_utils import (  # noqa: E402
    build_synthetic_onnx_model_bytes,
    ensure_cuda_dll_paths,
    run_ort_session_strict,
)

REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime-gpu"),  # falls back to checking "onnxruntime" distribution metadata
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("psutil", "psutil"),
    ("pytest", "pytest"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]


def _resolve_dist_version(dist_name: str, module_obj: Any) -> str:
    candidates = [dist_name]
    if dist_name == "onnxruntime-gpu":
        candidates.append("onnxruntime")
    for name in candidates:
        try:
            return f"{name}=={importlib.metadata.version(name)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    return getattr(module_obj, "__version__", "unknown")


def validate_python_environment() -> dict[str, Any]:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    expected_venv = (PROJECT_ROOT / ".venv").resolve()
    actual_prefix = Path(sys.prefix).resolve()

    imported_packages: dict[str, str] = {}
    failed_imports: dict[str, str] = {}

    ensure_cuda_dll_paths()

    for import_name, dist_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            imported_packages[import_name] = _resolve_dist_version(dist_name, mod)
        except Exception as exc:  # noqa: BLE001
            failed_imports[import_name] = f"{type(exc).__name__}: {exc}"

    # Verify ONNX Runtime CPU & GPU packages are not installed simultaneously
    installed_ort_dists: list[str] = []
    for ort_dist in ("onnxruntime", "onnxruntime-gpu"):
        try:
            ver = importlib.metadata.version(ort_dist)
            installed_ort_dists.append(f"{ort_dist}=={ver}")
        except importlib.metadata.PackageNotFoundError:
            pass

    simultaneous_ort_conflict = len(installed_ort_dists) > 1

    passed = (
        sys.version_info[:2] == (3, 11)
        and in_venv
        and len(failed_imports) == 0
        and not simultaneous_ort_conflict
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "in_virtualenv": in_venv,
        "venv_prefix": str(actual_prefix),
        "matches_project_venv": actual_prefix == expected_venv,
        "installed_ort_distributions": installed_ort_dists,
        "simultaneous_ort_conflict": simultaneous_ort_conflict,
        "imported_packages": imported_packages,
        "failed_imports": failed_imports,
    }


def validate_pytorch() -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_compiled_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }

    # CPU tensor operation
    torch.manual_seed(42)
    cpu_a = torch.randn(256, 256, dtype=torch.float32, device="cpu")
    cpu_b = torch.randn(256, 256, dtype=torch.float32, device="cpu")
    cpu_out = cpu_a @ cpu_b
    cpu_ok = (
        cpu_out.device.type == "cpu"
        and tuple(cpu_out.shape) == (256, 256)
        and bool(torch.isfinite(cpu_out).all().item())
    )
    result["cpu_execution"] = {
        "status": "PASS" if cpu_ok else "FAIL",
        "device": str(cpu_out.device),
        "output_shape": list(cpu_out.shape),
        "output_norm": float(torch.linalg.norm(cpu_out).item()),
    }

    # CUDA tensor operation
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        torch.cuda.reset_peak_memory_stats(0)
        gpu_a = cpu_a.to("cuda:0")
        gpu_b = cpu_b.to("cuda:0")
        gpu_out = gpu_a @ gpu_b
        torch.cuda.synchronize(0)

        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(0))
        reserved_bytes = int(torch.cuda.memory_reserved(0))
        max_abs_diff = float((gpu_out.cpu() - cpu_out).abs().max().item())
        actually_on_gpu = (
            gpu_out.is_cuda
            and gpu_out.device.type == "cuda"
            and peak_allocated_bytes > 0
            and max_abs_diff < 1e-2
        )
        result["gpu_device"] = {
            "index": 0,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_mib": round(props.total_memory / (1024 * 1024), 2),
            "peak_allocated_bytes_during_op": peak_allocated_bytes,
            "reserved_bytes_after_op": reserved_bytes,
        }
        result["cuda_execution"] = {
            "status": "PASS" if actually_on_gpu else "FAIL",
            "executed_on_gpu": actually_on_gpu,
            "tensor_device": str(gpu_out.device),
            "max_abs_diff_vs_cpu": max_abs_diff,
            "output_norm": float(torch.linalg.norm(gpu_out).item()),
        }
        overall_ok = cpu_ok and actually_on_gpu
    else:
        result["gpu_device"] = None
        result["cuda_execution"] = {
            "status": "NOT_AVAILABLE",
            "executed_on_gpu": False,
            "reason": "torch.cuda.is_available() returned False",
        }
        overall_ok = cpu_ok

    result["status"] = "PASS" if overall_ok else "FAIL"
    return result


def validate_onnx_and_ort() -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import onnx

    ensure_cuda_dll_paths()
    import onnxruntime as ort

    model_bytes = build_synthetic_onnx_model_bytes(
        channels=3, height=112, width=112, embedding_dim=128, seed=42
    )
    parsed_model = onnx.load_from_string(model_bytes)
    onnx.checker.check_model(parsed_model, full_check=True)

    onnx_report = {
        "status": "PASS",
        "onnx_version": onnx.__version__,
        "ir_version": int(parsed_model.ir_version),
        "opset_version": int(parsed_model.opset_import[0].version),
        "serialized_bytes": len(model_bytes),
        "node_op_types": [node.op_type for node in parsed_model.graph.node],
    }

    rng = np.random.default_rng(123)
    sample_input = rng.standard_normal((2, 3, 112, 112), dtype=np.float32)

    providers = ort.get_available_providers()
    cpu_res = run_ort_session_strict(model_bytes, "CPUExecutionProvider", sample_input)
    cpu_ok = cpu_res["output_finite"] and cpu_res["output_shape"] == [2, 128]

    ort_report: dict[str, Any] = {
        "ort_version": ort.__version__,
        "ort_device_discovery": ort.get_device(),
        "available_providers": providers,
        "cpu_inference": {
            "status": "PASS" if cpu_ok else "FAIL",
            **cpu_res,
        },
    }

    if "CUDAExecutionProvider" in providers:
        try:
            cuda_res = run_ort_session_strict(model_bytes, "CUDAExecutionProvider", sample_input)
            actually_cuda = (
                cuda_res["output_finite"]
                and cuda_res["output_shape"] == [2, 128]
                and cuda_res["output_device"].lower() == "cuda"
                and "CUDAExecutionProvider" in cuda_res["active_providers"]
            )
            ort_report["cuda_inference"] = {
                "status": "PASS" if actually_cuda else "FAIL",
                "executed_on_gpu": actually_cuda,
                "cpu_fallback_disabled": True,
                **cuda_res,
            }
            ort_ok = cpu_ok and actually_cuda
        except Exception as exc:  # noqa: BLE001
            ort_report["cuda_inference"] = {
                "status": "FAIL",
                "executed_on_gpu": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            ort_ok = False
    else:
        ort_report["cuda_inference"] = {
            "status": "NOT_AVAILABLE",
            "executed_on_gpu": False,
            "reason": "CUDAExecutionProvider not present in ort.get_available_providers()",
        }
        ort_ok = cpu_ok

    ort_report["status"] = "PASS" if ort_ok else "FAIL"
    return onnx_report, ort_report


def validate_opencv_and_models() -> dict[str, Any]:
    import cv2
    import numpy as np

    models_cfg = load_yaml_config("configs/models.yaml")
    manifest = load_model_manifest(models_cfg["manifest_path"])

    yunet_cfg = models_cfg["models"]["yunet_2023mar"]
    yunet_path = resolve_project_path(yunet_cfg["model_path"])

    yunet_report: dict[str, Any]
    if yunet_path.is_file():
        detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        blank_img = np.zeros((320, 320, 3), dtype=np.uint8)
        retval, faces = detector.detect(blank_img)
        yunet_report = {
            "status": "PASS" if retval == 1 else "FAIL",
            "model_path": yunet_cfg["model_path"],
            "file_size_bytes": yunet_path.stat().st_size,
            "sha256": compute_sha256(yunet_path),
            "detect_retval": int(retval),
            "blank_image_detections": 0 if faces is None else int(len(faces)),
        }
    else:
        yunet_report = {
            "status": "NOT_AVAILABLE",
            "model_path": yunet_cfg["model_path"],
            "reason": "YuNet ONNX artifact not found on disk",
        }

    # Verify SHA-256 integrity of all downloaded models in manifest
    manifest_checks: list[dict[str, Any]] = []
    manifest_ok = True
    for entry in manifest["models"]:
        rel_path = entry["local_path"]
        abs_path = resolve_project_path(rel_path)
        if entry["status"].startswith("downloaded"):
            exists = abs_path.is_file()
            actual_hash = compute_sha256(abs_path) if exists else None
            hash_match = exists and (actual_hash == entry["sha256"])
            if not hash_match:
                manifest_ok = False
            manifest_checks.append(
                {
                    "model_id": entry["model_id"],
                    "status": "PASS" if hash_match else "FAIL",
                    "exists": exists,
                    "sha256_match": hash_match,
                    "commercial_use_status": entry["commercial_use_status"],
                    "production_eligible": entry["production_eligible"],
                }
            )
        else:
            manifest_checks.append(
                {
                    "model_id": entry["model_id"],
                    "status": "NOT_AVAILABLE",
                    "exists": abs_path.is_file(),
                    "reason": entry["status"],
                    "commercial_use_status": entry["commercial_use_status"],
                    "production_eligible": entry["production_eligible"],
                }
            )

    overall_ok = yunet_report["status"] == "PASS" and manifest_ok
    return {
        "status": "PASS" if overall_ok else "FAIL",
        "opencv_version": cv2.__version__,
        "yunet_loading": yunet_report,
        "manifest_integrity": manifest_checks,
    }


def main() -> int:
    print("=" * 76)
    print("Face Biometric Research Workbench — Environment Validation (Milestone P0)")
    print("=" * 76)

    py_res = validate_python_environment()
    print(f"[1/4] Python & Virtualenv  : {py_res['status']} (Python {py_res['python_version']}, venv={py_res['in_virtualenv']})")

    torch_res = validate_pytorch()
    print(
        f"[2/4] PyTorch CPU & CUDA   : {torch_res['status']} "
        f"(CPU={torch_res['cpu_execution']['status']}, "
        f"CUDA={torch_res['cuda_execution']['status']}, "
        f"executed_on_gpu={torch_res['cuda_execution']['executed_on_gpu']})"
    )

    onnx_res, ort_res = validate_onnx_and_ort()
    print(
        f"[3/4] ONNX & ONNX Runtime  : {ort_res['status']} "
        f"(ONNX={onnx_res['status']}, ORT_CPU={ort_res['cpu_inference']['status']}, "
        f"ORT_CUDA={ort_res['cuda_inference']['status']}, "
        f"executed_on_gpu={ort_res['cuda_inference']['executed_on_gpu']})"
    )

    cv_res = validate_opencv_and_models()
    print(
        f"[4/4] OpenCV & Model Hashes: {cv_res['status']} "
        f"(OpenCV {cv_res['opencv_version']}, YuNet={cv_res['yunet_loading']['status']})"
    )

    gpu_summary = {
        "gpu_detected": bool(torch_res["cuda_available"]),
        "gpu_name": torch_res["gpu_device"]["name"] if torch_res["gpu_device"] else None,
        "pytorch_executed_on_gpu": bool(torch_res["cuda_execution"]["executed_on_gpu"]),
        "onnxruntime_executed_on_gpu": bool(ort_res["cuda_inference"]["executed_on_gpu"]),
        "verified_gpu_execution": bool(
            torch_res["cuda_execution"]["executed_on_gpu"]
            and ort_res["cuda_inference"]["executed_on_gpu"]
        ),
    }

    all_passed = all(
        section["status"] == "PASS"
        for section in (py_res, torch_res, onnx_res, ort_res, cv_res)
    )

    full_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if all_passed else "FAIL",
        "python": py_res,
        "pytorch": torch_res,
        "onnx": onnx_res,
        "onnxruntime": ort_res,
        "opencv_and_models": cv_res,
        "gpu_summary": gpu_summary,
    }

    output_path = resolve_project_path("reports/environment_validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    print("-" * 76)
    print(f"GPU Execution Verified     : {gpu_summary['verified_gpu_execution']} ({gpu_summary['gpu_name']})")
    print(f"Overall Validation Status  : {full_report['overall_status']}")
    print(f"Saved Validation JSON      : {output_path}")
    print("=" * 76)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
