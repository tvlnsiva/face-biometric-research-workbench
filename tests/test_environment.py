"""Automated tests for Python environment isolation and pinned package imports."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import pytest

from src.common.onnx_utils import ensure_cuda_dll_paths

REQUIRED_IMPORTS = [
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "PIL",
    "yaml",
    "tqdm",
    "psutil",
    "pytest",
    "matplotlib",
    "seaborn",
]


def test_python_version_is_311() -> None:
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11.x, got {sys.version}"


def test_running_inside_isolated_virtualenv() -> None:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    assert sys.prefix != base_prefix, "Interpreter is not running inside an isolated virtual environment"


@pytest.mark.parametrize("module_name", REQUIRED_IMPORTS)
def test_required_package_imports(module_name: str) -> None:
    ensure_cuda_dll_paths()
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_no_simultaneous_ort_cpu_and_gpu_packages() -> None:
    installed = []
    for dist_name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            importlib.metadata.version(dist_name)
            installed.append(dist_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    assert len(installed) == 1, (
        f"Exactly one of onnxruntime or onnxruntime-gpu must be installed; found: {installed}"
    )
