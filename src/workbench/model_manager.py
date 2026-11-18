"""On-Demand Model Lifecycle & Hardware Resource Manager (`src/workbench/model_manager.py`).

Designed for the Windows 11 + NVIDIA RTX 3060 Laptop GPU (`6,144 MiB` VRAM, `~16 GB` RAM) budget:
  - Loads zero models at server startup.
  - Instantiates detectors and recognizers lazily on first use.
  - Bounds concurrent GPU sessions (`max_gpu_detectors=1`, `max_gpu_recognizers=2`) and auto-evicts
    LRU GPU sessions when exceeded or when free GPU VRAM drops below `1,200 MiB`.
  - Exposes explicit load / unload / release-all operations and real-time CPU/RAM/GPU telemetry.
"""

from __future__ import annotations

import gc
import os
import platform
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import onnxruntime as ort
import psutil
import torch

from src.detection.scrfd_detector import SCRFDDetector
from src.detection.yunet_detector import YuNetDetector
from src.p3_research.algorithmic_variants import ResearchSCRFDWrapper
from src.recognition.base_embedder import MissingModelArtifactError
from src.recognition.candidate_embedders import MobileFaceNetEmbedder, SFaceEmbedder
from src.recognition.resnet50_embedder import ReferenceResNet50Embedder
from src.workbench.contracts import WorkbenchError


@dataclass
class LoadedModelEntry:
    """Metadata and live handle for an instantiated detector or recognizer session."""

    cache_key: str
    model_id: str
    category: str
    runtime: str
    device: str
    precision: str
    instance: Any
    init_time_ms: float
    loaded_at_utc: str
    last_used_epoch: float
    model_sha256: str | None

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.cache_key,
            "model_id": self.model_id,
            "category": self.category,
            "runtime": self.runtime,
            "device": self.device,
            "precision": self.precision,
            "init_time_ms": round(self.init_time_ms, 2),
            "loaded_at_utc": self.loaded_at_utc,
            "last_used_seconds_ago": round(max(0.0, time.time() - self.last_used_epoch), 1),
            "model_sha256": self.model_sha256,
        }


class ModelResourceManager:
    """Thread-safe lazy loader and GPU VRAM budget manager for research models."""

    def __init__(
        self,
        max_gpu_detectors: int = 1,
        max_gpu_recognizers: int = 2,
        min_free_vram_mib: float = 1200.0,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, LoadedModelEntry] = OrderedDict()
        self.max_gpu_detectors = int(max_gpu_detectors)
        self.max_gpu_recognizers = int(max_gpu_recognizers)
        self.min_free_vram_mib = float(min_free_vram_mib)
        self._process = psutil.Process(os.getpid())

    @staticmethod
    def _utc_now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _normalize_device(device: str | None, default_gpu: bool = True) -> str:
        if device is None:
            return "cuda" if (default_gpu and torch.cuda.is_available()) else "cpu"
        d = str(device).lower().strip()
        if d.startswith("cuda"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if d == "cpu":
            return "cpu"
        raise WorkbenchError(f"Unsupported execution device '{device}'.", code="INVALID_DEVICE")

    def get_hardware_snapshot(self) -> dict[str, Any]:
        """Collect current CPU, system RAM, process RSS, and CUDA/GPU memory metrics."""
        vm = psutil.virtual_memory()
        proc_rss_mib = float(self._process.memory_info().rss) / (1024.0 * 1024.0)

        gpu_info: dict[str, Any] = {
            "available": bool(torch.cuda.is_available()),
            "name": None,
            "vram_total_mib": None,
            "vram_free_mib": None,
            "vram_device_used_mib": None,
            "torch_allocated_mib": None,
            "torch_reserved_mib": None,
        }
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                total_mib = float(props.total_memory) / (1024.0 * 1024.0)
                free_b, total_b = torch.cuda.mem_get_info(0)
                free_mib = float(free_b) / (1024.0 * 1024.0)
                used_mib = float(total_b - free_b) / (1024.0 * 1024.0)
                gpu_info.update(
                    {
                        "name": props.name,
                        "vram_total_mib": round(total_mib, 1),
                        "vram_free_mib": round(free_mib, 1),
                        "vram_device_used_mib": round(used_mib, 1),
                        "torch_allocated_mib": round(float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0), 2),
                        "torch_reserved_mib": round(float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0), 2),
                    }
                )
            except Exception:  # noqa: BLE001
                gpu_info["name"] = torch.cuda.get_device_name(0)

        with self._lock:
            loaded_list = [e.summary() for e in self._entries.values()]

        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_total_mib": round(float(vm.total) / (1024.0 * 1024.0), 1),
            "ram_used_mib": round(float(vm.used) / (1024.0 * 1024.0), 1),
            "ram_available_mib": round(float(vm.available) / (1024.0 * 1024.0), 1),
            "process_rss_mib": round(proc_rss_mib, 1),
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "onnxruntime_providers": ort.get_available_providers(),
            "gpu": gpu_info,
            "loaded_models": loaded_list,
        }

    def _flush_gpu_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def _evict_gpu_if_needed(self, incoming_category: str, incoming_device: str) -> list[str]:
        evicted: list[str] = []
        if not incoming_device.startswith("cuda"):
            return evicted

        cap = self.max_gpu_detectors if incoming_category == "detection" else self.max_gpu_recognizers
        gpu_same_cat = [
            k for k, e in self._entries.items() if e.category == incoming_category and e.device.startswith("cuda")
        ]
        while len(gpu_same_cat) >= cap:
            oldest_key = gpu_same_cat.pop(0)
            entry = self._entries.pop(oldest_key, None)
            if entry is not None:
                del entry.instance
                evicted.append(oldest_key)

        if torch.cuda.is_available():
            try:
                free_b, _ = torch.cuda.mem_get_info(0)
                free_mib = float(free_b) / (1024.0 * 1024.0)
                if free_mib < self.min_free_vram_mib:
                    for k in list(self._entries.keys()):
                        if self._entries[k].device.startswith("cuda"):
                            ent = self._entries.pop(k)
                            del ent.instance
                            evicted.append(k)
                            self._flush_gpu_cache()
                            free_b2, _ = torch.cuda.mem_get_info(0)
                            if float(free_b2) / (1024.0 * 1024.0) >= self.min_free_vram_mib:
                                break
            except Exception:  # noqa: BLE001
                pass

        if evicted:
            self._flush_gpu_cache()
        return evicted

    def get_detector(self, detector_id: str, device: str | None = None) -> LoadedModelEntry:
        """Retrieve or lazily load a face detector (`yunet_2023mar` or `scrfd_500m_research`)."""
        with self._lock:
            if detector_id == "yunet_2023mar":
                eff_dev = "cpu"
                runtime = "opencv_dnn"
                precision = "fp32"
            elif detector_id == "scrfd_500m_research":
                eff_dev = self._normalize_device(device, default_gpu=True)
                runtime = "onnxruntime"
                precision = "fp32"
            else:
                raise WorkbenchError(f"Unknown detector_id='{detector_id}'.", code="UNKNOWN_DETECTOR")

            key = f"det::{detector_id}::{runtime}::{eff_dev}::{precision}"
            if key in self._entries:
                ent = self._entries[key]
                ent.last_used_epoch = time.time()
                self._entries.move_to_end(key)
                return ent

            self._evict_gpu_if_needed("detection", eff_dev)

            t0 = time.perf_counter()
            if detector_id == "yunet_2023mar":
                inst: Any = YuNetDetector(config_path="configs/models.yaml")
                sha = None
                from src.common.config_loader import compute_sha256

                sha = compute_sha256(inst.model_path)
            else:
                base_scrfd = SCRFDDetector(device=eff_dev, config_path="configs/p2_models.yaml")
                inst = ResearchSCRFDWrapper(base_scrfd)
                sha = base_scrfd.model_sha256
            init_ms = (time.perf_counter() - t0) * 1000.0

            entry = LoadedModelEntry(
                cache_key=key,
                model_id=detector_id,
                category="detection",
                runtime=runtime,
                device=eff_dev,
                precision=precision,
                instance=inst,
                init_time_ms=init_ms,
                loaded_at_utc=self._utc_now(),
                last_used_epoch=time.time(),
                model_sha256=sha,
            )
            self._entries[key] = entry
            return entry

    def get_recognizer(
        self,
        recognizer_id: str,
        runtime: str | None = None,
        device: str | None = None,
        precision: str | None = None,
    ) -> LoadedModelEntry:
        """Retrieve or lazily load a face recognition embedder."""
        with self._lock:
            if recognizer_id == "adaface_ir50_webface4m":
                eff_rt = str(runtime or "onnxruntime").lower()
                eff_dev = self._normalize_device(device, default_gpu=True)
                eff_prec = str(precision or ("fp16" if eff_dev.startswith("cuda") else "fp32")).lower()
                if eff_prec == "fp16" and eff_dev == "cpu":
                    raise WorkbenchError(
                        "FP16 precision is only supported on CUDA devices; select `fp32` for CPU.",
                        code="UNSUPPORTED_PRECISION_ON_CPU",
                    )
            elif recognizer_id == "mobilefacenet_w600k_research":
                eff_rt = "onnxruntime"
                eff_dev = self._normalize_device(device, default_gpu=True)
                eff_prec = "fp32"
                if precision and str(precision).lower() != "fp32":
                    raise WorkbenchError(
                        "MobileFaceNet W600K distributes FP32 ONNX weights only.",
                        code="UNSUPPORTED_PRECISION",
                    )
            elif recognizer_id == "sface_2021dec":
                eff_rt = str(runtime or "onnxruntime").lower()
                if eff_rt == "opencv_dnn":
                    if device and str(device).lower().startswith("cuda"):
                        raise WorkbenchError(
                            "OpenCV FaceRecognizerSF (`opencv_dnn`) supports CPU execution only.",
                            code="UNSUPPORTED_DEVICE",
                        )
                    eff_dev = "cpu"
                else:
                    eff_dev = self._normalize_device(device, default_gpu=True)
                eff_prec = "fp32"
            elif recognizer_id == "reference_resnet50_baseline":
                try:
                    _ = ReferenceResNet50Embedder(model_id="reference_resnet50_baseline", config_path="configs/models.yaml")
                except MissingModelArtifactError as exc:
                    raise WorkbenchError(
                        str(exc),
                        code="MISSING_MODEL_WEIGHTS",
                        status_code=422,
                        details={"model_id": recognizer_id, "status": "BLOCKED_MISSING_REFERENCE_RESNET50_WEIGHTS"},
                    ) from exc
                raise WorkbenchError("Unexpected state for retired baseline.", code="MODEL_ERROR")
            else:
                raise WorkbenchError(f"Unknown recognizer_id='{recognizer_id}'.", code="UNKNOWN_RECOGNIZER")

            key = f"rec::{recognizer_id}::{eff_rt}::{eff_dev}::{eff_prec}"
            if key in self._entries:
                ent = self._entries[key]
                ent.last_used_epoch = time.time()
                self._entries.move_to_end(key)
                return ent

            self._evict_gpu_if_needed("recognition", eff_dev)

            t0 = time.perf_counter()
            if recognizer_id == "adaface_ir50_webface4m":
                inst: Any = ReferenceResNet50Embedder(
                    model_id="adaface_ir50_webface4m",
                    runtime=eff_rt,
                    device=eff_dev,
                    precision=eff_prec,
                    batch_size=8,
                    config_path="configs/p2_models.yaml",
                )
            elif recognizer_id == "mobilefacenet_w600k_research":
                inst = MobileFaceNetEmbedder(
                    device=eff_dev,
                    batch_size=8,
                    config_path="configs/p2_models.yaml",
                )
            else:
                inst = SFaceEmbedder(
                    runtime=eff_rt,
                    device=eff_dev,
                    config_path="configs/p2_models.yaml",
                )
            init_ms = (time.perf_counter() - t0) * 1000.0

            entry = LoadedModelEntry(
                cache_key=key,
                model_id=recognizer_id,
                category="recognition",
                runtime=eff_rt,
                device=eff_dev,
                precision=eff_prec,
                instance=inst,
                init_time_ms=init_ms,
                loaded_at_utc=self._utc_now(),
                last_used_epoch=time.time(),
                model_sha256=getattr(inst, "model_sha256", None),
            )
            self._entries[key] = entry
            return entry

    def unload(self, cache_key: str | None = None, unload_all: bool = False) -> list[str]:
        """Release one or all cached model sessions and reclaim GPU/CPU memory."""
        removed: list[str] = []
        with self._lock:
            if unload_all:
                for k in list(self._entries.keys()):
                    ent = self._entries.pop(k)
                    del ent.instance
                    removed.append(k)
            elif cache_key and cache_key in self._entries:
                ent = self._entries.pop(cache_key)
                del ent.instance
                removed.append(cache_key)
        if removed:
            self._flush_gpu_cache()
        return removed
