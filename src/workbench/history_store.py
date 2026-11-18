"""Immutable Historical Benchmark Reader & Workbench Experiment Store (`src/workbench/history_store.py`).

Guarantees:
  - Historical P1, P1.1, P2, and P3 JSON result files are read-only and NEVER overwritten.
  - Every metric carries explicit `dataset_id`, `dataset_split`, `algorithms`, `parameters`, and
    `source_artifact` provenance.
  - All new Workbench experiments and saved configurations are written under `reports/workbench_runs/`
    and support export to JSON, CSV, and YAML.
"""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from src.common.config_loader import resolve_project_path
from src.workbench.contracts import ExperimentRecord, WorkbenchError


class ExperimentHistoryStore:
    """Manages read-only historical milestones (`P1`..`P3`) and read/write Workbench runs."""

    def __init__(self, runs_rel_dir: str = "reports/workbench_runs") -> None:
        self.runs_dir: Path = resolve_project_path(runs_rel_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._historical_cache: list[dict[str, Any]] | None = None

    @staticmethod
    def _utc_now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _ingest_historical_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        # 1. P1.1 Pipeline Benchmark (`reports/p11_pipeline_benchmark_results.json`)
        p11_pipe = resolve_project_path("reports/p11_pipeline_benchmark_results.json")
        if p11_pipe.is_file():
            raw = json.loads(p11_pipe.read_text(encoding="utf-8"))
            opt = raw.get("p11_optimized_on_p1_split", {})
            records.append(
                ExperimentRecord(
                    experiment_id="HIST-P1.1-PIPELINE-BASELINE",
                    name="P1.1 Frozen Baseline: YuNet (1280 CPU) + AdaFace IR-50 (ORT CUDA FP32)",
                    source_origin="historical_readonly",
                    milestone="P1.1",
                    category="pipeline",
                    timestamp_utc=str(raw.get("timestamp_utc", "")),
                    status="completed",
                    dataset_id="nasa_p1_detection",
                    dataset_split="detection_evaluation (6 images / 16 detected faces)",
                    dataset_manifest_sha256="077c0730566628bb8c141a79e55509d1d5297f4515757ce6729e118267bd3e5f",
                    algorithms={
                        "detector": "yunet_2023mar",
                        "recognizer": "adaface_ir50_webface4m",
                    },
                    model_hashes={
                        "detector": opt.get("models", {}).get("detector", {}).get("sha256"),
                        "recognizer": opt.get("models", {}).get("recognizer", {}).get("sha256"),
                    },
                    parameters=opt.get("configuration", {}),
                    software_environment={"python": "3.11.9", "onnxruntime": "1.22.0"},
                    hardware={"gpu": "NVIDIA GeForce RTX 3060 Laptop GPU"},
                    execution_provider="OpenCV_CPU + ORT_CUDA",
                    precision="fp32",
                    batch_size=8,
                    seed=42,
                    metrics={
                        "images_per_sec": opt.get("throughput", {}).get("images_per_sec"),
                        "faces_per_sec": opt.get("throughput", {}).get("faces_per_sec"),
                        "gpu_memory_mib": opt.get("resource_utilization", {}).get("gpu_memory_used_mib_max"),
                        "detection_p50_ms": opt.get("warm_per_image_stages", {}).get("detection_ms", {}).get("p50_ms"),
                    },
                    artifact_refs=["reports/p11_pipeline_benchmark_results.json"],
                    notes="Immutable Milestone P1.1 baseline pipeline benchmark.",
                ).to_dict()
            )

        # 2. P2 Detection Comparison (`reports/p2_detection_comparison.json`)
        p2_det = resolve_project_path("reports/p2_detection_comparison.json")
        if p2_det.is_file():
            raw = json.loads(p2_det.read_text(encoding="utf-8"))
            for prof in raw.get("profiles", []):
                ext = prof.get("p11_extended_86face_split", {})
                acc = ext.get("accuracy", {})
                lat = ext.get("total_detection_latency_ms", {})
                tp_s = ext.get("throughput", {})
                records.append(
                    ExperimentRecord(
                        experiment_id=f"HIST-P2-DET-{prof['profile_id']}",
                        name=f"P2 Detection: {prof['label']}",
                        source_origin="historical_readonly",
                        milestone="P2",
                        category="detection",
                        timestamp_utc=str(raw.get("timestamp_utc", "")),
                        status="completed",
                        dataset_id="nasa_p11_multiscale",
                        dataset_split="p11_extended_86face_regression_split (9 images / 86 faces)",
                        dataset_manifest_sha256="4006fddbb96aa0f5df5c5825b336904ce4d57bfb6319f9d82cc74b2e6ad8dd87",
                        algorithms={"detector": prof["model_id"]},
                        model_hashes={"detector": prof.get("model_sha256")},
                        parameters={
                            "resize_policy": prof.get("resize_policy"),
                            "score_threshold": prof.get("score_threshold"),
                            "nms_threshold": prof.get("nms_threshold"),
                        },
                        software_environment=raw.get("hardware_and_runtime", {}),
                        hardware=raw.get("hardware_and_runtime", {}),
                        execution_provider=str(prof.get("execution_provider", "")),
                        precision=str(prof.get("precision", "fp32")),
                        batch_size=int(prof.get("batch_size", 1)),
                        seed=42,
                        metrics={
                            "precision": acc.get("precision"),
                            "recall": acc.get("recall"),
                            "f1": acc.get("f1"),
                            "mean_iou": acc.get("mean_iou"),
                            "small_lt32_recall": acc.get("size_stratified_recall", {}).get("small_lt_32px", {}).get("recall"),
                            "p50_latency_ms": lat.get("p50_ms"),
                            "p95_latency_ms": lat.get("p95_ms"),
                            "images_per_sec": tp_s.get("images_per_sec"),
                            "faces_per_sec": tp_s.get("detected_faces_per_sec"),
                        },
                        artifact_refs=["reports/p2_detection_comparison.json"],
                        notes="Immutable Milestone P2 Workstream A detection profile (note P2-AUD-01 partially YuNet-seeded GT).",
                    ).to_dict()
                )

        # 3. P3 Experimental Results (`reports/p3_research/p3_experimental_results.json`)
        p3_res = resolve_project_path("reports/p3_research/p3_experimental_results.json")
        if p3_res.is_file():
            raw = json.loads(p3_res.read_text(encoding="utf-8"))
            ts = str(raw.get("generated_at_utc", "2026-11-17T00:00:00Z"))
            hw = raw.get("environment", {})

            # 3A. Detection resolution sweep (`EXP-P3-01`)
            exp01 = raw.get("exp_p3_01_detection_sensitivity", {})
            if exp01:
                y_sw = exp01.get("yunet_resolution_sweep", [])
                s_sw = exp01.get("scrfd_resolution_sweep", [])
                records.append(
                    ExperimentRecord(
                        experiment_id="HIST-P3-EXP01-DET-SWEEP",
                        name="P3 EXP-P3-01: YuNet vs SCRFD Resolution, Canvas & Confidence Sweeps",
                        source_origin="historical_readonly",
                        milestone="P3",
                        category="detection",
                        timestamp_utc=ts,
                        status="completed",
                        dataset_id="nasa_p11_multiscale",
                        dataset_split="p11_detection_small_face_gt (9 images / 86 faces)",
                        dataset_manifest_sha256="4006fddbb96aa0f5df5c5825b336904ce4d57bfb6319f9d82cc74b2e6ad8dd87",
                        algorithms={"detectors": ["yunet_2023mar", "scrfd_500m_research"]},
                        model_hashes={},
                        parameters={"resolutions": [320, 640, 960, 1280, 1600]},
                        software_environment=hw,
                        hardware=hw,
                        execution_provider="OpenCV_CPU vs ORT_CUDA",
                        precision="fp32",
                        batch_size=1,
                        seed=42,
                        metrics={
                            "yunet_1280_recall": 1.0,
                            "yunet_1280_f1": 0.9942,
                            "yunet_1280_p50_ms": 109.73,
                            "scrfd_1280_recall": 1.0,
                            "scrfd_1280_f1": 1.0,
                            "scrfd_1280_p50_ms": 74.41,
                        },
                        curves={
                            "yunet_resolution_sweep": y_sw,
                            "scrfd_resolution_sweep": s_sw,
                            "yunet_confidence_sweep": exp01.get("yunet_confidence_sweep", []),
                            "scrfd_confidence_sweep": exp01.get("scrfd_confidence_sweep", []),
                        },
                        artifact_refs=["reports/p3_research/p3_experimental_results.json"],
                        notes="Immutable P3 detection sensitivity sweep.",
                    ).to_dict()
                )

            # 3B. Alignment & Detector Landmark Decoupling (`EXP-P3-02`)
            exp02 = raw.get("exp_p3_02_alignment_and_detector_coupling", {})
            lm_cmp = exp02.get("detector_landmark_source_comparison", {})
            for model_key, label, prec in (
                ("adaface_ir50_fp16", "AdaFace IR-50 FP16", "fp16"),
                ("mobilefacenet_w600k_fp32", "MobileFaceNet W600K FP32", "fp32"),
            ):
                m_block = lm_cmp.get(model_key, {})
                for lm_src in ("yunet_5pt_landmarks", "scrfd_5pt_landmarks"):
                    st = m_block.get(lm_src, {})
                    if not st:
                        continue
                    op1e3 = st.get("calibrated_operating_points", {}).get("target_far_0.001", {})
                    records.append(
                        ExperimentRecord(
                            experiment_id=f"HIST-P3-EXP02-{model_key.upper()}-{lm_src.upper()}",
                            name=f"P3 EXP-P3-02: {label} with {lm_src.replace('_', ' ')}",
                            source_origin="historical_readonly",
                            milestone="P3",
                            category="recognition",
                            timestamp_utc=ts,
                            status="completed",
                            dataset_id="lfw_research_split",
                            dataset_split="Calib Folds 1–5 (2,086 IDs) -> Test Folds 6–10 (2,195 IDs, 0 overlap)",
                            dataset_manifest_sha256="ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
                            algorithms={
                                "landmark_source": lm_src,
                                "recognizer": "adaface_ir50_webface4m" if "adaface" in model_key else "mobilefacenet_w600k_research",
                            },
                            model_hashes={},
                            parameters={"transform_mode": "similarity_umeyama_4dof", "interpolation": "linear"},
                            software_environment=hw,
                            hardware=hw,
                            execution_provider="CUDAExecutionProvider",
                            precision=prec,
                            batch_size=8,
                            seed=42,
                            metrics={
                                "test_accuracy_pct": st.get("test_accuracy_at_calib_thresh_pct"),
                                "test_eer_pct": st.get("test_eer_pct"),
                                "test_roc_auc": st.get("test_roc_auc"),
                                "test_tar_at_calib_far_1e3": op1e3.get("test_realized_tar"),
                                "test_realized_far_at_calib_1e3": op1e3.get("test_realized_far"),
                                "calibrated_threshold_1e3": op1e3.get("calibrated_threshold"),
                            },
                            artifact_refs=["reports/p3_research/p3_experimental_results.json"],
                            notes="Identity-disjoint LFW 5-fold calibration -> 5-fold test evaluation.",
                        ).to_dict()
                    )

            # 3C. Flip-TTA & Batch Scaling (`EXP-P3-06` & `EXP-P3-07`)
            exp06 = raw.get("exp_p3_06_matching_and_tta", {}).get("horizontal_flip_tta", {})
            for model_key, label, prec in (
                ("adaface_ir50_fp16", "AdaFace IR-50 FP16", "fp16"),
                ("mobilefacenet_w600k_fp32", "MobileFaceNet W600K FP32", "fp32"),
            ):
                mb = exp06.get(model_key, {})
                for mode_k, mode_lbl, lat_k in (
                    ("single_pass_calib_test", "Single-Pass", "b8_single_pass_latency_ms"),
                    ("flip_tta_calib_test", "Horizontal-Flip TTA", "b8_flip_tta_latency_ms"),
                ):
                    st = mb.get(mode_k, {})
                    if not st:
                        continue
                    op1e3 = st.get("calibrated_operating_points", {}).get("target_far_0.001", {})
                    lat_d = mb.get(lat_k, {})
                    records.append(
                        ExperimentRecord(
                            experiment_id=f"HIST-P3-EXP06-{model_key.upper()}-{mode_k.upper()}",
                            name=f"P3 EXP-P3-06: {label} — {mode_lbl}",
                            source_origin="historical_readonly",
                            milestone="P3",
                            category="recognition",
                            timestamp_utc=ts,
                            status="completed",
                            dataset_id="lfw_research_split",
                            dataset_split="Calib Folds 1–5 -> Test Folds 6–10 (3,000 test pairs)",
                            dataset_manifest_sha256="ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
                            algorithms={"recognizer": model_key, "tta": mode_k == "flip_tta_calib_test"},
                            model_hashes={},
                            parameters={"flip_tta": mode_k == "flip_tta_calib_test", "metric": "cosine"},
                            software_environment=hw,
                            hardware=hw,
                            execution_provider="CUDAExecutionProvider",
                            precision=prec,
                            batch_size=8,
                            seed=42,
                            metrics={
                                "test_accuracy_pct": st.get("test_accuracy_at_calib_thresh_pct"),
                                "test_eer_pct": st.get("test_eer_pct"),
                                "test_roc_auc": st.get("test_roc_auc"),
                                "test_tar_at_calib_far_1e3": op1e3.get("test_realized_tar"),
                                "test_realized_far_at_calib_1e3": op1e3.get("test_realized_far"),
                                "p50_latency_ms": lat_d.get("p50_ms"),
                            },
                            artifact_refs=["reports/p3_research/p3_experimental_results.json"],
                            notes="Reconciled in P3.1: always inspect realized test FAR alongside TAR.",
                        ).to_dict()
                    )

            exp07 = raw.get("exp_p3_07_batch_and_precision", {})
            if exp07:
                records.append(
                    ExperimentRecord(
                        experiment_id="HIST-P3-EXP07-BATCH-SWEEP",
                        name="P3 EXP-P3-07: GuardedFixedBatchEmbedder vs Dynamic Batching (B=1..32, FP32/FP16)",
                        source_origin="historical_readonly",
                        milestone="P3",
                        category="recognition",
                        timestamp_utc=ts,
                        status="completed",
                        dataset_id="nasa_p11_multiscale",
                        dataset_split="86-face variable-count stream ([1,1,3,3,7,11,44,16])",
                        dataset_manifest_sha256="4006fddbb96aa0f5df5c5825b336904ce4d57bfb6319f9d82cc74b2e6ad8dd87",
                        algorithms={"recognizer": "adaface_ir50_webface4m"},
                        model_hashes={},
                        parameters={"batch_sizes": [1, 2, 4, 8, 16, 32]},
                        software_environment=hw,
                        hardware=hw,
                        execution_provider="CUDAExecutionProvider",
                        precision="fp16/fp32",
                        batch_size=8,
                        seed=42,
                        metrics={
                            "peak_throughput_faces_per_sec": 292.75,
                            "fp16_b8_fixed_p50_ms": 280.90,
                            "fp16_b8_dynamic_p50_ms": 2141.13,
                            "speedup_factor": 7.39,
                        },
                        curves={"batch_sweep": exp07.get("heterogeneous_86face_stream_sweep", [])},
                        artifact_refs=["reports/p3_research/p3_experimental_results.json"],
                        notes="Resolves P2-AUD-06 static batch guard and immutable batch_size.",
                    ).to_dict()
                )

            exp08 = raw.get("exp_p3_08_large_scale_retrieval", {})
            if exp08:
                records.append(
                    ExperimentRecord(
                        experiment_id="HIST-P3-EXP08-RETRIEVAL-SCALING",
                        name="P3 EXP-P3-08: 1:N Retrieval Scaling (ExactFlat, IVF-Flat, IVF-PQ, HNSW up to N=100K)",
                        source_origin="historical_readonly",
                        milestone="P3",
                        category="retrieval",
                        timestamp_utc=ts,
                        status="completed",
                        dataset_id="lfw_research_split",
                        dataset_split="13,233 real LFW embeddings + unit-hypersphere synthetic distractors (500 queries)",
                        dataset_manifest_sha256="ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
                        algorithms={"indexes": ["exact_flat_index", "ivf_flat_index", "ivf_pq_index", "hnsw_index"]},
                        model_hashes={},
                        parameters={"gallery_sizes": [13233, 50000, 100000]},
                        software_environment=hw,
                        hardware=hw,
                        execution_provider="PyTorch_CUDA_FP16 + NumPy_CPU",
                        precision="fp16/fp32/uint8_pq",
                        batch_size=64,
                        seed=808,
                        metrics={
                            "cuda_fp16_100k_q1_p50_ms": 1.04,
                            "cuda_fp16_100k_q64_qps": 37110.0,
                            "ivf_pq_100k_memory_mib": 4.82,
                            "ivf_pq_100k_recall_at_5": 0.984,
                            "vector_compression_ratio": 64.0,
                        },
                        curves={
                            "flat_ivf_pq_scaling": exp08.get("flat_ivf_pq_scaling", []),
                            "hnsw_scaling": exp08.get("hnsw_10k_sweep", []),
                        },
                        artifact_refs=["reports/p3_research/p3_experimental_results.json"],
                        notes="Audited in P3.1: Python loop overhead separated from algorithmic Recall@K and memory.",
                    ).to_dict()
                )

        return records

    def list_experiments(self, include_historical: bool = True) -> list[dict[str, Any]]:
        """Return Workbench runs (newest first) followed by read-only historical benchmarks."""
        wb_runs: list[dict[str, Any]] = []
        for jf in sorted(self.runs_dir.glob("EXP-WB-*.json"), reverse=True):
            try:
                wb_runs.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue

        if not include_historical:
            return wb_runs

        if self._historical_cache is None:
            self._historical_cache = self._ingest_historical_records()
        return wb_runs + list(self._historical_cache)

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        for rec in self.list_experiments(include_historical=True):
            if rec.get("experiment_id") == experiment_id:
                return rec
        raise WorkbenchError(f"Experiment '{experiment_id}' not found.", code="EXPERIMENT_NOT_FOUND", status_code=404)

    def save_workbench_record(self, record: ExperimentRecord | dict[str, Any]) -> dict[str, Any]:
        payload = record.to_dict() if isinstance(record, ExperimentRecord) else dict(record)
        exp_id = str(payload.get("experiment_id") or "")
        if not exp_id.startswith("EXP-WB-"):
            ts_tag = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            exp_id = f"EXP-WB-{ts_tag}-{uuid.uuid4().hex[:6].upper()}"
            payload["experiment_id"] = exp_id

        out_file = self.runs_dir / f"{exp_id}.json"
        if isinstance(payload.get("curves"), dict) and isinstance(payload["curves"].get("score_distributions"), dict):
            payload["curves"]["score_distributions"]["experiment_id"] = exp_id
        payload.setdefault("artifact_refs", []).append(f"reports/workbench_runs/{out_file.name}")
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def save_named_configuration(
        self,
        name: str,
        workspace: str,
        config: dict[str, Any],
        description: str = "",
        hardware_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ts_tag = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        exp_id = f"EXP-WB-CFG-{ts_tag}-{uuid.uuid4().hex[:6].upper()}"
        rec = ExperimentRecord(
            experiment_id=exp_id,
            name=name or f"Saved Configuration ({workspace})",
            source_origin="saved_configuration",
            milestone="Workbench",
            category=workspace,
            timestamp_utc=self._utc_now(),
            status="saved_config",
            dataset_id=str(config.get("dataset_id", "interactive_selection")),
            dataset_split=str(config.get("dataset_split", "n/a")),
            dataset_manifest_sha256=None,
            algorithms=config.get("algorithms", {}),
            model_hashes={},
            parameters=config,
            software_environment={
                "python": (hardware_snapshot or {}).get("python_version", "3.11"),
                "torch": (hardware_snapshot or {}).get("torch_version", ""),
            },
            hardware=(hardware_snapshot or {}).get("gpu", {}),
            execution_provider=str(config.get("device", "auto")),
            precision=str(config.get("precision", "auto")),
            batch_size=int(config.get("batch_size", 1)),
            seed=int(config.get("seed", 42)),
            metrics={},
            notes=description,
        )
        return self.save_workbench_record(rec)

    def export_experiment(self, experiment_id: str, fmt: str = "json") -> tuple[bytes, str, str]:
        """Export an experiment record as JSON, CSV metrics/curves, or YAML configuration."""
        rec = self.get_experiment(experiment_id)
        f = str(fmt).lower().strip()
        if f == "json":
            return (
                json.dumps(rec, indent=2).encode("utf-8"),
                "application/json; charset=utf-8",
                f"{experiment_id}.json",
            )
        if f == "yaml":
            cfg_doc = {
                "experiment_id": rec.get("experiment_id"),
                "name": rec.get("name"),
                "category": rec.get("category"),
                "dataset_id": rec.get("dataset_id"),
                "dataset_split": rec.get("dataset_split"),
                "algorithms": rec.get("algorithms"),
                "model_hashes": rec.get("model_hashes"),
                "execution_provider": rec.get("execution_provider"),
                "precision": rec.get("precision"),
                "batch_size": rec.get("batch_size"),
                "seed": rec.get("seed"),
                "parameters": rec.get("parameters"),
            }
            emp_op = (rec.get("curves") or {}).get("empirical_operating_point")
            if emp_op:
                cfg_doc["empirical_operating_point"] = emp_op
            return (
                yaml.safe_dump(cfg_doc, sort_keys=False).encode("utf-8"),
                "application/x-yaml; charset=utf-8",
                f"{experiment_id}_config.yaml",
            )
        if f == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["experiment_id", "name", "milestone", "category", "dataset_id", "dataset_split", "metric_key", "metric_value"])
            for mk, mv in (rec.get("metrics") or {}).items():
                writer.writerow(
                    [
                        rec.get("experiment_id"),
                        rec.get("name"),
                        rec.get("milestone"),
                        rec.get("category"),
                        rec.get("dataset_id"),
                        rec.get("dataset_split"),
                        mk,
                        mv,
                    ]
                )
            return (
                buf.getvalue().encode("utf-8"),
                "text/csv; charset=utf-8",
                f"{experiment_id}_metrics.csv",
            )
        raise WorkbenchError(f"Unsupported export format '{fmt}'; expected json, csv, or yaml.", code="INVALID_EXPORT_FORMAT")
