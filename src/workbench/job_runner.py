"""Serialized Asynchronous Experiment Job Queue (`src/workbench/job_runner.py`).

Executes multi-step sweeps, retrieval scaling benchmarks, and pipeline runs on a single background
worker (`max_workers=1`) so concurrent GPU experiments cannot exhaust `6 GB` RTX 3060 VRAM, while
exposing real-time progress, cooperative cancellation, structured errors, and automatic persistence.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from src.workbench.contracts import ExperimentRecord, WorkbenchError
from src.workbench.executors import WorkbenchExecutionEngine
from src.workbench.history_store import ExperimentHistoryStore


@dataclass
class AsyncJobEntry:
    job_id: str
    job_type: str
    name: str
    config: dict[str, Any]
    save_experiment: bool
    status: str = "queued"  # "queued" | "running" | "completed" | "failed" | "cancelled"
    progress_pct: float = 0.0
    stage_message: str = "Queued"
    submitted_at_utc: str = ""
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    experiment_id: str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_summary(self, include_result: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "name": self.name,
            "status": self.status,
            "progress_pct": round(float(self.progress_pct), 1),
            "stage_message": self.stage_message,
            "submitted_at_utc": self.submitted_at_utc,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "experiment_id": self.experiment_id,
            "error": self.error,
        }
        if include_result:
            payload["config"] = self.config
            payload["result"] = self.result
        return payload


class ExperimentJobRunner:
    """Single-worker async experiment runner with queue status and cancellation."""

    def __init__(self, engine: WorkbenchExecutionEngine, history: ExperimentHistoryStore) -> None:
        self.engine = engine
        self.history = history
        self._lock = threading.RLock()
        self._jobs: dict[str, AsyncJobEntry] = {}
        self._order: list[str] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wb-exp-worker")

    @staticmethod
    def _utc_now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def submit(
        self,
        job_type: str,
        name: str,
        config: dict[str, Any],
        save_experiment: bool = True,
    ) -> dict[str, Any]:
        if job_type not in {"detection_sweep", "retrieval_benchmark", "pipeline_run"}:
            raise WorkbenchError(f"Unsupported async job_type='{job_type}'.", code="INVALID_JOB_TYPE")

        job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
        entry = AsyncJobEntry(
            job_id=job_id,
            job_type=job_type,
            name=name or f"{job_type} ({job_id})",
            config=dict(config),
            save_experiment=bool(save_experiment),
            submitted_at_utc=self._utc_now(),
        )
        with self._lock:
            self._jobs[job_id] = entry
            self._order.append(job_id)
            queued_ahead = sum(1 for j in self._jobs.values() if j.status in {"queued", "running"} and j.job_id != job_id)

        self._executor.submit(self._execute_job, job_id)
        return {
            "job_id": job_id,
            "status": entry.status,
            "queued_ahead": queued_ahead,
            "submitted_at_utc": entry.submitted_at_utc,
        }

    def _execute_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.stage_message = "Cancelled before execution started."
                job.finished_at_utc = self._utc_now()
                return
            job.status = "running"
            job.started_at_utc = self._utc_now()
            job.stage_message = "Starting experiment..."

        def _prog(pct: float, msg: str) -> None:
            with self._lock:
                job.progress_pct = max(0.0, min(100.0, float(pct)))
                job.stage_message = msg

        def _cancelled() -> bool:
            return job.cancel_event.is_set()

        try:
            model_hashes: dict[str, str | None] = {}
            exec_prov = "cpu"
            exec_prec = "fp32"
            exec_bs = 1

            if job.job_type == "detection_sweep":
                res = self.engine.run_detection_sweep(job.config, progress_cb=_prog, cancel_check=_cancelled)
                cat = "detection"
                ds_id = str(job.config.get("dataset_id", "nasa_p11_multiscale"))
                ds_split = res.get("dataset_split", ds_id)
                det_id = res["detector_id"]
                algos = {"detector": det_id}
                if det_id in self.engine.registry:
                    model_hashes["detector"] = self.engine.registry[det_id].model_sha256
                exec_prov = str(job.config.get("base_params", {}).get("device", "cpu" if det_id == "yunet_2023mar" else "cuda"))
                metrics = res["summary_metrics"]
                curves = {"sweep_points": res["points"]}
                stage_graph: list[dict[str, Any]] = []
            elif job.job_type == "retrieval_benchmark":
                res = self.engine.run_retrieval_benchmark(job.config, progress_cb=_prog, cancel_check=_cancelled)
                cat = "retrieval"
                ds_id = "lfw_plus_synthetic_hypersphere" if res["synthetic_distractors_count"] > 0 else "lfw_research_split"
                ds_split = f"N={res['gallery_size']} ({res['real_embeddings_count']} real + {res['synthetic_distractors_count']} synthetic)"
                algos = {"retrieval_index": res["index_id"], "gallery_embedder": "adaface_ir50_webface4m"}
                if "adaface_ir50_webface4m" in self.engine.registry:
                    model_hashes["gallery_embedder"] = self.engine.registry["adaface_ir50_webface4m"].model_sha256
                exec_prov = str(res["index_params"].get("backend", "numpy_cpu"))
                exec_prec = "fp16" if "fp16" in exec_prov else ("uint8_pq" if res["index_id"] == "ivf_pq_index" else "fp32")
                exec_bs = int(res["query_count"])
                metrics = {k: v for k, v in res["metrics"].items() if not isinstance(v, dict)}
                curves = {"exact_vs_ann_samples": res["exact_vs_ann_query_samples"]}
                stage_graph = []
            else:
                res = self.engine.run_pipeline_experiment(job.config, progress_cb=_prog, cancel_check=_cancelled)
                cat = "pipeline"
                ds_id = res["dataset_id"]
                ds_split = str(res.get("dataset_split") or ds_id)
                det_id = str(job.config.get("detection", {}).get("algorithm_id", "yunet_2023mar"))
                rec_id = str(job.config.get("recognition", {}).get("algorithm_id", "adaface_ir50_webface4m"))
                algos = {"detector": det_id, "recognizer": rec_id}
                if det_id in self.engine.registry:
                    model_hashes["detector"] = self.engine.registry[det_id].model_sha256
                rmeta = res.get("recognizer_meta") or {}
                model_hashes["recognizer"] = rmeta.get("model_sha256") or (
                    self.engine.registry[rec_id].model_sha256 if rec_id in self.engine.registry else None
                )
                exec_prov = str(rmeta.get("device") or job.config.get("recognition", {}).get("params", {}).get("device", "cpu"))
                exec_prec = str(rmeta.get("precision") or job.config.get("recognition", {}).get("params", {}).get("precision", "fp32"))
                exec_bs = int(rmeta.get("batch_size") or job.config.get("recognition", {}).get("params", {}).get("batch_size", 1))
                metrics = res["metrics"]
                curves = res.get("curves", {})
                stage_graph = res.get("stage_graph", [])

            ds_manifest_sha = next(
                (d.get("manifest_sha256") for d in self.engine.datasets.list_datasets() if d.get("dataset_id") == ds_id),
                None,
            )

            hw = self.engine.models.get_hardware_snapshot()
            exp_id: str | None = None
            if job.save_experiment:
                rec = ExperimentRecord(
                    experiment_id="",
                    name=job.name,
                    source_origin="workbench_run",
                    milestone="Workbench",
                    category=cat,
                    timestamp_utc=self._utc_now(),
                    status="completed",
                    dataset_id=ds_id,
                    dataset_split=str(ds_split),
                    dataset_manifest_sha256=ds_manifest_sha,
                    algorithms=algos,
                    model_hashes=model_hashes,
                    parameters=job.config,
                    software_environment={
                        "python": hw["python_version"],
                        "torch": hw["torch_version"],
                        "onnxruntime": hw["onnxruntime_version"],
                    },
                    hardware=hw["gpu"],
                    execution_provider=exec_prov,
                    precision=exec_prec,
                    batch_size=exec_bs,
                    seed=int(job.config.get("seed", 42)),
                    metrics=metrics,
                    curves=curves,
                    stage_graph=stage_graph,
                )
                saved = self.history.save_workbench_record(rec)
                exp_id = saved["experiment_id"]

            with self._lock:
                job.status = "completed"
                job.progress_pct = 100.0
                job.stage_message = "Completed"
                job.finished_at_utc = self._utc_now()
                job.result = res
                job.experiment_id = exp_id

        except WorkbenchError as exc:
            with self._lock:
                if exc.code == "JOB_CANCELLED":
                    job.status = "cancelled"
                    job.stage_message = "Cancelled by user."
                else:
                    job.status = "failed"
                    job.stage_message = exc.message
                job.error = {"code": exc.code, "message": exc.message, "details": exc.details}
                job.finished_at_utc = self._utc_now()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job.status = "failed"
                job.stage_message = f"Unexpected error: {exc}"
                job.error = {
                    "code": "INTERNAL_EXECUTION_ERROR",
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
                job.finished_at_utc = self._utc_now()

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise WorkbenchError(f"Job '{job_id}' not found.", code="JOB_NOT_FOUND", status_code=404)
            job = self._jobs[job_id]
            job.cancel_event.set()
            if job.status == "queued":
                job.status = "cancelled"
                job.stage_message = "Cancelled while queued."
                job.finished_at_utc = self._utc_now()
            return {"job_id": job_id, "cancel_requested": True, "status": job.status}

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise WorkbenchError(f"Job '{job_id}' not found.", code="JOB_NOT_FOUND", status_code=404)
            return self._jobs[job_id].to_summary(include_result=True)

    def list_jobs(self) -> dict[str, Any]:
        with self._lock:
            items = [self._jobs[jid].to_summary(include_result=False) for jid in reversed(self._order)]
            active = next((it for it in items if it["status"] == "running"), None)
            return {"active_job": active, "jobs": items}
