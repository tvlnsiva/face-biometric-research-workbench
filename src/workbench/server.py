"""HTTP/1.1 Research Workbench Server (`src/workbench/server.py`).

Serves both:
  - `/api/v1/*` typed JSON & binary image endpoints backed by `WorkbenchExecutionEngine`
  - Static SPA assets from `frontend/`
Default port: `6173` (strictly enforced in the `6000–6999` range, avoiding `8000`-range ports and
browser-blocked X11/IRC ports `6000`, `6566`, `6665–6669`).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from src.common.config_loader import resolve_project_path
from src.workbench import (
    BROWSER_RESTRICTED_6000_PORTS,
    DEFAULT_WORKBENCH_HOST,
    DEFAULT_WORKBENCH_PORT,
)
from src.workbench.contracts import WorkbenchError
from src.workbench.datasets_io import DatasetCatalog
from src.workbench.executors import WorkbenchExecutionEngine
from src.workbench.history_store import ExperimentHistoryStore
from src.workbench.job_runner import ExperimentJobRunner
from src.workbench.model_manager import ModelResourceManager
from src.workbench.registry import (
    export_registry_schema_json,
    get_full_registry_payload,
    validate_pipeline_configuration,
)


ALLOWED_LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_JSON_BODY_BYTES: int = 24 * 1024 * 1024


def validate_workbench_host(host: str) -> str:
    h = str(host or "").strip().lower()
    if h not in ALLOWED_LOCAL_HOSTS:
        raise ValueError(
            f"Workbench binds strictly to localhost ({sorted(ALLOWED_LOCAL_HOSTS)}); external/wildcard host '{host}' is prohibited."
        )
    return h


def validate_workbench_port(port: int) -> int:
    p = int(port)
    if not (6000 <= p <= 6999):
        raise ValueError(
            f"Workbench port must be in the 6000-6999 range (8000-range ports are prohibited); got {p}."
        )
    if p in BROWSER_RESTRICTED_6000_PORTS:
        raise ValueError(
            f"Port {p} is blocked by Chromium/Firefox (`ERR_UNSAFE_PORT`). Use a safe 6000-range port such as 6173."
        )
    return p


class WorkbenchAppContext:
    """Shared singleton services attached to the HTTP server instance."""

    def __init__(self, port: int = DEFAULT_WORKBENCH_PORT) -> None:
        self.port = port
        self.frontend_dir: Path = resolve_project_path("frontend")
        self.models = ModelResourceManager()
        self.datasets = DatasetCatalog()
        self.history = ExperimentHistoryStore()
        self.engine = WorkbenchExecutionEngine(self.models, self.datasets)
        self.jobs = ExperimentJobRunner(self.engine, self.history)
        self.registry_payload = get_full_registry_payload()

    def get_system_status(self) -> dict[str, Any]:
        hw = self.models.get_hardware_snapshot()
        jobs_info = self.jobs.list_jobs()
        exps = self.history.list_experiments(include_historical=True)
        completed_n = sum(1 for e in exps if e.get("status") == "completed")
        failed_n = sum(1 for j in jobs_info["jobs"] if j.get("status") == "failed")
        return {
            "status": "ok",
            "port": self.port,
            "hardware_and_env": hw,
            "active_job": jobs_info["active_job"],
            "counts": {
                "registered_algorithms": len(self.registry_payload["algorithms"]),
                "available_datasets": len(self.datasets.list_datasets()),
                "total_experiments": len(exps),
                "completed_experiments": completed_n,
                "failed_jobs_session": failed_n,
                "loaded_models": len(hw["loaded_models"]),
            },
        }


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    """Thread-safe HTTP handler routing `/api/v1/*` and `frontend/*`."""

    server_version = "FaceBiometricResearchWorkbench/1.0"

    @property
    def ctx(self) -> WorkbenchAppContext:
        return self.server.wb_context  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep console quiet during high-frequency polling/tests
        return

    def _send_json(self, payload: Any, status_code: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status_code: int = 200,
        download_filename: str | None = None,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        if download_filename:
            self.send_header("Content-Disposition", f'attachment; filename="{download_filename}"')
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > MAX_JSON_BODY_BYTES:
            # Drain up to 64 MiB in constant 64 KiB discard chunks so Windows Winsock delivers
            # the structured HTTP 413 JSON response cleanly instead of aborting with TCP RST (10053).
            self.close_connection = True
            if length <= 64 * 1024 * 1024:
                rem = length
                while rem > 0:
                    chunk = self.rfile.read(min(65536, rem))
                    if not chunk:
                        break
                    rem -= len(chunk)
            raise WorkbenchError(
                f"Request body ({length} bytes) exceeds maximum permitted payload ({MAX_JSON_BODY_BYTES} bytes).",
                code="PAYLOAD_TOO_LARGE",
                status_code=413,
            )
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise WorkbenchError(f"Malformed JSON request body: {exc}", code="INVALID_JSON") from exc
        if not isinstance(parsed, dict):
            raise WorkbenchError("JSON body must be an object.", code="INVALID_JSON")
        return parsed

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if path.startswith("/api/v1/"):
                self._handle_api_get(path, qs)
            else:
                self._serve_static_asset(path)
        except WorkbenchError as exc:
            self._send_json(exc.to_dict(), status_code=exc.status_code)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=4),
                    }
                },
                status_code=500,
            )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
            if path.startswith("/api/v1/"):
                self._handle_api_post(path, body)
            else:
                raise WorkbenchError(f"Unknown POST route '{path}'.", code="NOT_FOUND", status_code=404)
        except WorkbenchError as exc:
            self._send_json(exc.to_dict(), status_code=exc.status_code)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=4),
                    }
                },
                status_code=500,
            )

    def _handle_api_get(self, path: str, qs: dict[str, list[str]]) -> None:
        if path == "/api/v1/system/status":
            self._send_json(self.ctx.get_system_status())
            return

        if path == "/api/v1/registry":
            self._send_json(self.ctx.registry_payload)
            return

        if path == "/api/v1/datasets":
            self._send_json({"datasets": self.ctx.datasets.list_datasets()})
            return

        if path.startswith("/api/v1/datasets/") and path.endswith("/items"):
            parts = path.strip("/").split("/")
            ds_id = unquote(parts[3])
            kind = (qs.get("kind") or ["images"])[0]
            split_f = (qs.get("split") or ["all"])[0]
            offset = int((qs.get("offset") or ["0"])[0])
            limit = int((qs.get("limit") or ["50"])[0])
            self._send_json(
                self.ctx.datasets.get_dataset_items(ds_id, kind=kind, split_filter=split_f, offset=offset, limit=limit)
            )
            return

        if path == "/api/v1/images/raw":
            ref = (qs.get("ref") or [""])[0]
            img_path = self.ctx.datasets.resolve_image_ref(unquote(ref))
            mime = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
            self._send_bytes(img_path.read_bytes(), mime, cache_control="private, max-age=300")
            return

        if path == "/api/v1/jobs":
            self._send_json(self.ctx.jobs.list_jobs())
            return

        if path.startswith("/api/v1/jobs/"):
            job_id = unquote(path.split("/api/v1/jobs/", 1)[1].strip("/"))
            self._send_json(self.ctx.jobs.get_job(job_id))
            return

        if path == "/api/v1/experiments":
            inc_hist = (qs.get("include_historical") or ["true"])[0].lower() != "false"
            self._send_json({"experiments": self.ctx.history.list_experiments(include_historical=inc_hist)})
            return

        if path.startswith("/api/v1/experiments/") and path.endswith("/export"):
            parts = path.strip("/").split("/")
            exp_id = unquote(parts[3])
            fmt = (qs.get("format") or ["json"])[0]
            raw_b, ctype, fname = self.ctx.history.export_experiment(exp_id, fmt=fmt)
            self._send_bytes(raw_b, ctype, download_filename=fname)
            return

        if path.startswith("/api/v1/experiments/"):
            exp_id = unquote(path.split("/api/v1/experiments/", 1)[1].strip("/"))
            self._send_json(self.ctx.history.get_experiment(exp_id))
            return

        raise WorkbenchError(f"Unknown API endpoint '{path}'.", code="NOT_FOUND", status_code=404)

    def _handle_api_post(self, path: str, body: dict[str, Any]) -> None:
        if path == "/api/v1/models/load":
            model_id = str(body.get("model_id", ""))
            cat_spec = self.ctx.engine.registry.get(model_id)
            if cat_spec is None:
                raise WorkbenchError(f"Unknown model_id='{model_id}'.", code="UNKNOWN_MODEL")
            if cat_spec.category == "detection":
                ent = self.ctx.models.get_detector(model_id, device=body.get("device"))
            elif cat_spec.category == "recognition":
                ent = self.ctx.models.get_recognizer(
                    model_id,
                    runtime=body.get("runtime"),
                    device=body.get("device"),
                    precision=body.get("precision"),
                )
            else:
                raise WorkbenchError(f"Algorithm '{model_id}' does not have neural weights to load.", code="NOT_A_MODEL")
            self._send_json(
                {
                    "status": "loaded",
                    "entry": ent.summary(),
                    "hardware": self.ctx.models.get_hardware_snapshot(),
                }
            )
            return

        if path == "/api/v1/models/unload":
            removed = self.ctx.models.unload(
                cache_key=body.get("key"),
                unload_all=bool(body.get("unload_all", False)),
            )
            self._send_json(
                {
                    "status": "unloaded",
                    "unloaded_keys": removed,
                    "hardware": self.ctx.models.get_hardware_snapshot(),
                }
            )
            return

        if path == "/api/v1/images/inspect":
            self._send_json(self.ctx.datasets.inspect_image(str(body.get("image_ref", ""))))
            return

        if path == "/api/v1/images/scan-directory":
            self._send_json(
                self.ctx.datasets.scan_local_directory(
                    directory_path=str(body.get("directory_path", "")),
                    limit=int(body.get("limit", 120)),
                )
            )
            return

        if path == "/api/v1/images/upload":
            self._send_json(
                self.ctx.datasets.save_uploaded_image(
                    filename=str(body.get("filename", "upload.jpg")),
                    data_base64=str(body.get("data_base64", "")),
                )
            )
            return

        if path == "/api/v1/detect/run":
            res = self.ctx.engine.run_detection(
                image_ref=str(body.get("image_ref", "")),
                detector_id=str(body.get("detector_id", "yunet_2023mar")),
                params=body.get("params") or {},
                compare_baseline=bool(body.get("compare_baseline", False)),
            )
            if bool(body.get("save_experiment", False)):
                hw = res["hardware_snapshot"]
                rec = self.ctx.history.save_workbench_record(
                    {
                        "experiment_id": "",
                        "name": str(body.get("experiment_name") or f"Detection ({res['execution_meta']['detector_id']} on {res['image_meta']['image_id']})"),
                        "source_origin": "workbench_run",
                        "milestone": "Workbench",
                        "category": "detection",
                        "timestamp_utc": time_utc_now(),
                        "status": "completed",
                        "dataset_id": "single_image",
                        "dataset_split": res["image_meta"]["image_ref"],
                        "dataset_manifest_sha256": None,
                        "algorithms": {"detector": res["execution_meta"]["detector_id"]},
                        "model_hashes": {"detector": res["execution_meta"]["model_sha256"]},
                        "parameters": res["parameters_used"],
                        "software_environment": {"python": hw["python_version"], "onnxruntime": hw["onnxruntime_version"]},
                        "hardware": hw["gpu"],
                        "execution_provider": res["execution_meta"]["device"],
                        "precision": res["execution_meta"]["precision"],
                        "batch_size": 1,
                        "seed": 42,
                        "metrics": {
                            "detection_count": res["detection_count"],
                            "precision": res["gt_evaluation"].get("precision"),
                            "recall": res["gt_evaluation"].get("recall"),
                            "f1": res["gt_evaluation"].get("f1"),
                            "mean_iou": res["gt_evaluation"].get("mean_iou"),
                            "p50_latency_ms": res["latency_and_throughput"]["total_detect_ms"],
                            "images_per_sec": res["latency_and_throughput"]["images_per_sec"],
                        },
                        "curves": {},
                        "stage_graph": [],
                        "artifact_refs": [],
                        "notes": "Saved interactive detection experiment.",
                    }
                )
                res["saved_experiment_id"] = rec["experiment_id"]
            self._send_json(res)
            return

        if path == "/api/v1/align/run":
            self._send_json(
                self.ctx.engine.run_alignment_lab(
                    image_ref=str(body.get("image_ref", "")),
                    face_index=int(body.get("face_index", 0)),
                    landmark_source=str(body.get("landmark_source", "yunet_2023mar")),
                    compare_both_detectors=bool(body.get("compare_both_detectors", True)),
                    compare_transform_mode=body.get("compare_transform_mode"),
                    params=body.get("params") or {},
                )
            )
            return

        if path == "/api/v1/quality/run":
            self._send_json(
                self.ctx.engine.run_quality_lab(
                    image_refs=list(body.get("image_refs") or []),
                    detector_id=str(body.get("detector_id", "yunet_2023mar")),
                    recognizer_id=str(body.get("recognizer_id", "adaface_ir50_webface4m")),
                    quality_params=body.get("quality_params") or {},
                    degradation=body.get("degradation") or {},
                )
            )
            return

        if path == "/api/v1/verify/run":
            res = self.ctx.engine.run_verification_1to1(
                image_a_ref=str(body.get("image_a_ref", "")),
                image_b_ref=str(body.get("image_b_ref", "")),
                face_index_a=int(body.get("face_index_a", 0)),
                face_index_b=int(body.get("face_index_b", 0)),
                detector_id=str(body.get("detector_id", "yunet_2023mar")),
                detector_params=body.get("detector_params") or {},
                alignment_params=body.get("alignment_params") or {},
                recognizer_id=str(body.get("recognizer_id", "adaface_ir50_webface4m")),
                recognizer_params=body.get("recognizer_params") or {},
                metric=str(body.get("metric", "cosine")),
                threshold=float(body.get("threshold", 0.2169)),
            )
            if bool(body.get("save_experiment", False)):
                hw = self.ctx.models.get_hardware_snapshot()
                det_used = str(body.get("detector_id", "yunet_2023mar"))
                det_sha = self.ctx.engine.registry.get(det_used).model_sha256 if det_used in self.ctx.engine.registry else None
                rec = self.ctx.history.save_workbench_record(
                    {
                        "experiment_id": "",
                        "name": str(body.get("experiment_name") or f"1:1 Verify ({res['recognizer_meta']['recognizer_id']})"),
                        "source_origin": "workbench_run",
                        "milestone": "Workbench",
                        "category": "recognition",
                        "timestamp_utc": time_utc_now(),
                        "status": "completed",
                        "dataset_id": "interactive_pair",
                        "dataset_split": f"{body.get('image_a_ref')} vs {body.get('image_b_ref')}",
                        "dataset_manifest_sha256": None,
                        "algorithms": {"detector": det_used, "recognizer": res["recognizer_meta"]["recognizer_id"]},
                        "model_hashes": {"detector": det_sha, "recognizer": res["recognizer_meta"]["model_sha256"]},
                        "parameters": body,
                        "software_environment": {
                            "python": hw["python_version"],
                            "torch": hw["torch_version"],
                            "onnxruntime": hw["onnxruntime_version"],
                        },
                        "hardware": hw["gpu"],
                        "execution_provider": res["recognizer_meta"]["device"],
                        "precision": res["recognizer_meta"]["precision"],
                        "batch_size": res["recognizer_meta"]["batch_size"],
                        "seed": 42,
                        "metrics": {
                            **res["metrics"],
                            "threshold": res["decision"]["threshold"],
                            "is_match": 1 if res["decision"]["is_match"] else 0,
                            "recognition_inference_ms": res["stage_latency_ms"]["recognition_inference_ms"],
                        },
                        "curves": {},
                        "stage_graph": [],
                        "artifact_refs": [],
                        "notes": "Saved 1:1 verification run.",
                    }
                )
                res["saved_experiment_id"] = rec["experiment_id"]
            self._send_json(res)
            return

        if path == "/api/v1/identify/search":
            self._send_json(
                self.ctx.engine.run_identification_search(
                    probe_image_ref=str(body.get("probe_image_ref", "")),
                    probe_face_index=int(body.get("probe_face_index", 0)),
                    gallery_source=str(body.get("gallery_source", "nasa_p11_verification")),
                    gallery_limit=int(body.get("gallery_limit", 120)),
                    custom_refs=body.get("custom_refs"),
                    detector_id=str(body.get("detector_id", "yunet_2023mar")),
                    recognizer_id=str(body.get("recognizer_id", "adaface_ir50_webface4m")),
                    recognizer_params=body.get("recognizer_params") or {},
                    index_id=str(body.get("index_id", "exact_flat_index")),
                    index_params=body.get("index_params") or {},
                    top_k=int(body.get("top_k", 10)),
                    threshold=float(body.get("threshold", 0.2169)),
                )
            )
            return

        if path == "/api/v1/pipeline/validate":
            res = validate_pipeline_configuration(body, self.ctx.engine.registry)
            self._send_json(res.to_dict())
            return

        if path == "/api/v1/jobs/submit":
            self._send_json(
                self.ctx.jobs.submit(
                    job_type=str(body.get("job_type", "")),
                    name=str(body.get("name", "")),
                    config=body.get("config") or {},
                    save_experiment=bool(body.get("save_experiment", True)),
                ),
                status_code=202,
            )
            return

        if path.startswith("/api/v1/jobs/") and path.endswith("/cancel"):
            parts = path.strip("/").split("/")
            job_id = unquote(parts[3])
            self._send_json(self.ctx.jobs.cancel(job_id))
            return

        if path == "/api/v1/experiments/save-config":
            saved = self.ctx.history.save_named_configuration(
                name=str(body.get("name", "Saved Configuration")),
                workspace=str(body.get("workspace", "pipeline")),
                config=body.get("config") or {},
                description=str(body.get("description", "")),
                hardware_snapshot=self.ctx.models.get_hardware_snapshot(),
            )
            self._send_json(saved, status_code=201)
            return

        raise WorkbenchError(f"Unknown POST API endpoint '{path}'.", code="NOT_FOUND", status_code=404)

    def _serve_static_asset(self, url_path: str) -> None:
        rel = url_path.lstrip("/") or "index.html"
        target = (self.ctx.frontend_dir / rel).resolve()
        if not str(target).startswith(str(self.ctx.frontend_dir.resolve())):
            raise WorkbenchError("Path traversal forbidden.", code="FORBIDDEN", status_code=403)
        if not target.is_file():
            target = (self.ctx.frontend_dir / "index.html").resolve()
        if not target.is_file():
            raise WorkbenchError("Frontend index.html not found.", code="FRONTEND_MISSING", status_code=404)

        ext = target.suffix.lower()
        mime_map = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".mjs": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon",
        }
        ctype = mime_map.get(ext) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), ctype, cache_control="no-cache")


def time_utc_now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def create_workbench_server(
    host: str = DEFAULT_WORKBENCH_HOST,
    port: int = DEFAULT_WORKBENCH_PORT,
) -> ThreadingHTTPServer:
    valid_host = validate_workbench_host(host)
    valid_port = validate_workbench_port(port)
    export_registry_schema_json()
    httpd = ThreadingHTTPServer((valid_host, valid_port), WorkbenchRequestHandler)
    httpd.wb_context = WorkbenchAppContext(port=valid_port)  # type: ignore[attr-defined]
    return httpd


def main() -> None:
    parser = argparse.ArgumentParser(description="Face Biometric Research Workbench Server")
    parser.add_argument("--host", default=DEFAULT_WORKBENCH_HOST, help="Bind host (default 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WORKBENCH_PORT,
        help="Port in 6000-6999 range (default 6173; 8000-range prohibited)",
    )
    args = parser.parse_args()

    srv = create_workbench_server(host=args.host, port=args.port)
    print(f"Face Biometric Research Workbench listening at http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
