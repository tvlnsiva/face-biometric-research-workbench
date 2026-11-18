# Face Biometric Research Workbench — Typed Backend Experiment API Contract (`v1`)

- **Document ID:** `WORKBENCH-API-CONTRACT-V1`
- **Base URL:** `http://127.0.0.1:6173/api/v1` *(Port configurable within `6001–6999`, default `6173`)*
- **Transport:** Localhost HTTP/1.1 (`JSON` UTF-8 payloads for structured contracts; `image/jpeg` / `image/png` binary streams for native-resolution images and aligned crops; `text/csv` and `application/x-yaml` for exports).

---

## 1. System, Environment & Resource Endpoints

| Method & Path | Request Schema | Response Schema | Purpose |
| :--- | :--- | :--- | :--- |
| `GET /api/v1/system/status` | — | `{ status, port, python_version, torch_version, ort_version, ort_providers, cpu_info, ram_used_mib, ram_total_mib, gpu_available, gpu_name, vram_allocated_mib, vram_reserved_mib, vram_total_mib, loaded_models: [...], active_job: {...}, counts: {...} }` | Live dashboard telemetry, hardware/VRAM status, and active job summary. |
| `GET /api/v1/registry` | — | `{ schema_version, categories, algorithms: [...], models: [...], canonical_landmarks_112, stage_definitions: [...] }` | Authoritative algorithm, model, provenance, and parameter-schema registry driving dynamic UI controls. |
| `POST /api/v1/models/load` | `{ model_id, runtime?, device?, precision? }` | `{ status, key, model_id, runtime, device, precision, init_time_ms, loaded_models, vram_allocated_mib }` | Explicitly warm/load a model on demand through `ModelResourceManager`. |
| `POST /api/v1/models/unload` | `{ key? , unload_all?: bool }` | `{ status, unloaded_keys: [...], loaded_models, vram_allocated_mib }` | Release GPU/CPU model sessions and flush CUDA cache (`torch.cuda.empty_cache()`). |

---

## 2. Dataset, Image Explorer & Binary Image Streaming Endpoints

| Method & Path | Request / Query Params | Response Schema | Purpose |
| :--- | :--- | :--- | :--- |
| `GET /api/v1/datasets` | — | `{ datasets: [{ dataset_id, name, source, license, provenance, restrictions, num_images, num_identities, num_faces, split_info, manifest_path, manifest_sha256, has_detection_gt, has_verification_pairs }] }` | Lists all permitted local datasets, manifests, splits, and licensing/provenance terms. |
| `GET /api/v1/datasets/{dataset_id}/items` | `?offset=0&limit=60&split=...&kind=images\|pairs` | `{ dataset_id, kind, total, offset, limit, items: [...], distributions: {...} }` | Paginated dataset images or ground-truth verification pairs (`ground_truth` strictly separated from predictions). |
| `POST /api/v1/images/inspect` | `{ image_ref }` *(dataset `image_path`, registered local path, or base64 data upload stored in session scratch)* | `{ image_ref, image_id, width, height, channels, format, file_size_bytes, sha256, url, ground_truth_boxes: [...] \| null, ground_truth_source }` | Returns native dimensions, SHA-256, metadata, and ground-truth boxes (if defined in P1/P1.1 manifests). |
| `POST /api/v1/images/scan-directory` | `{ directory_path, limit?: 100 }` | `{ directory_path, count, images: [{ image_ref, filename, width, height, file_size_bytes }] }` | Loads local image directories for multi-image research exploration. |
| `POST /api/v1/images/upload` | `{ filename, data_base64 }` | `{ image_ref, image_id, width, height, sha256, file_size_bytes, url }` | Registers user-selected local browser file(s) in local session scratch (`reports/workbench_runs/uploads/`). |
| `GET /api/v1/images/raw` | `?ref=<encoded_image_ref>` | Binary `image/jpeg` or `image/png` (`Cache-Control: private, max-age=300`) | Streams native-resolution original image without resampling. |

---

## 3. Synchronous Interactive Lab Endpoints (Single Image / Pair / Probe)

| Method & Path | Request Schema | Response Schema | Purpose |
| :--- | :--- | :--- | :--- |
| `POST /api/v1/detect/run` | `{ image_ref, detector_id, params: {...}, compare_baseline?: bool }` | `{ image_meta, config_used, detections: [...], gt_evaluation: { available, tp, fp, fn, precision, recall, f1, mean_iou, small_recall, medium_recall, large_recall, matches, unmatched_gt } \| { available: false }, stage_profile_ms, resource_snapshot, baseline_comparison? }` | Interactive Face Detection Lab execution in original pixel coordinates. |
| `POST /api/v1/align/run` | `{ image_ref, face_index?: 0, landmark_source: "yunet_2023mar"\|"scrfd_500m_research", compare_both_sources?: bool, params: { transform_mode, interpolation, crop_width, crop_height, zoom_factor, extra_rotation_deg, translate_x_px, translate_y_px, jitter_sigma_px, seed } }` | `{ image_meta, primary: { source_landmarks, canonical_template, transformed_landmarks, affine_matrix_2x3, crop_bounds_xyxy, inter_ocular_px, execution_time_ms, crop_data_url }, secondary?: {...}, landmark_displacement?: { per_point_px, mean_px, max_px }, crop_diff_data_url?, crop_mae? }` | Interactive Landmark & Alignment Lab with correspondence geometry and crop difference visualization. |
| `POST /api/v1/quality/run` | `{ image_refs: [...], detector_id, recognizer_id?, degradation?: { blur_sigma, effective_res, gamma, jpeg_quality }, warning_thresholds?: {...} }` | `{ items: [{ image_ref, face_id, crop_data_url, classical_p1: {...}, extended_p3: {...}, quality_category, raw_vs_learned_note }], ranking: [...], template_weights?: {...} }` | Interactive Face Quality Assessment Lab across single or multiple images + controlled degradations. |
| `POST /api/v1/verify/run` | `{ image_a_ref, image_b_ref, face_index_a?: 0, face_index_b?: 0, detector_id, alignment_params, recognizer_id, recognizer_params: { runtime, device, precision, batch_size, flip_tta, l2_normalize, experimental_preprocess_override? }, metric, threshold, calibration_source? }` | `{ face_a: {...}, face_b: {...}, all_faces_a: [...], all_faces_b: [...], metrics: { cosine, normalized_l2, unnormalized_l2, normalized_l1, euclidean_unit_dist }, decision: { metric, score, threshold, distance_from_threshold, is_match, calibration_context, score_semantics_note }, visual_comparison: { crop_diff_data_url, landmark_displacement_px }, latency_ms, preprocessing_warning? }` | `1:1` Verification Lab; returns embeddings & all metrics so subsequent threshold slider moves update instantaneously without re-running inference. |
| `POST /api/v1/identify/search` | `{ probe_image_ref, probe_face_index?: 0, gallery_source: "nasa_p11"\|"lfw_sample"\|"custom_refs", custom_image_refs?: [...], gallery_limit?: 200, detector_id, recognizer_id, recognizer_params, index_type, index_params, top_k, threshold }` | `{ probe_face: {...}, gallery_size, index_summary: {...}, candidates: [{ rank, template_id, subject_tag_eval_only, image_ref, crop_data_url, score, metric, above_threshold }], search_latency_ms }` | Interactive `1:N` candidate retrieval with click-to-compare gallery templates. |
| `POST /api/v1/pipeline/validate` | `{ stages: {...} }` | `{ valid: bool, errors: [...], warnings: [...], normalized_config: {...} }` | Pre-execution compatibility validator for the Pipeline Experiment Builder. |

---

## 4. Asynchronous Experiment Queue, History & Export Endpoints

| Method & Path | Request Schema | Response Schema | Purpose |
| :--- | :--- | :--- | :--- |
| `POST /api/v1/jobs/submit` | `{ job_type: "detection_sweep" \| "quality_evrc" \| "retrieval_benchmark" \| "pipeline_run" \| "dataset_verification", name, config: {...}, save_experiment?: true }` | `{ job_id, status: "queued"\|"running", position_in_queue, submitted_at_utc }` | Submits multi-step or dataset-scale experiments to the serialized worker queue. |
| `GET /api/v1/jobs` | — | `{ active_job, jobs: [...] }` | Lists queued, running, completed, failed, and cancelled jobs with progress (`0..100%`). |
| `GET /api/v1/jobs/{job_id}` | — | `{ job_id, status, progress_pct, stage_message, started_at_utc, finished_at_utc, experiment_id?, result?, error? }` | Polls job execution status, live stage message, and completed experiment record. |
| `POST /api/v1/jobs/{job_id}/cancel` | — | `{ job_id, cancel_requested: true, status }` | Signals cooperative cancellation (`cancel_event.set()`). |
| `GET /api/v1/experiments` | `?include_historical=true` | `{ experiments: [...] }` | Unified list of read-only historical P1/P1.1/P2/P3 benchmarks + saved Workbench experiments. |
| `GET /api/v1/experiments/{experiment_id}` | — | Full reproducible experiment record (`experiment_id`, `timestamp_utc`, `dataset`, `manifest_sha256`, `models`, `config`, `software_env`, `hardware`, `metrics`, `curves`, `stage_graph`, `status`). | Retrieves complete experiment record for inspection, reproduction, or comparison. |
| `POST /api/v1/experiments/save-config` | `{ name, description?, workspace, config }` | `{ experiment_id, saved_path }` | Persists a named configuration / preset under `reports/workbench_runs/`. |
| `GET /api/v1/experiments/{experiment_id}/export` | `?format=json\|csv\|yaml` | Downloadable `.json`, `.csv` metrics table, or `.yaml` reproducible configuration. | Exports results and configurations without touching historical artifacts. |
