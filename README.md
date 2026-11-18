# Face Biometric Research Workbench

A local, reproducible research workbench for evaluating face detection, alignment, quality assessment, recognition, and 1:N face matching across accuracy, robustness, and performance.

> **Research & Evaluation Disclaimer**
> This repository provides an **algorithmic research and experimental evaluation workbench only**. It is **not** certified, hardened, licensed, or represented as a production biometric identification, authentication, access-control, or surveillance system. No biometric datasets, face images, identity records, or pretrained model weights are distributed with this repository.

---

## 1. Overview & Research Objectives

The **Face Biometric Research Workbench** is a self-contained, localhost-only laboratory designed for systematic inspection, parameter ablation, and side-by-side comparison of core stages in a modern 2D facial analysis pipeline. Rather than treating face recognition as an opaque black box, the workbench exposes every intermediate geometric, photometric, and vector-space stage so researchers can:

1. **Inspect & Modify Stage Parameters:** Dynamically adjust detection thresholds, Non-Maximum Suppression (NMS) regimes, 5-point landmark alignment transforms, quality filters, embedding precision/TTA modes, similarity metrics, and Approximate Nearest Neighbor (ANN) index hyperparameters.
2. **Track Cross-Stage Configuration Deltas:** Audit active parameter deviations from baseline registry defaults across all pipeline stages and compare two experimental configurations side by side with dataset/protocol mismatch safeguards.
3. **Analyze Empirical Score Distributions & Operating Points:** Explore genuine vs. impostor similarity/distance histograms, scrub decision thresholds ($\tau$) interactively with zero backend re-inference, inspect synchronized empirical ROC/DET operating points, and trace false-accept (`FP`) / false-reject (`FN`) pairs while keeping calibration and held-out test partitions strictly separated.
4. **Perform Pixel-Exact Visual Diagnostics:** Inspect bounding boxes, 5-point facial landmarks, canonical `112×112` warped crops, crop-difference heatmaps, and continuous annotation overlay opacity (`0–100%`) without altering underlying image pixels.
5. **Measure Hardware & Runtime Trade-Offs:** Profile latency, throughput, CPU/RAM usage, and GPU VRAM residency across `CPU` and `CUDA` (`FP32` / `FP16`) execution providers.

---

## 2. Research Scope

- **Face Detection:** Multi-scale single-face and crowd face localization, bounding-box + 5-point landmark regression, input resolution scaling policies (`preserve_aspect_max_dim`, `native`, `fixed`), and Hard Greedy NMS vs. Gaussian Soft-NMS ablations.
- **Face Alignment:** Canonical `112×112` 5-point ArcFace template registration comparing `4-DoF` Umeyama similarity transforms, `6-DoF` unconstrained least-squares affine warps, `+10%` context margin zoom-out, interpolation kernels (`nearest`, `linear`, `cubic`, `lanczos4`), and controlled landmark jitter / roll perturbations.
- **Face Quality Assessment:** Classical geometric and photometric indicators (Laplacian variance sharpness, luminance mean, contrast standard deviation, inter-ocular distance, roll/yaw symmetry), raw pre-normalization embedding norm (`||z_raw||_2`) utility analysis, controlled synthetic degradation experiments (Gaussian blur, downscale-upscale, gamma, JPEG compression), and norm-weighted multi-image template pooling.
- **Face Recognition & Embedding Extraction:** `512-D` and `128-D` hypersphere feature extraction across PyTorch, ONNX Runtime (`CPUExecutionProvider` / `CUDAExecutionProvider`), and OpenCV DNN runtimes, including `FP32` vs. `FP16` precision, horizontal Flip Test-Time Augmentation (TTA), and dynamic vs. guarded fixed-batch execution.
- **1:1 Face Verification & Matching:** Pairwise verification across `cosine` similarity, `normalized_l2`, `unnormalized_l2`, and `normalized_l1` metrics with out-of-sample threshold calibration and interactive operating-point exploration.
- **1:N Face Retrieval:** Exact vs. approximate vector search comparing `ExactFlat` (`CPU` / `CUDA FP32` / `CUDA FP16`), `IVF-Flat`, `IVF-PQ` (Asymmetric Distance Computation with subspace product quantization), and pure-Python `HNSW` graph search across recall@K, query latency, build time, and memory footprint.
- **Accuracy, Robustness & Performance Evaluation:** Unified reproducibility ledger capturing software/hardware fingerprints, SHA-256 model/manifest digests, stage timings, and exportable `JSON`, `CSV`, and `YAML` artifacts.

---

## 3. Architecture & Major Components

The workbench uses a three-layer architecture that separates core computer-vision implementations from the localhost HTTP adapter and zero-build browser UI:

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Browser SPA (`http://127.0.0.1:6173`) — 12 Research Workspaces (Vanilla ES2022)  │
│  • Schema-Driven Parameter Editors & Cross-Stage Delta Drawer (`config_delta.js`)│
│  • Genuine/Impostor Histogram & Zero-Reinference τ Explorer (`threshold_explorer`)│
│  • Pixel-Exact Canvas Viewer with 0–100% Overlay Opacity (`image_viewer.js`)     │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ HTTP/1.1 (`/api/v1/*`, localhost-only)
┌───────────────────────────────────────▼──────────────────────────────────────────┐
│ Workbench Adapter Layer (`src/workbench/`)                                       │
│  • `server.py`        — ThreadingHTTPServer, loopback & safe-port guards         │
│  • `registry.py`      — Dynamic algorithm, model & parameter schema registry     │
│  • `model_manager.py` — Lazy loader, CUDA/RAM telemetry & LRU VRAM session caps  │
│  • `datasets_io.py`   — Read-only dataset catalog & path-confined image loader   │
│  • `executors.py`     — Thin stage coordinators invoking core Python modules     │
│  • `job_runner.py`    — Serialized (`max_workers=1`) async queue & cancellation  │
│  • `history_store.py` — Experiment persistence & `JSON` / `CSV` / `YAML` exports │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Direct Python Calls
┌───────────────────────────────────────▼──────────────────────────────────────────┐
│ Core Research Modules (`src/`)                                                   │
│  • `src/detection/`   — `YuNetDetector`, `SCRFDDetector`                         │
│  • `src/alignment/`   — `FaceAligner` (4-DoF Umeyama similarity)                 │
│  • `src/quality/`     — `FaceQualityAssessor`                                    │
│  • `src/recognition/` — `AdaFaceIRBackbone`, `ResNet50Embedder`,               │
│                         `MobileFaceNetEmbedder`, `SFaceEmbedder`, `onnx_export`  │
│  • `src/matching/`    — `ExactFlatIndex`, `IVFFlatIndex`, `IVFPQIndex`, `HNSW`   │
│  • `src/p3_research/` — Soft-NMS, 6-DoF affine, degradations, Flip-TTA, metrics  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Repository Layout

```text
face-biometric-research-workbench/
├── configs/                        # YAML pipeline, benchmark & experiment configurations
├── docs/
│   └── architecture/
│       └── WORKBENCH_API_CONTRACT.md  # Localhost `/api/v1` HTTP/JSON endpoint specification
├── frontend/                       # Zero-build ES2022 Single-Page Application (12 workspaces)
│   ├── index.html
│   ├── styles/workbench.css
│   └── js/                         # Core state/API/canvas/chart components & 12 workspace modules
├── models/
│   ├── detection/                  # Place user-downloaded detection ONNX weights here (gitignored)
│   ├── recognition/                # Place user-downloaded recognition weights here (gitignored)
│   └── manifests/
│       └── model_manifest.json     # Expected file paths, SHA-256 digests & upstream license metadata
├── datasets/                       # Empty placeholders; place user-acquired datasets here (gitignored)
├── reports/                        # Empty placeholder; local session runs saved here (gitignored)
├── requirements/
│   ├── base.txt                    # CPU-only pinned dependencies (`onnxruntime`)
│   ├── gpu.txt                     # NVIDIA CUDA pinned dependencies (`onnxruntime-gpu`, `cu128`)
│   └── dev.txt                     # Test runner (`pytest`)
├── scripts/
│   ├── validate_environment.py     # Python, PyTorch, ONNX Runtime & GPU verification utility
│   └── export_adaface_onnx.py      # AdaFace IR-50 PyTorch (`.pt`) -> ONNX (`FP32`/`FP16`) exporter
├── src/                            # Core CV algorithms & localhost workbench backend
├── tests/                          # Self-contained Python (`pytest`) & JavaScript (`node`) tests
└── environment.yml                 # Optional Conda environment wrapper
```

---

## 4. Supported Algorithms & Models (Implemented vs. Unbundled / Out-of-Scope)

> **Important:** Algorithm adapters and mathematical implementations are included in source code, whereas **neural network weight binaries (`.onnx`, `.pt`) are not bundled** and must be downloaded independently by the researcher (see [Section 9](#9-dataset--model-weight-acquisition-policy)).

### 4.1 Implemented in Workbench Source Code

| Stage | Algorithm / Adapter ID | Implementation Status | Runtimes / Precisions | Notes |
| :--- | :--- | :---: | :--- | :--- |
| **Detection** | `yunet_2023mar` (`YuNetDetector`) | **Implemented** *(weights user-supplied)* | OpenCV DNN (`CPU`, `FP32`) | Multi-stride `[8, 16, 32]` CNN face + 5-landmark detector. |
| **Detection** | `scrfd_500m_research` (`SCRFDDetector` + `ResearchSCRFDWrapper`) | **Implemented** *(weights user-supplied)* | ONNX Runtime (`CUDA`/`CPU`, `FP32`) | Supports Hard Greedy NMS and Gaussian Soft-NMS ablations. |
| **Alignment** | `five_point_canonical_aligner` (`FaceAligner` + research variants) | **Implemented** *(no external weights)* | NumPy / OpenCV (`CPU`) | `4-DoF` Umeyama similarity, `6-DoF` affine, `+10%` zoom-out, 4 interpolations. |
| **Quality** | `hybrid_classical_and_norm_quality` (`FaceQualityAssessor` + P3 features) | **Implemented** *(no external weights)* | NumPy / OpenCV (`CPU`) | Sharpness, exposure, contrast, pose symmetry, `\|\|z_raw\|\|_2`, synthetic degradations. |
| **Recognition** | `adaface_ir50_webface4m` (`AdaFaceIRBackbone` / `ResNet50Embedder`) | **Implemented** *(weights user-supplied)* | ONNX Runtime & PyTorch (`CUDA`/`CPU`, `FP32`/`FP16`) | `512-D` IR-50 backbone; dynamic batching & horizontal Flip-TTA supported. |
| **Recognition** | `mobilefacenet_w600k_research` (`MobileFaceNetEmbedder`) | **Implemented** *(weights user-supplied)* | ONNX Runtime (`CUDA`/`CPU`, `FP32`) | Lightweight `512-D` embedding extractor; dynamic batching supported. |
| **Recognition** | `sface_2021dec` (`SFaceEmbedder`) | **Implemented** *(weights user-supplied)* | ONNX Runtime (`CUDA`/`CPU`), OpenCV DNN (`CPU`, `FP32`) | `128-D` embedding extractor; upstream ONNX graph enforces static `batch_size = 1`. |
| **Matching** | `pairwise_vector_matcher` (`compute_pair_scores`) | **Implemented** *(no external weights)* | Vectorized NumPy | `cosine`, `normalized_l2`, `unnormalized_l2`, `normalized_l1`. |
| **Retrieval** | `exact_flat_index`, `ivf_flat_index`, `ivf_pq_index`, `hnsw_index` | **Implemented** *(no external weights)* | NumPy (`CPU`) & PyTorch (`CUDA FP32`/`FP16` on `ExactFlat`) | Exact brute-force reference + 3 configurable ANN index structures. |

### 4.2 Unbundled Slots, Non-Implemented Modules & Future Candidates

| Component | Status | Explanation |
| :--- | :---: | :--- |
| `reference_resnet50_baseline` (Reference External ResNet50 Slot) | **Adapter Slot Only (`MISSING_LOCAL_WEIGHTS`)** | Configurable adapter slot for user-supplied `512-D` ResNet-50 checkpoints/ONNX models. Raises an explicit `HTTP 422 MISSING_MODEL_WEIGHTS` error when absent; never substitutes ImageNet weights. |
| Semantic Facial Occlusion Classifier | **Not Implemented** | Explicitly disclosed in Workspace `05` as `NOT_IMPLEMENTED` rather than fabricating heuristic occlusion scores. |
| Compiled C++ / FAISS Graph ANN Backends (`N >= 100K` HNSW) | **Not Implemented (Future Candidate)** | Current `HNSWIndex` is a transparent pure-Python research implementation capped at `N <= 15,000` vectors. |
| Model Training / Fine-Tuning Pipelines | **Out of Scope** | This workbench is strictly an inference, evaluation, and algorithmic ablation laboratory. |

---

## 5. Hardware & Software Prerequisites

- **Operating System:** Windows 10/11 (64-bit) or Linux (x86_64).
- **Python:** Python `3.11.x` (isolated virtual environment strongly recommended).
- **Compute Hardware:**
  - **CPU Mode:** Any modern multi-core x86_64 processor with `>= 8 GB` RAM (`requirements/base.txt`).
  - **GPU Mode (Recommended):** NVIDIA CUDA-capable GPU (`>= 6 GB` VRAM recommended) with compatible NVIDIA drivers supporting CUDA `12.x` (`requirements/gpu.txt`).
- **Browser:** Recent Chromium-based browser (Chrome, Edge) or Firefox for the ES2022 Single-Page Application (recommended viewport `1920×1080` or `1440×900`).
- **Optional Test Runtime:** Node.js `>= 18` (only if running the standalone frontend coordinate/unit verification script `tests/js_coordinates_check.mjs`).

---

## 6. Clean Environment Installation & Startup

### 6.1 Clone the Repository & Create an Isolated Virtual Environment

> **Important Dependency Rule:** Install **either** `requirements/gpu.txt` (CUDA workstation) **or** `requirements/base.txt` (CPU-only machine) — **never both in the same environment**, because `onnxruntime` and `onnxruntime-gpu` share the same Python module namespace.

**Windows (PowerShell):**
```powershell
git clone https://github.com/tvlnsiva/face-biometric-research-workbench.git
cd face-biometric-research-workbench

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Option A — NVIDIA CUDA GPU environment:
python -m pip install --upgrade pip
python -m pip install -r requirements/gpu.txt -r requirements/dev.txt

# Option B — CPU-only environment (use instead of Option A if no NVIDIA GPU is present):
# python -m pip install -r requirements/base.txt -r requirements/dev.txt
```

**Linux / macOS (Bash):**
```bash
git clone https://github.com/tvlnsiva/face-biometric-research-workbench.git
cd face-biometric-research-workbench

python3.11 -m venv .venv
source .venv/bin/activate

# Choose EITHER requirements/gpu.txt OR requirements/base.txt:
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt -r requirements/dev.txt
```

### 6.2 Run Self-Contained Verification Tests (No Weights or Datasets Required)

Even before downloading external model weights or datasets, you can verify your Python runtime, synthetic ONNX/PyTorch execution, vector-retrieval indexes, and frontend ES2022 mathematical modules:

```bash
python -m pytest -v
node tests/js_coordinates_check.mjs
```
*(Tests that require optional third-party model weights or local evaluation datasets will cleanly report `SKIPPED` until those files are placed locally.)*

### 6.3 Start the Local Research Workbench Server

```bash
python -m src.workbench.server --host 127.0.0.1 --port 6173
```

Expected readiness message:
```text
Face Biometric Research Workbench listening at http://127.0.0.1:6173
```

---

## 7. Accessing the Local Workbench

1. Open **`http://127.0.0.1:6173`** in your web browser.
2. **Localhost Confinement & Port Policy:**
   - By design, `server.py` refuses wildcard (`0.0.0.0`, `*`) or external LAN interface bindings and binds exclusively to loopback (`127.0.0.1`, `localhost`, or `::1`).
   - Custom ports are restricted to `6000–6999` (default `6173`), automatically excluding browser-restricted unsafe ports (`6000`, `6566`, `6665–6669`, `6697`).
3. **12 Interactive Research Workspaces:**
   - `01 · Dashboard` — Live CPU/RAM/GPU telemetry, loaded model sessions, and quick navigation.
   - `02 · Image Explorer` — Local session image upload (`+ Load Local Image(s)`), metadata/SHA-256 inspection, and side-by-side viewer.
   - `03 · Face Detection Lab` — Bounding-box & 5-landmark detection, NMS ablations, and `0–100%` overlay opacity inspection.
   - `04 · Landmark & Alignment Lab` — `4-DoF` vs. `6-DoF` warps, SVD affine decomposition, and `TURBO` crop difference maps.
   - `05 · Face Quality Lab` — Classical + norm quality profiling, interactive degradation simulator, and template weighting.
   - `06 · Recognition & Matching Lab` — `1:1` multi-face verification, instant zero-reinference threshold slider, aligned-crop alpha blend, and `1:N` probe search.
   - `07 · 1:N Retrieval Lab` — `ExactFlat`, `IVF-Flat`, `IVF-PQ`, and `HNSW` recall/latency/memory benchmarking.
   - `08 · Pipeline Builder` — End-to-end 8-stage pipeline validation, inline configuration delta, and async execution.
   - `09 · Algorithm Comparison` — Multi-run comparison tables, Two-Run Configuration Comparison, and Genuine/Impostor Score Distribution & Threshold Explorer.
   - `10 · Dataset & Eval Explorer` — Read-only inspection of locally installed evaluation splits and licensing rules.
   - `11 · Experiment History` — Reproducibility ledger, configuration reload, and `JSON` / `CSV` / `YAML` exports.
   - `12 · Model & Algo Registry` — Full parameter schemas, SHA-256 verification status, and on-demand model load/release controls.

---

## 8. Research Workflow & Experiment Reproducibility

- **Session-Local Image Upload Without Full Datasets:** You can immediately test single-image detection, alignment, quality, and `1:1` verification by importing local test images via **`+ Load Local Image(s)`** in `02 · Image Explorer` (stored strictly locally under `reports/workbench_runs/uploads/`, gitignored) once at least one detector and recognizer ONNX model are placed in `models/`.
- **Cross-Stage Configuration Auditing (`P-1`):** Any parameter change across Detection, Alignment, Quality, Recognition, or Retrieval is tracked in real time in the header **`Active Configuration Delta`** drawer against registry defaults, excluding inactive conditional parameters.
- **Calibration vs. Held-Out Test Isolation (`P-2`):** On pairwise verification datasets such as LFW (`10` subject-disjoint folds), calibration folds (`Folds 1–5`) and held-out evaluation folds (`Folds 6–10`) are stored in separate partitions and never pooled silently.
- **Provenance & Exports:** Completed workbench runs are written atomically as independent JSON records under `reports/workbench_runs/EXP-WB-*.json` recording model SHA-256 hashes, software versions, hardware metadata, stage parameters, and empirical operating points, and can be exported as `JSON`, `CSV`, or `YAML`.

---

## 9. Dataset & Model-Weight Acquisition Policy

**No model weights, checkpoints, or biometric datasets are distributed in this repository.** A public source-code license does not grant rights to redistribute upstream pretrained weights or face datasets. Users must obtain weights and datasets independently from their authoritative upstream owners, review and comply with all applicable license terms, and place files at the relative paths below.

### 9.1 Model Weight Placement (`models/manifests/model_manifest.json`)

| `model_id` | Local Relative Target Path | Upstream Source & Notes | Expected SHA-256 |
| :--- | :--- | :--- | :--- |
| `yunet_2023mar` | `models/detection/face_detection_yunet_2023mar.onnx` | [OpenCV Zoo — `face_detection_yunet`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (`2023mar`) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `sface_2021dec` | `models/recognition/face_recognition_sface_2021dec.onnx` | [OpenCV Zoo — `face_recognition_sface`](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (`2021dec`) | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| `scrfd_500m_research` | `models/detection/research_only/scrfd_500m.onnx` | [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) (`det_500m.onnx`, non-commercial research only) | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` |
| `mobilefacenet_w600k_research` | `models/recognition/research_only/mobilefacenet_w600k_mbf.onnx` | [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) (`buffalo_s` `w600k_mbf.onnx`, non-commercial research only) | `9cc6e4a73eb9f9fb8fb3a2be1a7aa8ea4f0d8784ab232a3ad8f5b8f12661cd92` |
| `adaface_ir50_webface4m_pt` | `models/recognition/public_resnet50/adaface_ir50_webface4m.pt` | [AdaFace / CVLFace (`mk-minchul/AdaFace`)](https://github.com/mk-minchul/AdaFace) (`ir50` WebFace4M checkpoint; non-commercial research only). Run `python scripts/export_adaface_onnx.py` to generate `.onnx` (`FP32`) and `_fp16.onnx` (`FP16`) locally. | `43bd2d57c9c2dfca82be3bfdebc2fb5cdab4c16ca38ffad6fb3920b9f38f865b` |

After placing model files, run `python scripts/validate_environment.py` (or inspect Workspace `12 · Model & Algo Registry`) to verify SHA-256 integrity.

### 9.2 Dataset Placement Policy

- **Interactive Local Images:** Import your own permitted test images directly through `02 · Image Explorer` (`+ Load Local Image(s)`), or place images inside `datasets/samples/`.
- **LFW Academic Verification Benchmark (`Optional, Research Only`):** If your institution/use case complies with the UMass Labeled Faces in the Wild academic benchmark terms, place `pairs.txt` at `datasets/evaluation/lfw_research_only/pairs.txt` and extracted `lfw_funneled/<Person>/<Person>_0001.jpg` folders under `datasets/evaluation/lfw_research_only/lfw_funneled/`. All files under `datasets/` remain gitignored.

---

## 10. Known Limitations

1. **No Bundled Weights or Datasets:** Fresh clones start with `MISSING_LOCAL_WEIGHTS` until models are placed under `models/` per Section 9.
2. **Static Batch Size (`B = 1`) on `SFace`:** Upstream `face_recognition_sface_2021dec.onnx` defines a static `[1, 3, 112, 112]` input tensor; requesting `batch_size > 1` intentionally returns `HTTP 400 STATIC_BATCH_VIOLATION`.
3. **CPU Execution on `YuNet`:** `YuNetDetector` executes via OpenCV `cv2.FaceDetectorYN` (`CPU FP32`).
4. **Pure-Python `HNSWIndex` Scale Guard (`N <= 15,000`):** To prevent excessive pure-Python graph construction times and laptop RAM lockups, `HNSWIndex` enforces `gallery_size <= 15,000`. Use `ExactFlat`, `IVF-Flat`, or `IVF-PQ` for larger synthetic/benchmark galleries up to `1,000,000` vectors.
5. **Interactive Verification Pair Cap (`<= 600` Pairs in Browser Queue):** Interactive pipeline runs in Workspace `08` cap stratified pair sampling at `600` pairs (`300` Calibration + `300` Held-Out Test) for responsive interactive turnaround.
6. **GPU Session Eviction Caps:** `ModelResourceManager` retains at most `1` CUDA detector and `2` CUDA recognizers concurrently (evicting least-recently-used sessions automatically) to protect `6 GB`-class GPUs from out-of-memory failures.

---

## 11. Licensing & Third-Party Attribution

### 11.1 Separation of Source Code, Weights, and Training Datasets

Source code in this repository is distinct from external pretrained weights and biometric training datasets:
- **Pretrained Model Weights & Training Datasets:** Upstream models trained on `WebFace4M`, `WebFace600K`, `MS-Celeb-1M`, `CASIA-WebFace`, or `WIDER FACE` carry **non-commercial research restrictions** or unresolved upstream dataset provenance (`production_eligible: false` in [`models/manifests/model_manifest.json`](models/manifests/model_manifest.json)). Users are solely responsible for verifying licensing compliance before using any third-party weights or datasets.

### 11.2 Third-Party Source Attribution & Notices

- **AdaFace IR-50 Backbone (`src/recognition/adaface_ir50.py`):**
  Adapted from [`mk-minchul/AdaFace`](https://github.com/mk-minchul/AdaFace) (`net.py`) and [`minchul/cvlface_adaface_ir50_webface4m`](https://huggingface.co/minchul/cvlface_adaface_ir50_webface4m) (`models/iresnet/model.py`), Minchul Kim, Anil K. Jain, and Xiaoming Liu, *"AdaFace: Quality Adaptive Margin for Face Recognition"*, CVPR 2022, released under the **MIT License**:
  > Copyright (c) 2022 Minchul Kim
  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  > The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
- **OpenCV Zoo Models (`YuNet` & `SFace` Adapters):**
  Interfaces with OpenCV `FaceDetectorYN` (`YuNet`,Wu et al., 2023; upstream model directory MIT License, Copyright (c) 2020 Shiqi Yu; WIDER FACE training provenance) and `SFace` (Zhong et al., 2021; upstream OpenCV Zoo wrapper Apache-2.0 License).
- **InsightFace Research Candidates (`SCRFD` & `MobileFaceNet` Adapters):**
  ONNX inference pre/post-processing compatible with [`deepinsight/insightface`](https://github.com/deepinsight/insightface) (`SCRFD`, Guo et al., ICLR 2022; `MobileFaceNet`, Chen et al., 2018). Upstream InsightFace pretrained weights are strictly restricted to **non-commercial research only**.

---

## 12. Non-Production Research Disclaimer

This software is provided **"AS IS"**, without warranty of any kind, strictly for **academic, scientific, and algorithmic research evaluation**. It has **not** undergone biometric security certification, presentation-attack (spoofing) evaluation, demographic fairness auditing, privacy-compliance certification, or network-security hardening, and **must not** be deployed in production biometric identification, law-enforcement, border-control, employment, or commercial decision-making systems.
