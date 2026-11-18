"""Registry-Driven Algorithm, Model & Parameter Schema for the Research Workbench.

Reads authoritative metadata from `models/manifests/model_manifest.json`, `configs/models.yaml`,
and `configs/p2_models.yaml` (without modifying them) and exposes dynamic parameter schemas and
pre-execution compatibility validation across all pipeline stages.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from src.alignment.face_aligner import FaceAligner
from src.common.config_loader import compute_sha256, load_yaml_config, resolve_project_path
from src.p3_research.algorithmic_variants import ARCFACE_5PT_112
from src.workbench.contracts import (
    AlgorithmSpec,
    ParameterSpec,
    PipelineValidationResult,
    WorkbenchError,
)


def _load_manifest_index() -> dict[str, dict[str, Any]]:
    manifest_path = resolve_project_path("models/manifests/model_manifest.json")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {m["model_id"]: m for m in raw.get("models", [])}


def build_algorithm_registry() -> dict[str, AlgorithmSpec]:
    """Construct the complete registry of implemented algorithms and model-specific controls."""
    manifest_by_id = _load_manifest_index()
    p2_cfg = load_yaml_config("configs/p2_models.yaml")
    p1_cfg = load_yaml_config("configs/models.yaml")

    def _model_avail(rel_path: str | None) -> tuple[str, str | None]:
        if not rel_path:
            return "AVAILABLE", None
        p = resolve_project_path(rel_path)
        if p.is_file():
            return "AVAILABLE", compute_sha256(p)
        return "MISSING_LOCAL_WEIGHTS", None

    yunet_meta = manifest_by_id["yunet_2023mar"]
    yunet_status, yunet_sha = _model_avail(yunet_meta["local_path"])

    scrfd_meta = manifest_by_id["scrfd_500m_research"]
    scrfd_status, scrfd_sha = _model_avail(scrfd_meta["local_path"])

    ada_onnx_meta = manifest_by_id["adaface_ir50_webface4m"]
    ada_fp16_meta = manifest_by_id["adaface_ir50_webface4m_fp16"]
    ada_pt_meta = manifest_by_id["adaface_ir50_webface4m_pt"]
    ada_status, ada_sha = _model_avail(ada_onnx_meta["local_path"])

    sface_meta = manifest_by_id["sface_2021dec"]
    sface_status, sface_sha = _model_avail(sface_meta["local_path"])

    mbf_meta = manifest_by_id["mobilefacenet_w600k_research"]
    mbf_status, mbf_sha = _model_avail(mbf_meta["local_path"])

    ref_meta = manifest_by_id["reference_resnet50_baseline"]
    ref_status, ref_sha = _model_avail(ref_meta["local_path"])

    registry: list[AlgorithmSpec] = [
        # =====================================================================
        # 1. FACE DETECTION ALGORITHMS
        # =====================================================================
        AlgorithmSpec(
            algorithm_id="yunet_2023mar",
            name="YuNet (OpenCV Zoo 2023-Mar)",
            category="detection",
            description=(
                "Lightweight anchor-free/multi-stride [8, 16, 32] CNN detector producing "
                "bounding boxes and 5 facial landmarks in original image pixel coordinates via "
                "OpenCV FaceDetectorYN (CPU FP32)."
            ),
            paper_reference="Wu et al., 'YuNet: A Tiny Millisecond-level Face Detector', Machine Intelligence Research, 2023",
            source_implementation="src/detection/yunet_detector.py :: YuNetDetector",
            model_id="yunet_2023mar",
            checkpoint_path=yunet_meta["local_path"],
            model_sha256=yunet_sha,
            input_requirements={
                "color_order": "BGR",
                "value_range": "[0, 255] float32",
                "layout": "HWC BGR uint8 -> FaceDetectorYN internal NCHW",
                "dynamic_spatial": True,
            },
            output_schema={
                "detections": "list[DetectedFace] (bbox xywh in original px, 5-pt landmarks in original px, confidence)",
            },
            supported_devices=["cpu"],
            supported_precisions=["fp32"],
            supported_runtimes=["opencv_dnn"],
            dynamic_batch_supported=False,
            license_provenance={
                "code_license": yunet_meta["license"]["code_license"],
                "weights_license": yunet_meta["license"]["weights_license"],
                "training_dataset": "WIDER FACE",
                "commercial_use_status": yunet_meta["commercial_use_status"],
                "production_eligible": False,
            },
            availability_status=yunet_status,
            parameters=[
                ParameterSpec(
                    name="resize_mode",
                    label="Input Resize Policy",
                    control_type="select",
                    value_type="string",
                    default="preserve_aspect_max_dim",
                    description="Controls how source images are resized before YuNet inference (`preserve_aspect_max_dim` avoids upscaling smaller images and caps longest edge).",
                    impact="both",
                    options=[
                        {"value": "preserve_aspect_max_dim", "label": "Preserve Aspect Ratio (Max Dim Cap)"},
                        {"value": "native", "label": "Native Unresized Resolution"},
                        {"value": "fixed", "label": "Fixed Width × Height"},
                    ],
                ),
                ParameterSpec(
                    name="max_input_dim",
                    label="Maximum Input Dimension (px)",
                    control_type="slider",
                    value_type="int",
                    default=1280,
                    min_value=160,
                    max_value=2240,
                    step=32,
                    description="Longest-side cap when `resize_mode='preserve_aspect_max_dim'`. `1280` recovers 100% of `<32 px` faces on the NASA multi-scale split.",
                    impact="both",
                    visible_when={"resize_mode": "preserve_aspect_max_dim"},
                ),
                ParameterSpec(
                    name="fixed_width",
                    label="Fixed Input Width (px)",
                    control_type="number",
                    value_type="int",
                    default=640,
                    min_value=64,
                    max_value=2048,
                    step=32,
                    description="Target width used only when `resize_mode='fixed'`.",
                    impact="both",
                    visible_when={"resize_mode": "fixed"},
                ),
                ParameterSpec(
                    name="fixed_height",
                    label="Fixed Input Height (px)",
                    control_type="number",
                    value_type="int",
                    default=640,
                    min_value=64,
                    max_value=2048,
                    step=32,
                    description="Target height used only when `resize_mode='fixed'`.",
                    impact="both",
                    visible_when={"resize_mode": "fixed"},
                ),
                ParameterSpec(
                    name="score_threshold",
                    label="Confidence Threshold",
                    control_type="slider",
                    value_type="float",
                    default=0.60,
                    min_value=0.05,
                    max_value=0.98,
                    step=0.01,
                    description="Minimum objectness × classification score required to emit a face candidate (`0.60` P1.1 baseline; `0.75` suppresses spacesuit hardware FP).",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="nms_mode",
                    label="NMS Suppression Mode",
                    control_type="select",
                    value_type="string",
                    default="hard_greedy",
                    description="OpenCV FaceDetectorYN applies internal greedy Hard-NMS (`hard_greedy`). Selecting `gaussian_soft_nms` runs FaceDetectorYN with permissive internal NMS (`0.95`) and applies Bodla et al. Gaussian Soft-NMS decay post-hoc.",
                    impact="accuracy",
                    options=[
                        {"value": "hard_greedy", "label": "Hard Greedy NMS (Native)"},
                        {"value": "gaussian_soft_nms", "label": "Gaussian Soft-NMS (P3 Research)"},
                    ],
                ),
                ParameterSpec(
                    name="nms_threshold",
                    label="Hard-NMS IoU Threshold",
                    control_type="slider",
                    value_type="float",
                    default=0.30,
                    min_value=0.10,
                    max_value=0.85,
                    step=0.05,
                    description="IoU overlap threshold above which lower-scoring boxes are suppressed in Hard-NMS.",
                    impact="accuracy",
                    visible_when={"nms_mode": "hard_greedy"},
                ),
                ParameterSpec(
                    name="soft_nms_sigma",
                    label="Gaussian Soft-NMS Sigma",
                    control_type="slider",
                    value_type="float",
                    default=0.50,
                    min_value=0.10,
                    max_value=1.00,
                    step=0.05,
                    description="Gaussian decay width `sigma` in `s_i <- s_i * exp(-IoU^2 / sigma)`.",
                    impact="accuracy",
                    visible_when={"nms_mode": "gaussian_soft_nms"},
                ),
                ParameterSpec(
                    name="top_k",
                    label="Max Pre-NMS Candidates (top_k)",
                    control_type="number",
                    value_type="int",
                    default=5000,
                    min_value=50,
                    max_value=10000,
                    step=50,
                    description="Maximum candidate boxes retained prior to NMS.",
                    impact="both",
                ),
                ParameterSpec(
                    name="min_face_size_px",
                    label="Minimum Face Size Filter (px)",
                    control_type="slider",
                    value_type="float",
                    default=0.0,
                    min_value=0.0,
                    max_value=256.0,
                    step=4.0,
                    description="Post-detection filter discarding boxes with `min(width, height) < min_face_size_px` in original image pixels (`0` disables filtering).",
                    impact="accuracy",
                ),
            ],
        ),
        AlgorithmSpec(
            algorithm_id="scrfd_500m_research",
            name="SCRFD-500M-BNKPS (InsightFace Buffalo_SC)",
            category="detection",
            description=(
                "Sample and Computation Redistribution 3-stride [8, 16, 32] PAFPN detector with "
                "2 anchors/cell and 5-point landmark regression on ONNX Runtime CPU or CUDA (`io_binding`), "
                "including P3 canvas-padding, dual-axis inverse-scaling (`P2-AUD-08`), and Soft-NMS variants."
            ),
            paper_reference="Guo et al., 'Sample and Computation Redistribution for Efficient Face Detection', ICLR 2022",
            source_implementation="src/detection/scrfd_detector.py :: SCRFDDetector + src/p3_research/algorithmic_variants.py :: ResearchSCRFDWrapper",
            model_id="scrfd_500m_research",
            checkpoint_path=scrfd_meta["local_path"],
            model_sha256=scrfd_sha,
            input_requirements={
                "color_order": "RGB",
                "value_range": "[-1.0, 1.0] via (rgb - 127.5) / 128.0",
                "layout": "NCHW float32 [1, 3, H, W] (H, W multiples of 32)",
                "dynamic_spatial": True,
            },
            output_schema={
                "detections": "list[DetectedFace] (bbox xywh in original px, 5-pt landmarks in original px, confidence)",
            },
            supported_devices=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            supported_precisions=["fp32"],
            supported_runtimes=["onnxruntime"],
            dynamic_batch_supported=False,
            license_provenance={
                "code_license": scrfd_meta["license"]["code_license"],
                "weights_license": scrfd_meta["license"]["weights_license"],
                "training_dataset": "WIDER FACE (InsightFace 5-pt annotations)",
                "commercial_use_status": scrfd_meta["commercial_use_status"],
                "production_eligible": False,
            },
            availability_status=scrfd_status,
            parameters=[
                ParameterSpec(
                    name="device",
                    label="Execution Device",
                    control_type="select",
                    value_type="string",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    description="ONNX Runtime execution provider (`cuda` uses `CUDAExecutionProvider` with explicit `io_binding`; `cpu` uses `CPUExecutionProvider`).",
                    impact="performance",
                    options=[
                        {"value": "cuda", "label": "CUDA GPU (CUDAExecutionProvider)"},
                        {"value": "cpu", "label": "CPU (CPUExecutionProvider)"},
                    ]
                    if torch.cuda.is_available()
                    else [{"value": "cpu", "label": "CPU (CPUExecutionProvider)"}],
                ),
                ParameterSpec(
                    name="max_dim",
                    label="Max Canvas Dimension (stride-32 px)",
                    control_type="slider",
                    value_type="int",
                    default=1280,
                    min_value=320,
                    max_value=1920,
                    step=32,
                    description="Target maximum spatial dimension (rounded to stride-32). `1280` achieves 100% small-face recall on the NASA 86-face benchmark.",
                    impact="both",
                ),
                ParameterSpec(
                    name="canvas_mode",
                    label="Canvas Padding Strategy",
                    control_type="select",
                    value_type="string",
                    default="square_padded",
                    description="`square_padded` keeps GPU tensor shape constant (`max_dim × max_dim`), avoiding ORT cuDNN plan rebuild P95 spikes on mixed aspect ratios; `stride32_tight` trims zero-padding to `ceil(w/32)*32 × ceil(h/32)*32`, reducing P50 on fixed-aspect streams (`MOD-P3-03`).",
                    impact="performance",
                    options=[
                        {"value": "square_padded", "label": "Square Zero-Padded Canvas (Stable GPU P95)"},
                        {"value": "stride32_tight", "label": "Tight Stride-32 Aspect Canvas (Lower P50)"},
                    ],
                ),
                ParameterSpec(
                    name="inverse_scale_mode",
                    label="Inverse Coordinate Scaling",
                    control_type="select",
                    value_type="string",
                    default="exact_dual_xy",
                    description="`exact_dual_xy` (`MOD-P3-02` / `P2-AUD-08` fix) projects x and y independently using `(new_w/orig_w, new_h/orig_h)`, eliminating sub-pixel aspect-rounding drift vs. `single_scalar`.",
                    impact="accuracy",
                    options=[
                        {"value": "exact_dual_xy", "label": "Exact Dual-Axis (scale_x, scale_y) [Recommended]"},
                        {"value": "single_scalar", "label": "Single Scalar (new_h / orig_h) [P2 Historical]"},
                    ],
                ),
                ParameterSpec(
                    name="allow_upscale",
                    label="Upscale Images Smaller Than max_dim",
                    control_type="boolean",
                    value_type="bool",
                    default=False,
                    description="When enabled, images with `max(w, h) < max_dim` are upsampled up to `max_dim` (`InsightFace det_size` behavior); when disabled, native scale is preserved inside the padded canvas.",
                    impact="both",
                ),
                ParameterSpec(
                    name="score_threshold",
                    label="Confidence Threshold",
                    control_type="slider",
                    value_type="float",
                    default=0.50,
                    min_value=0.05,
                    max_value=0.95,
                    step=0.01,
                    description="Minimum anchor score across strides `[8, 16, 32]` (`0.50` default; plateau `F1=1.0` across `[0.30, 0.60]` on NASA 86-face split).",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="nms_mode",
                    label="NMS Method",
                    control_type="select",
                    value_type="string",
                    default="hard_greedy",
                    description="Suppression algorithm (`hard_greedy` step suppression vs. `gaussian_soft_nms` smooth overlap decay).",
                    impact="accuracy",
                    options=[
                        {"value": "hard_greedy", "label": "Hard Greedy NMS"},
                        {"value": "gaussian_soft_nms", "label": "Gaussian Soft-NMS"},
                    ],
                ),
                ParameterSpec(
                    name="nms_threshold",
                    label="Hard-NMS IoU Threshold",
                    control_type="slider",
                    value_type="float",
                    default=0.40,
                    min_value=0.10,
                    max_value=0.85,
                    step=0.05,
                    description="Maximum allowed IoU overlap before suppressing lower-confidence boxes.",
                    impact="accuracy",
                    visible_when={"nms_mode": "hard_greedy"},
                ),
                ParameterSpec(
                    name="soft_nms_sigma",
                    label="Gaussian Soft-NMS Sigma",
                    control_type="slider",
                    value_type="float",
                    default=0.50,
                    min_value=0.10,
                    max_value=1.00,
                    step=0.05,
                    description="Gaussian overlap decay parameter `sigma`.",
                    impact="accuracy",
                    visible_when={"nms_mode": "gaussian_soft_nms"},
                ),
                ParameterSpec(
                    name="top_k",
                    label="Max Detections Returned",
                    control_type="number",
                    value_type="int",
                    default=500,
                    min_value=1,
                    max_value=5000,
                    step=10,
                    description="Upper bound on detections returned per image.",
                    impact="both",
                ),
                ParameterSpec(
                    name="min_face_size_px",
                    label="Minimum Face Size Filter (px)",
                    control_type="slider",
                    value_type="float",
                    default=0.0,
                    min_value=0.0,
                    max_value=256.0,
                    step=4.0,
                    description="Optional post-filter requiring `min(w, h) >= min_face_size_px` in original image coordinates.",
                    impact="accuracy",
                ),
            ],
        ),
        # =====================================================================
        # 2. LANDMARK & ALIGNMENT ALGORITHMS
        # =====================================================================
        AlgorithmSpec(
            algorithm_id="five_point_canonical_aligner",
            name="5-Point Canonical Geometric Face Aligner (Umeyama / Affine)",
            category="landmark_alignment",
            description=(
                "Maps 5 ordered facial landmarks (`right_eye_subject`, `left_eye_subject`, `nose_tip`, "
                "`right_mouth_subject`, `left_mouth_subject`) onto the canonical ArcFace `112×112` "
                "template via closed-form 4-DoF proper similarity (`Umeyama`, `det > 0`), 6-DoF "
                "least-squares affine, or zoomed templates (`MOD-P3-06`, `MOD-P3-07`)."
            ),
            paper_reference="Umeyama, 'Least-Squares Estimation of Transformation Parameters Between Two Point Patterns', IEEE TPAMI 1991",
            source_implementation="src/alignment/face_aligner.py :: FaceAligner + src/p3_research/algorithmic_variants.py :: align_face_research",
            model_id=None,
            checkpoint_path=None,
            model_sha256=None,
            input_requirements={
                "image": "BGR uint8 HWC",
                "landmarks": "5 × 2 float64 coordinates in canonical subject eye/nose/mouth order",
            },
            output_schema={
                "aligned_bgr": "uint8 [crop_h, crop_w, 3]",
                "affine_matrix_2x3": "float64 [2, 3]",
                "crop_bounds_xyxy": "tuple[int, int, int, int] back-projected in original image",
            },
            supported_devices=["cpu"],
            supported_precisions=["fp64_geometry"],
            supported_runtimes=["numpy_opencv"],
            dynamic_batch_supported=True,
            license_provenance={
                "code_license": "First-Party Project Implementation (MIT/BSD NumPy & OpenCV primitives)",
                "production_eligible": True,
            },
            availability_status="AVAILABLE",
            parameters=[
                ParameterSpec(
                    name="transform_mode",
                    label="Geometric Transformation",
                    control_type="select",
                    value_type="string",
                    default="similarity_umeyama_4dof",
                    description="`similarity_umeyama_4dof` preserves isotropic aspect ratio (scale + rotation + translation, no shear/reflection — verified optimal in `EXP-P3-02`); `affine_least_squares_6dof` adds shear/anisotropic scale; `similarity_zoom_out_10pct` includes 10% wider context.",
                    impact="accuracy",
                    options=[
                        {"value": "similarity_umeyama_4dof", "label": "4-DoF Proper Similarity (Umeyama) [Baseline]"},
                        {"value": "affine_least_squares_6dof", "label": "6-DoF Full Least-Squares Affine"},
                        {"value": "similarity_zoom_out_10pct", "label": "4-DoF Similarity (+10% Context Zoom-Out)"},
                    ],
                ),
                ParameterSpec(
                    name="interpolation",
                    label="Warp Interpolation Kernel",
                    control_type="select",
                    value_type="string",
                    default="linear",
                    description="OpenCV `warpAffine` resampling filter (`linear` is `31.2×` faster than `lanczos4` with `<0.03 pp` accuracy delta in `EXP-P3-02`).",
                    impact="performance",
                    options=[
                        {"value": "nearest", "label": "Nearest Neighbor (INTER_NEAREST)"},
                        {"value": "linear", "label": "Bilinear (INTER_LINEAR) [Optimal]"},
                        {"value": "cubic", "label": "Bicubic 4×4 (INTER_CUBIC)"},
                        {"value": "lanczos4", "label": "Lanczos 8×8 (INTER_LANCZOS4)"},
                    ],
                ),
                ParameterSpec(
                    name="crop_width",
                    label="Output Crop Width (px)",
                    control_type="number",
                    value_type="int",
                    default=112,
                    min_value=64,
                    max_value=224,
                    step=16,
                    description="Aligned crop width (`112` required by `AdaFace`, `SFace`, and `MobileFaceNet`).",
                    impact="both",
                ),
                ParameterSpec(
                    name="crop_height",
                    label="Output Crop Height (px)",
                    control_type="number",
                    value_type="int",
                    default=112,
                    min_value=64,
                    max_value=224,
                    step=16,
                    description="Aligned crop height (`112` canonical).",
                    impact="both",
                ),
                ParameterSpec(
                    name="zoom_factor",
                    label="Template Scale / Zoom Factor",
                    control_type="slider",
                    value_type="float",
                    default=1.00,
                    min_value=0.70,
                    max_value=1.35,
                    step=0.02,
                    description="Scales canonical template coordinates around crop center (`1.00` = exact ArcFace template; `<1.0` zooms out; `>1.0` zooms in).",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="extra_rotation_deg",
                    label="Controlled Roll Perturbation (deg)",
                    control_type="slider",
                    value_type="float",
                    default=0.0,
                    min_value=-30.0,
                    max_value=30.0,
                    step=1.0,
                    description="Controlled in-plane roll misalignment (`EXP-P3-03`) applied around crop center after alignment (`0°` = normal).",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="translate_x_px",
                    label="Template X Offset (px)",
                    control_type="slider",
                    value_type="float",
                    default=0.0,
                    min_value=-20.0,
                    max_value=20.0,
                    step=1.0,
                    description="Horizontal shift applied to destination template coordinates (`0` = canonical).",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="translate_y_px",
                    label="Template Y Offset (px)",
                    control_type="slider",
                    value_type="float",
                    default=0.0,
                    min_value=-20.0,
                    max_value=20.0,
                    step=1.0,
                    description="Vertical shift applied to destination template coordinates (`0` = canonical).",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="jitter_sigma_px",
                    label="Controlled Landmark Noise σ (px)",
                    control_type="slider",
                    value_type="float",
                    default=0.0,
                    min_value=0.0,
                    max_value=12.0,
                    step=0.5,
                    description="Synthetic Gaussian noise (`N(0, σ²)`) added to source landmarks before alignment for robustness testing (`EXP-P3-03`).",
                    impact="accuracy",
                ),
            ],
        ),
        # =====================================================================
        # 3. FACE QUALITY ASSESSMENT ALGORITHMS
        # =====================================================================
        AlgorithmSpec(
            algorithm_id="hybrid_classical_and_norm_quality",
            name="P1 Classical + P3 Extended 12-Indicator Face Quality Suite",
            category="quality",
            description=(
                "Computes deterministic spatial/photometric measurements (`Laplacian variance`, "
                "`Tenengrad`, `brightness`, `contrast`, `exposure`, `8×8 JPEG blockiness`, `inter-ocular px`, "
                "`yaw asymmetry`, `roll deg`, `bbox relative area`, `detector confidence`) alongside "
                "pre-L2 deep embedding norm `||z_raw||_2` (`EXP-P3-05`), strictly separating raw pixel "
                "measurements from model-dependent feature norms."
            ),
            paper_reference="P1 Classical Quality + Kim et al., CVPR 2022 (`||z_raw||_2` proxy; audited in P3.1)",
            source_implementation="src/quality/face_quality.py :: FaceQualityAssessor + src/p3_research/algorithmic_variants.py :: compute_extended_quality_features",
            model_id=None,
            checkpoint_path=None,
            model_sha256=None,
            input_requirements={"source_image_bgr": "uint8 HWC", "aligned_crop_bgr": "uint8 112x112x3"},
            output_schema={"classical_p1": "QualityAssessment", "extended_p3": "dict[str, float]"},
            supported_devices=["cpu", "cuda"],
            supported_precisions=["fp32", "fp16"],
            supported_runtimes=["numpy_opencv"],
            dynamic_batch_supported=True,
            license_provenance={"code_license": "First-Party Project Implementation", "production_eligible": True},
            availability_status="AVAILABLE",
            parameters=[
                ParameterSpec(
                    name="min_face_width_px",
                    label="Warning Threshold: Min Face Width (px)",
                    control_type="slider",
                    value_type="float",
                    default=float(p1_cfg["quality"]["warning_thresholds"]["min_face_width_px"]),
                    min_value=16.0,
                    max_value=160.0,
                    step=4.0,
                    description="Triggers `face_dimensions_below_minimum` warning below this width.",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="min_sharpness_laplacian_var",
                    label="Warning Threshold: Min Laplacian Sharpness",
                    control_type="slider",
                    value_type="float",
                    default=float(p1_cfg["quality"]["warning_thresholds"]["min_sharpness_laplacian_var"]),
                    min_value=5.0,
                    max_value=250.0,
                    step=5.0,
                    description="Triggers `low_sharpness_possible_blur` warning when crop Laplacian variance falls below threshold.",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="min_brightness_mean",
                    label="Warning Threshold: Min Brightness",
                    control_type="slider",
                    value_type="float",
                    default=float(p1_cfg["quality"]["warning_thresholds"]["min_brightness_mean"]),
                    min_value=5.0,
                    max_value=120.0,
                    step=5.0,
                    description="Triggers `low_brightness_underexposed` warning.",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="max_brightness_mean",
                    label="Warning Threshold: Max Brightness",
                    control_type="slider",
                    value_type="float",
                    default=float(p1_cfg["quality"]["warning_thresholds"]["max_brightness_mean"]),
                    min_value=160.0,
                    max_value=250.0,
                    step=5.0,
                    description="Triggers `high_brightness_overexposed` warning.",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="min_contrast_std",
                    label="Warning Threshold: Min Contrast Std",
                    control_type="slider",
                    value_type="float",
                    default=float(p1_cfg["quality"]["warning_thresholds"]["min_contrast_std"]),
                    min_value=5.0,
                    max_value=60.0,
                    step=2.0,
                    description="Triggers `low_contrast` warning.",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="reject_quantile",
                    label="EVRC Quality Reject Fraction",
                    control_type="slider",
                    value_type="float",
                    default=0.05,
                    min_value=0.0,
                    max_value=0.40,
                    step=0.05,
                    description="Fraction of lowest-quality items filtered out in quality-based rejection analysis.",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="primary_quality_metric",
                    label="Active Ranking / Filtering Metric",
                    control_type="select",
                    value_type="string",
                    default="learned_embedding_l2_norm",
                    description="Selects which implemented quality indicator drives ranking and EVRC filtering (`learned_embedding_l2_norm` achieved `ρ = 0.5903` on AdaFace genuine cosine in `EXP-P3-05`).",
                    impact="accuracy",
                    options=[
                        {"value": "learned_embedding_l2_norm", "label": "Pre-L2 Embedding Norm ||z_raw||_2"},
                        {"value": "sharpness_tenengrad", "label": "Tenengrad Sobel Sharpness"},
                        {"value": "sharpness_laplacian_var", "label": "Laplacian Variance Sharpness"},
                        {"value": "classical_composite_score", "label": "Classical 4-Factor Composite [0..1]"},
                        {"value": "exposure_score", "label": "Photometric Exposure Score [0..1]"},
                        {"value": "inter_ocular_px", "label": "Inter-Ocular Distance (px)"},
                        {"value": "detector_confidence", "label": "Face Detector Confidence"},
                    ],
                ),
            ],
        ),
        # =====================================================================
        # 4. FACE RECOGNITION EMBEDDING MODELS
        # =====================================================================
        AlgorithmSpec(
            algorithm_id="adaface_ir50_webface4m",
            name="AdaFace IR-50 WebFace4M (512-D Backbone)",
            category="recognition",
            description=(
                "50-layer Improved Residual Network (`43,585,600` params) trained with Quality-Adaptive "
                "Margin (`AdaFace`) on WebFace4M (`4.2M` images). Supports ONNX Opset 17 `FP32` / `FP16` "
                "and native PyTorch `.pt` checkpoint execution, `GuardedFixedBatchEmbedder`, and `Flip-TTA`."
            ),
            paper_reference="Kim, Jain & Liu, 'AdaFace: Quality Adaptive Margin for Face Recognition', CVPR 2022",
            source_implementation="src/recognition/resnet50_embedder.py :: ReferenceResNet50Embedder + src/recognition/adaface_ir50.py",
            model_id="adaface_ir50_webface4m",
            checkpoint_path=ada_onnx_meta["local_path"],
            model_sha256=ada_sha,
            input_requirements={
                "crop_size_wh": [112, 112],
                "color_order": "RGB",
                "mean": [127.5, 127.5, 127.5],
                "std": [127.5, 127.5, 127.5],
                "formula": "(rgb_uint8 - 127.5) / 127.5 -> [-1.0, 1.0]",
            },
            output_schema={"embedding_dim": 512, "l2_normalized": True, "raw_norm_quality_proxy": True},
            supported_devices=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            supported_precisions=["fp16", "fp32"] if torch.cuda.is_available() else ["fp32"],
            supported_runtimes=["onnxruntime", "pytorch"],
            dynamic_batch_supported=True,
            license_provenance={
                "code_license": ada_onnx_meta["license"]["code_license"],
                "weights_license": ada_onnx_meta["license"]["weights_license"],
                "training_dataset": "WebFace4M (205,990 IDs, 4.2M imgs)",
                "commercial_use_status": ada_onnx_meta["commercial_use_status"],
                "production_eligible": False,
                "checkpoints": {
                    "onnx_fp32": {"path": ada_onnx_meta["local_path"], "sha256": ada_onnx_meta["sha256"]},
                    "onnx_fp16": {"path": ada_fp16_meta["local_path"], "sha256": ada_fp16_meta["sha256"]},
                    "pytorch_pt": {"path": ada_pt_meta["local_path"], "sha256": ada_pt_meta["sha256"]},
                },
            },
            availability_status=ada_status,
            parameters=[
                ParameterSpec(
                    name="runtime",
                    label="Execution Runtime",
                    control_type="select",
                    value_type="string",
                    default="onnxruntime",
                    description="Selects `onnxruntime` (`.onnx` FP32 or FP16 graph) or native `pytorch` (`adaface_ir50_webface4m.pt`).",
                    impact="performance",
                    options=[
                        {"value": "onnxruntime", "label": "ONNX Runtime (FP32 / FP16 .onnx)"},
                        {"value": "pytorch", "label": "PyTorch Native (.pt state_dict)"},
                    ],
                ),
                ParameterSpec(
                    name="device",
                    label="Execution Device",
                    control_type="select",
                    value_type="string",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    description="Compute device (`FP16` requires `cuda`).",
                    impact="performance",
                    options=[
                        {"value": "cuda", "label": "CUDA GPU (RTX 3060)"},
                        {"value": "cpu", "label": "CPU"},
                    ]
                    if torch.cuda.is_available()
                    else [{"value": "cpu", "label": "CPU"}],
                ),
                ParameterSpec(
                    name="precision",
                    label="Numeric Precision",
                    control_type="select",
                    value_type="string",
                    default="fp16" if torch.cuda.is_available() else "fp32",
                    description="Tensor precision (`fp16` on CUDA halves model VRAM `166 -> 83 MiB` with `0.999999` cosine parity vs. `fp32`).",
                    impact="performance",
                    options=[
                        {"value": "fp16", "label": "FP16 Half Precision (CUDA Only)"},
                        {"value": "fp32", "label": "FP32 Single Precision"},
                    ]
                    if torch.cuda.is_available()
                    else [{"value": "fp32", "label": "FP32 Single Precision"}],
                ),
                ParameterSpec(
                    name="batch_size",
                    label="Inference Batch Size",
                    control_type="select",
                    value_type="int",
                    default=8,
                    description="Mini-batch size for multi-face extraction (`B=8` optimal `292.8 faces/s` with fixed batch padding).",
                    impact="performance",
                    options=[{"value": b, "label": f"Batch = {b}"} for b in (1, 2, 4, 8, 16, 32)],
                ),
                ParameterSpec(
                    name="batch_policy",
                    label="GPU Batch Scheduling Policy",
                    control_type="select",
                    value_type="string",
                    default="guarded_fixed_padded",
                    description="`guarded_fixed_padded` (`MOD-P3-11` / `P2-AUD-06` fix) pads tail chunks to constant `batch_size`, speeding up mixed-face streams `7.39×` by avoiding ORT cuDNN plan rebuilds.",
                    impact="performance",
                    options=[
                        {"value": "guarded_fixed_padded", "label": "Guarded Fixed-Batch Zero-Padding [7.39× Faster]"},
                        {"value": "dynamic_unpadded", "label": "Dynamic Unpadded Chunks"},
                    ],
                ),
                ParameterSpec(
                    name="flip_tta",
                    label="Horizontal-Flip TTA (2× Passes)",
                    control_type="boolean",
                    value_type="bool",
                    default=False,
                    description="Sums `z(x) + z(hflip(x))` prior to L2 normalization (`2.001×` latency; note P3.1 finding that realized FAR at `1/1500` calib threshold shifted `1 -> 4` FPs on LFW).",
                    impact="both",
                ),
                ParameterSpec(
                    name="l2_normalize",
                    label="Unit Hypersphere L2 Normalization",
                    control_type="boolean",
                    value_type="bool",
                    default=True,
                    description="Projects raw `512`-D embedding onto `S^511` (`||e||_2 = 1.0`). Disabling exposes raw radial magnitude distortion (`EXP-P3-06`).",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="template_aggregation",
                    label="Multi-Shot Template Aggregation",
                    control_type="select",
                    value_type="string",
                    default="learned_norm_weighted",
                    description="Spherical pooling strategy when aggregating multiple enrollment shots per template (`EXP-P3-05`).",
                    impact="accuracy",
                    options=[
                        {"value": "learned_norm_weighted", "label": "Softmax ||z_raw||_2 Weighted Spherical Pool"},
                        {"value": "classical_quality_weighted", "label": "Classical Quality Weighted Spherical Pool"},
                        {"value": "uniform_mean", "label": "Uniform Spherical Mean"},
                        {"value": "single_first", "label": "Single First Shot Only"},
                    ],
                ),
                ParameterSpec(
                    name="locked_preprocessing_summary",
                    label="Required Model Preprocessing (Locked Default)",
                    control_type="select",
                    value_type="string",
                    default="RGB | (x - 127.5) / 127.5 -> [-1, 1] | 112x112",
                    description="Verified upstream CVLFace preprocessing specification. Protected against accidental silent edits unless Experimental Override is explicitly enabled.",
                    impact="accuracy",
                    read_only=True,
                    options=[
                        {
                            "value": "RGB | (x - 127.5) / 127.5 -> [-1, 1] | 112x112",
                            "label": "RGB | (x - 127.5) / 127.5 -> [-1, 1] | 112×112 (Locked)",
                        }
                    ],
                ),
                ParameterSpec(
                    name="allow_experimental_preprocess_override",
                    label="Enable Experimental Preprocessing Override (Warning)",
                    control_type="boolean",
                    value_type="bool",
                    default=False,
                    description="Explicitly unlocks color-order / normalization override for controlled failure-mode ablation ONLY.",
                    impact="accuracy",
                    experimental_warning="Overriding a pretrained recognizer's required color order or pixel scaling severely degrades biometric accuracy and invalidates calibrated thresholds.",
                ),
                ParameterSpec(
                    name="override_color_order",
                    label="Override Color Order",
                    control_type="select",
                    value_type="string",
                    default="RGB",
                    description="Experimental color channel order override.",
                    impact="accuracy",
                    visible_when={"allow_experimental_preprocess_override": True},
                    options=[{"value": "RGB", "label": "RGB (Canonical)"}, {"value": "BGR", "label": "BGR (Channel-Swapped Ablation)"}],
                ),
            ],
        ),
        AlgorithmSpec(
            algorithm_id="mobilefacenet_w600k_research",
            name="MobileFaceNet W600K (InsightFace Buffalo_S, 512-D)",
            category="recognition",
            description=(
                "Compact `3.4M`-parameter depthwise-separable CNN trained with fixed-margin `ArcFace` "
                "on `WebFace600K` (`512`-D output, dynamic batch ONNX `FP32`)."
            ),
            paper_reference="Chen et al., 'MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices', CCBR 2018",
            source_implementation="src/recognition/candidate_embedders.py :: MobileFaceNetEmbedder",
            model_id="mobilefacenet_w600k_research",
            checkpoint_path=mbf_meta["local_path"],
            model_sha256=mbf_sha,
            input_requirements={
                "crop_size_wh": [112, 112],
                "color_order": "RGB",
                "mean": [127.5, 127.5, 127.5],
                "std": [127.5, 127.5, 127.5],
                "formula": "(rgb_uint8 - 127.5) / 127.5 -> [-1.0, 1.0]",
            },
            output_schema={"embedding_dim": 512, "l2_normalized": True, "raw_norm_quality_proxy": False},
            supported_devices=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            supported_precisions=["fp32"],
            supported_runtimes=["onnxruntime"],
            dynamic_batch_supported=True,
            license_provenance={
                "code_license": mbf_meta["license"]["code_license"],
                "weights_license": mbf_meta["license"]["weights_license"],
                "training_dataset": "WebFace600K (InsightFace Non-Commercial)",
                "commercial_use_status": mbf_meta["commercial_use_status"],
                "production_eligible": False,
            },
            availability_status=mbf_status,
            parameters=[
                ParameterSpec(
                    name="device",
                    label="Execution Device",
                    control_type="select",
                    value_type="string",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    description="ONNX Runtime execution provider.",
                    impact="performance",
                    options=[
                        {"value": "cuda", "label": "CUDA GPU (CUDAExecutionProvider)"},
                        {"value": "cpu", "label": "CPU (CPUExecutionProvider)"},
                    ]
                    if torch.cuda.is_available()
                    else [{"value": "cpu", "label": "CPU (CPUExecutionProvider)"}],
                ),
                ParameterSpec(
                    name="batch_size",
                    label="Inference Batch Size",
                    control_type="select",
                    value_type="int",
                    default=8,
                    description="Dynamic ONNX batch size (`1..32`).",
                    impact="performance",
                    options=[{"value": b, "label": f"Batch = {b}"} for b in (1, 2, 4, 8, 16, 32)],
                ),
                ParameterSpec(
                    name="batch_policy",
                    label="GPU Batch Scheduling Policy",
                    control_type="select",
                    value_type="string",
                    default="guarded_fixed_padded",
                    description="Constant-shape zero-padded batching vs. dynamic unpadded chunks.",
                    impact="performance",
                    options=[
                        {"value": "guarded_fixed_padded", "label": "Guarded Fixed-Batch Zero-Padding"},
                        {"value": "dynamic_unpadded", "label": "Dynamic Unpadded Chunks"},
                    ],
                ),
                ParameterSpec(
                    name="flip_tta",
                    label="Horizontal-Flip TTA (2× Passes)",
                    control_type="boolean",
                    value_type="bool",
                    default=False,
                    description="Sums `z(x) + z(hflip(x))` before L2 normalization (`2.054×` latency).",
                    impact="both",
                ),
                ParameterSpec(
                    name="l2_normalize",
                    label="Unit Hypersphere L2 Normalization",
                    control_type="boolean",
                    value_type="bool",
                    default=True,
                    description="Normalizes `512`-D vector to unit L2 length.",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="locked_preprocessing_summary",
                    label="Required Model Preprocessing (Locked Default)",
                    control_type="select",
                    value_type="string",
                    default="RGB | (x - 127.5) / 127.5 -> [-1, 1] | 112x112",
                    description="Verified InsightFace `arcface_onnx.py` preprocessing contract.",
                    impact="accuracy",
                    read_only=True,
                    options=[
                        {
                            "value": "RGB | (x - 127.5) / 127.5 -> [-1, 1] | 112x112",
                            "label": "RGB | (x - 127.5) / 127.5 -> [-1, 1] | 112×112 (Locked)",
                        }
                    ],
                ),
                ParameterSpec(
                    name="allow_experimental_preprocess_override",
                    label="Enable Experimental Preprocessing Override (Warning)",
                    control_type="boolean",
                    value_type="bool",
                    default=False,
                    description="Unlocks experimental color-order override with warning.",
                    impact="accuracy",
                    experimental_warning="Changing color order from RGB to BGR breaks MobileFaceNet channel alignment.",
                ),
                ParameterSpec(
                    name="override_color_order",
                    label="Override Color Order",
                    control_type="select",
                    value_type="string",
                    default="RGB",
                    description="Experimental color channel order override.",
                    impact="accuracy",
                    visible_when={"allow_experimental_preprocess_override": True},
                    options=[{"value": "RGB", "label": "RGB (Canonical)"}, {"value": "BGR", "label": "BGR (Channel-Swapped Ablation)"}],
                ),
            ],
        ),
        AlgorithmSpec(
            algorithm_id="sface_2021dec",
            name="SFace MobileFaceNet (OpenCV Zoo 2021-Dec, 128-D)",
            category="recognition",
            description=(
                "OpenCV Zoo `128`-D MobileFaceNet trained with Sigmoid-Constrained Hypersphere (`SFace`) "
                "loss. Graph input is static `[1, 3, 112, 112]` (`batch_size=1` only) and embeds "
                "`(x - 127.5) * 0.0078125` inside its first two ONNX scalar nodes (`P2` verified)."
            ),
            paper_reference="Zhong et al., 'SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition', IEEE TIP 2021",
            source_implementation="src/recognition/candidate_embedders.py :: SFaceEmbedder",
            model_id="sface_2021dec",
            checkpoint_path=sface_meta["local_path"],
            model_sha256=sface_sha,
            input_requirements={
                "crop_size_wh": [112, 112],
                "color_order": "RGB (swapRB=True)",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "formula": "Raw RGB float32 [0, 255] (internal ONNX nodes subtract 127.5 & multiply 1/128)",
            },
            output_schema={"embedding_dim": 128, "l2_normalized": True, "static_batch_1_only": True},
            supported_devices=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            supported_precisions=["fp32"],
            supported_runtimes=["onnxruntime", "opencv_dnn"],
            dynamic_batch_supported=False,
            license_provenance={
                "code_license": sface_meta["license"]["code_license"],
                "weights_license": sface_meta["license"]["weights_license"],
                "training_dataset": "MS-Celeb-1M / CASIA-WebFace / VGGFace2 (upstream SFace)",
                "commercial_use_status": sface_meta["commercial_use_status"],
                "production_eligible": False,
            },
            availability_status=sface_status,
            parameters=[
                ParameterSpec(
                    name="runtime",
                    label="Execution Runtime",
                    control_type="select",
                    value_type="string",
                    default="onnxruntime",
                    description="`onnxruntime` supports CPU and CUDA; `opencv_dnn` (`cv2.FaceRecognizerSF`) supports CPU only.",
                    impact="performance",
                    options=[
                        {"value": "onnxruntime", "label": "ONNX Runtime (CUDA / CPU)"},
                        {"value": "opencv_dnn", "label": "OpenCV FaceRecognizerSF (CPU Only)"},
                    ],
                ),
                ParameterSpec(
                    name="device",
                    label="Execution Device",
                    control_type="select",
                    value_type="string",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    description="Compute device (`opencv_dnn` requires `cpu`).",
                    impact="performance",
                    visible_when={"runtime": "onnxruntime"},
                    options=[
                        {"value": "cuda", "label": "CUDA GPU"},
                        {"value": "cpu", "label": "CPU"},
                    ]
                    if torch.cuda.is_available()
                    else [{"value": "cpu", "label": "CPU"}],
                ),
                ParameterSpec(
                    name="flip_tta",
                    label="Horizontal-Flip TTA (2× Sequential Passes)",
                    control_type="boolean",
                    value_type="bool",
                    default=False,
                    description="Sums `z(x) + z(hflip(x))` before L2 normalization.",
                    impact="both",
                ),
                ParameterSpec(
                    name="l2_normalize",
                    label="Unit Hypersphere L2 Normalization",
                    control_type="boolean",
                    value_type="bool",
                    default=True,
                    description="Normalizes `128`-D vector onto `S^127`.",
                    impact="accuracy",
                ),
                ParameterSpec(
                    name="locked_preprocessing_summary",
                    label="Required Model Preprocessing (Locked Default)",
                    control_type="select",
                    value_type="string",
                    default="RGB | Raw [0, 255] (Internal -127.5/128 in ONNX) | Static B=1",
                    description="Verified in P2 against `cv2.FaceRecognizerSF.feature()` (`cosine = 1.0000000`). Static `B=1` enforced.",
                    impact="accuracy",
                    read_only=True,
                    options=[
                        {
                            "value": "RGB | Raw [0, 255] (Internal -127.5/128 in ONNX) | Static B=1",
                            "label": "RGB | Raw [0, 255] (Internal ONNX Scale) | Static B=1",
                        }
                    ],
                ),
                ParameterSpec(
                    name="allow_experimental_preprocess_override",
                    label="Enable Experimental Preprocessing Override (Warning)",
                    control_type="boolean",
                    value_type="bool",
                    default=False,
                    description="Unlocks experimental BGR/RGB override with warning.",
                    impact="accuracy",
                    experimental_warning="Double-normalizing SFace inputs or swapping RGB->BGR degrades cosine separation.",
                ),
                ParameterSpec(
                    name="override_color_order",
                    label="Override Color Order",
                    control_type="select",
                    value_type="string",
                    default="RGB",
                    description="Experimental color channel order override.",
                    impact="accuracy",
                    visible_when={"allow_experimental_preprocess_override": True},
                    options=[{"value": "RGB", "label": "RGB (Canonical)"}, {"value": "BGR", "label": "BGR (Channel-Swapped Ablation)"}],
                ),
            ],
        ),
        AlgorithmSpec(
            algorithm_id="reference_resnet50_baseline",
            name="Reference ResNet50 Baseline Adapter Slot (Retired / Weights Absent Locally)",
            category="recognition",
            description=(
                "Historical P1 adapter entry preserved for provenance transparency (`RETIRED` in P3). "
                "Local checkpoint is intentionally absent (`MissingModelArtifactError` / `BLOCKED` status; "
                "never substituted with ImageNet weights)."
            ),
            paper_reference="Historical Milestone P1 Specification (Retired in Milestone P3)",
            source_implementation="src/recognition/resnet50_embedder.py :: ReferenceResNet50Embedder",
            model_id="reference_resnet50_baseline",
            checkpoint_path=ref_meta["local_path"],
            model_sha256=ref_sha,
            input_requirements={"crop_size_wh": [112, 112], "color_order": "RGB"},
            output_schema={"embedding_dim": 512, "status": "BLOCKED_MISSING_REFERENCE_RESNET50_WEIGHTS"},
            supported_devices=["cpu", "cuda"],
            supported_precisions=["fp32"],
            supported_runtimes=["onnxruntime", "pytorch"],
            dynamic_batch_supported=True,
            license_provenance={
                "code_license": "User-Supplied / External",
                "weights_license": "Pending / Retired in P3",
                "commercial_use_status": "pending_internal_verification",
                "production_eligible": False,
            },
            availability_status=ref_status,
            parameters=[],
        ),
        # =====================================================================
        # 5. PAIRWISE SIMILARITY / DISTANCE MATCHING METRICS
        # =====================================================================
        AlgorithmSpec(
            algorithm_id="pairwise_vector_matcher",
            name="Hypersphere & Euclidean Pairwise Verification Matcher",
            category="matching",
            description=(
                "Evaluates `1:1` similarity/distance (`cosine`, `normalized_l2`, `unnormalized_l2`, "
                "`normalized_l1`) and compares against calibrated operating thresholds (`Folds 1–5` "
                "out-of-sample LFW calibration or manual user threshold)."
            ),
            paper_reference="EXP-P3-06 Exact Unit-Sphere Metric Equivalence (`||u - v||_2^2 = 2 - 2<u, v>`)",
            source_implementation="src/p3_research/algorithmic_variants.py :: compute_pair_scores, evaluate_calibrated_verification",
            model_id=None,
            checkpoint_path=None,
            model_sha256=None,
            input_requirements={"raw_a": "(N, D) float32", "raw_b": "(N, D) float32"},
            output_schema={"score": "float64 (higher = more similar)", "is_match": "bool"},
            supported_devices=["cpu"],
            supported_precisions=["fp64"],
            supported_runtimes=["numpy"],
            dynamic_batch_supported=True,
            license_provenance={"code_license": "First-Party Project Implementation", "production_eligible": True},
            availability_status="AVAILABLE",
            parameters=[
                ParameterSpec(
                    name="metric",
                    label="Similarity / Distance Metric",
                    control_type="select",
                    value_type="string",
                    default="cosine",
                    description="`cosine` and `normalized_l2` (`-||u - v||_2`) are strictly monotonic on unit vectors (`0.000000` ROC AUC difference); `unnormalized_l2` exposes radial norm mismatch.",
                    impact="accuracy",
                    options=[
                        {"value": "cosine", "label": "Cosine Similarity <u, v> [-1..1]"},
                        {"value": "normalized_l2", "label": "Negated Normalized Euclidean (-||u - v||_2)"},
                        {"value": "unnormalized_l2", "label": "Negated Unnormalized L2 (-||z_a - z_b||_2)"},
                        {"value": "normalized_l1", "label": "Negated Normalized Manhattan (-||u - v||_1)"},
                    ],
                ),
                ParameterSpec(
                    name="threshold",
                    label="Decision Threshold (τ)",
                    control_type="slider",
                    value_type="float",
                    default=0.2169,
                    min_value=-2.00,
                    max_value=1.00,
                    step=0.002,
                    description="Verification decision boundary (`score >= τ -> Match`). Default `0.2169` corresponds to AdaFace IR-50 FP16 calibrated on LFW Folds 1–5 at `FAR <= 10^-3`.",
                    impact="accuracy",
                ),
            ],
        ),
        # =====================================================================
        # 6. 1:N VECTOR RETRIEVAL & ANN INDEXES
        # =====================================================================
        AlgorithmSpec(
            algorithm_id="exact_flat_index",
            name="ExactFlat Brute-Force Vector Search (CPU BLAS / CUDA FP32 / CUDA FP16)",
            category="retrieval",
            description=(
                "Exact `100%` recall exhaustive inner-product (`ip`) or squared-`L2` (`l2`) search "
                "using NumPy SGEMM on CPU or `torch.matmul` TensorCores (`FP32` / `FP16`) on GPU."
            ),
            paper_reference="Exact Linear Scan Baseline (`EXP-P3-08`: `1.04 ms` Q=1, `37,110 QPS` Q=64 at N=100K in CUDA FP16)",
            source_implementation="src/matching/retrieval_indexes.py :: ExactFlatIndex",
            model_id=None,
            checkpoint_path=None,
            model_sha256=None,
            input_requirements={"vectors": "(N, D) float32"},
            output_schema={"scores": "(Q, K) float32", "ids": "(Q, K) int64"},
            supported_devices=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            supported_precisions=["fp16", "fp32"] if torch.cuda.is_available() else ["fp32"],
            supported_runtimes=["numpy_cpu", "torch_cuda"],
            dynamic_batch_supported=True,
            license_provenance={"code_license": "First-Party Project Implementation", "production_eligible": True},
            availability_status="AVAILABLE",
            parameters=[
                ParameterSpec(
                    name="backend",
                    label="Compute Backend & Precision",
                    control_type="select",
                    value_type="string",
                    default="torch_cuda_fp16" if torch.cuda.is_available() else "numpy_cpu",
                    description="Selects CPU BLAS `FP32` or PyTorch CUDA `FP32`/`FP16` TensorCore matrix multiplication.",
                    impact="performance",
                    options=[
                        {"value": "torch_cuda_fp16", "label": "PyTorch CUDA FP16 TensorCore"},
                        {"value": "torch_cuda_fp32", "label": "PyTorch CUDA FP32"},
                        {"value": "numpy_cpu", "label": "NumPy CPU FP32 BLAS"},
                    ]
                    if torch.cuda.is_available()
                    else [{"value": "numpy_cpu", "label": "NumPy CPU FP32 BLAS"}],
                ),
                ParameterSpec(
                    name="metric",
                    label="Index Metric",
                    control_type="select",
                    value_type="string",
                    default="ip",
                    description="Inner product (`ip`, exact cosine on unit vectors) or squared Euclidean (`l2`).",
                    impact="accuracy",
                    options=[
                        {"value": "ip", "label": "Inner Product / Cosine (ip)"},
                        {"value": "l2", "label": "Squared Euclidean Distance (l2)"},
                    ],
                ),
                ParameterSpec(
                    name="batch_query_size",
                    label="Throughput Batch Query Size",
                    control_type="select",
                    value_type="int",
                    default=64,
                    description="Query batch size `Q` used when measuring batched QPS.",
                    impact="performance",
                    options=[{"value": q, "label": f"Q = {q}"} for q in (1, 16, 64, 128)],
                ),
            ],
        ),
        AlgorithmSpec(
            algorithm_id="ivf_flat_index",
            name="IVF-Flat Coarse Voronoi Partitioned Index",
            category="retrieval",
            description=(
                "Inverted File Index partitioning the unit hypersphere into `nlist` `MiniBatchKMeans` "
                "Voronoi cells and scanning only the top `nprobe` cells per query."
            ),
            paper_reference="Sivic & Zisserman, ICCV 2003 / Jégou et al., IEEE TPAMI 2011 (`EXP-P3-08`)",
            source_implementation="src/matching/retrieval_indexes.py :: IVFFlatIndex",
            model_id=None,
            checkpoint_path=None,
            model_sha256=None,
            input_requirements={"vectors": "(N, D) float32"},
            output_schema={"scores": "(Q, K) float32", "ids": "(Q, K) int64"},
            supported_devices=["cpu"],
            supported_precisions=["fp32"],
            supported_runtimes=["numpy_sklearn"],
            dynamic_batch_supported=True,
            license_provenance={"code_license": "First-Party Project Implementation", "production_eligible": True},
            availability_status="AVAILABLE",
            parameters=[
                ParameterSpec(
                    name="nlist",
                    label="Voronoi Partitions (nlist)",
                    control_type="select",
                    value_type="int",
                    default=256,
                    description="Number of coarse `MiniBatchKMeans` centroids.",
                    impact="both",
                    options=[{"value": v, "label": f"nlist = {v}"} for v in (32, 64, 128, 256, 512)],
                ),
                ParameterSpec(
                    name="nprobe",
                    label="Probed Cells per Query (nprobe)",
                    control_type="slider",
                    value_type="int",
                    default=16,
                    min_value=1,
                    max_value=128,
                    step=1,
                    description="Number of nearest Voronoi cells searched per query (`Recall@K` vs. latency knob).",
                    impact="both",
                ),
                ParameterSpec(
                    name="train_sample_max",
                    label="Coarse K-Means Training Sample Cap",
                    control_type="number",
                    value_type="int",
                    default=25000,
                    min_value=1000,
                    max_value=100000,
                    step=1000,
                    description="Maximum gallery vectors subsampled to fit coarse centroids.",
                    impact="both",
                ),
            ],
        ),
        AlgorithmSpec(
            algorithm_id="ivf_pq_index",
            name="IVF-PQ Residual Product Quantization Index (ADC)",
            category="retrieval",
            description=(
                "Coarse `IVF` partitioning + `M`-subspace `8`-bit Residual Product Quantization (`64×` "
                "code compression at `M=32` on `512`-D vectors) with Asymmetric Distance Computation (`ADC`)."
            ),
            paper_reference="Jégou, Douze & Schmid, 'Product Quantization for Nearest Neighbor Search', IEEE TPAMI 2011",
            source_implementation="src/matching/retrieval_indexes.py :: IVFPQIndex",
            model_id=None,
            checkpoint_path=None,
            model_sha256=None,
            input_requirements={"vectors": "(N, D) float32 (D divisible by m_subspaces)"},
            output_schema={"scores": "(Q, K) float32 (negated ADC squared L2)", "ids": "(Q, K) int64"},
            supported_devices=["cpu"],
            supported_precisions=["uint8_pq_codes"],
            supported_runtimes=["numpy_sklearn"],
            dynamic_batch_supported=True,
            license_provenance={"code_license": "First-Party Project Implementation", "production_eligible": True},
            availability_status="AVAILABLE",
            parameters=[
                ParameterSpec(
                    name="nlist",
                    label="Coarse Voronoi Cells (nlist)",
                    control_type="select",
                    value_type="int",
                    default=256,
                    description="Number of coarse centroids for residual computation.",
                    impact="both",
                    options=[{"value": v, "label": f"nlist = {v}"} for v in (32, 64, 128, 256)],
                ),
                ParameterSpec(
                    name="m_subspaces",
                    label="PQ Subquantizers (M)",
                    control_type="select",
                    value_type="int",
                    default=32,
                    description="Number of orthogonal subspaces (`M=32` = `32 B/vec` = `64×` compression on `512`-D; must divide `embedding_dim`).",
                    impact="both",
                    options=[{"value": v, "label": f"M = {v} ({v} B/vec)"} for v in (8, 16, 32, 64)],
                ),
                ParameterSpec(
                    name="nbits",
                    label="Bits per Subquantizer",
                    control_type="select",
                    value_type="int",
                    default=8,
                    description="Codebook bit-width (`8` bits = `K* = 256` centroids per subspace).",
                    impact="accuracy",
                    options=[
                        {"value": 8, "label": "8 bits (256 centroids/subspace)"},
                        {"value": 6, "label": "6 bits (64 centroids/subspace)"},
                        {"value": 4, "label": "4 bits (16 centroids/subspace)"},
                    ],
                ),
                ParameterSpec(
                    name="nprobe",
                    label="Probed Cells per Query (nprobe)",
                    control_type="slider",
                    value_type="int",
                    default=16,
                    min_value=1,
                    max_value=64,
                    step=1,
                    description="Number of inverted lists scanned via ADC lookup tables.",
                    impact="both",
                ),
                ParameterSpec(
                    name="distance_mode",
                    label="Distance Computation Mode",
                    control_type="select",
                    value_type="string",
                    default="adc_residual_lut",
                    description="Asymmetric Distance Computation (`ADC`) between uncompressed query residual and quantized database codes.",
                    impact="accuracy",
                    read_only=True,
                    options=[{"value": "adc_residual_lut", "label": "Residual ADC Lookup Table (Exact LUT)"}],
                ),
            ],
        ),
        AlgorithmSpec(
            algorithm_id="hnsw_index",
            name="HNSW Multi-Layer Proximity Graph Index (Python/NumPy Research Impl)",
            category="retrieval",
            description=(
                "Hierarchical Navigable Small World graph (`M`, `efConstruction`, `efSearch`). "
                "Pure Python/NumPy implementation faithfully measures algorithmic `Recall@K` and graph "
                "structure (capped at `N <= 15,000` to respect laptop build time; see `P3.1` audit)."
            ),
            paper_reference="Malkov & Yashunin, 'Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs', IEEE TPAMI 2020",
            source_implementation="src/matching/retrieval_indexes.py :: HNSWIndex",
            model_id=None,
            checkpoint_path=None,
            model_sha256=None,
            input_requirements={"vectors": "(N, D) unit float32 (N <= 15,000)"},
            output_schema={"scores": "(Q, K) float32", "ids": "(Q, K) int64"},
            supported_devices=["cpu"],
            supported_precisions=["fp32"],
            supported_runtimes=["python_numpy"],
            dynamic_batch_supported=True,
            license_provenance={"code_license": "First-Party Project Implementation", "production_eligible": True},
            availability_status="AVAILABLE",
            parameters=[
                ParameterSpec(
                    name="m",
                    label="Bi-Directional Links per Node (M)",
                    control_type="select",
                    value_type="int",
                    default=16,
                    description="Maximum outgoing edges per node on upper layers (`2*M` on layer 0).",
                    impact="both",
                    options=[{"value": v, "label": f"M = {v}"} for v in (8, 12, 16, 24, 32)],
                ),
                ParameterSpec(
                    name="ef_construction",
                    label="Build Beam Width (efConstruction)",
                    control_type="slider",
                    value_type="int",
                    default=80,
                    min_value=20,
                    max_value=200,
                    step=10,
                    description="Candidate heap width during graph insertion.",
                    impact="both",
                ),
                ParameterSpec(
                    name="ef_search",
                    label="Query Beam Width (efSearch)",
                    control_type="slider",
                    value_type="int",
                    default=64,
                    min_value=8,
                    max_value=256,
                    step=8,
                    description="Dynamic candidate list size during layer-0 greedy/beam search.",
                    impact="both",
                ),
            ],
        ),
    ]
    return {spec.algorithm_id: spec for spec in registry}


PIPELINE_STAGE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "stage_id": "image_input",
        "label": "1. Image Input & Decoding",
        "optional": False,
        "depends_on": [],
        "description": "Decodes native BGR uint8 frames (`cv2.imdecode`) and computes SHA-256 provenance.",
    },
    {
        "stage_id": "face_detection",
        "label": "2. Face & Landmark Detection",
        "optional": False,
        "depends_on": ["image_input"],
        "category": "detection",
        "description": "Detects face bounding boxes and 5-point landmarks in original pixel coordinates.",
    },
    {
        "stage_id": "alignment",
        "label": "3. Geometric Face Alignment",
        "optional": False,
        "depends_on": ["face_detection"],
        "category": "landmark_alignment",
        "description": "Warps each detected face via 5-point landmarks onto the canonical `112×112` recognition crop.",
    },
    {
        "stage_id": "quality_assessment",
        "label": "4. Face Quality Assessment",
        "optional": True,
        "depends_on": ["face_detection", "alignment"],
        "category": "quality",
        "reorderable_with": "face_recognition",
        "description": "Evaluates classical spatial/photometric indicators and optional post-recognition `||z_raw||_2` quality filtering.",
    },
    {
        "stage_id": "face_recognition",
        "label": "5. Face Feature Extraction",
        "optional": False,
        "depends_on": ["alignment"],
        "category": "recognition",
        "description": "Applies model-specific preprocessing and extracts `512`-D or `128`-D raw facial embeddings.",
    },
    {
        "stage_id": "embedding_normalization",
        "label": "6. Hypersphere Normalization",
        "optional": True,
        "depends_on": ["face_recognition"],
        "description": "Projects raw embeddings onto the unit L2 hypersphere (`||e||_2 = 1.0`).",
    },
    {
        "stage_id": "matching_retrieval",
        "label": "7. Matching / 1:N Retrieval",
        "optional": True,
        "depends_on": ["face_recognition"],
        "category": "matching",
        "description": "Computes pairwise verification scores or `1:N` candidate index retrieval.",
    },
    {
        "stage_id": "evaluation",
        "label": "8. Result & Latency Evaluation",
        "optional": True,
        "depends_on": ["face_detection"],
        "description": "Aggregates stage latencies, memory telemetry, and ground-truth accuracy metrics.",
    },
]


def validate_pipeline_configuration(
    pipeline_cfg: dict[str, Any],
    registry: dict[str, AlgorithmSpec] | None = None,
) -> PipelineValidationResult:
    """Validate cross-stage dependencies, static batch limits, device/precision rules, and model weights."""
    reg = registry or build_algorithm_registry()
    errors: list[str] = []
    warnings: list[str] = []

    det_stage = pipeline_cfg.get("detection", {})
    align_stage = pipeline_cfg.get("alignment", {})
    qual_stage = pipeline_cfg.get("quality", {})
    rec_stage = pipeline_cfg.get("recognition", {})
    norm_stage = pipeline_cfg.get("normalization", {"enabled": True})
    ret_stage = pipeline_cfg.get("retrieval", {"enabled": False})
    stage_order = pipeline_cfg.get(
        "stage_order",
        [s["stage_id"] for s in PIPELINE_STAGE_DEFINITIONS],
    )

    # 1. Stage ordering dependency checks
    idx_map = {sid: idx for idx, sid in enumerate(stage_order)}
    required_precedence = [
        ("image_input", "face_detection"),
        ("face_detection", "alignment"),
        ("alignment", "face_recognition"),
        ("face_recognition", "embedding_normalization"),
    ]
    for before, after in required_precedence:
        if before in idx_map and after in idx_map and idx_map[before] > idx_map[after]:
            errors.append(f"Invalid stage order: '{before}' must execute before '{after}'.")

    # 2. Detector checks
    det_id = str(det_stage.get("algorithm_id", "yunet_2023mar"))
    if det_id not in reg or reg[det_id].category != "detection":
        errors.append(f"Unknown or non-detection algorithm_id='{det_id}'.")
    else:
        det_spec = reg[det_id]
        det_params = det_stage.get("params", {})
        dev = str(det_params.get("device", det_spec.supported_devices[0])).lower()
        if dev not in det_spec.supported_devices:
            errors.append(f"Detector '{det_id}' does not support device='{dev}' (supported: {det_spec.supported_devices}).")
        if det_id == "scrfd_500m_research":
            max_dim = int(det_params.get("max_dim", 1280))
            if max_dim < 64 or max_dim % 32 != 0:
                errors.append(f"SCRFD `max_dim` must be a positive multiple of 32; got {max_dim}.")

    # 3. Alignment vs recognizer input crop check
    if not bool(det_stage.get("enable_landmarks", True)):
        errors.append("5-point alignment cannot execute when landmark output is disabled.")
    align_params = align_stage.get("params", {})
    cw = int(align_params.get("crop_width", 112))
    ch = int(align_params.get("crop_height", 112))

    # 4. Recognition model checks
    rec_enabled = bool(rec_stage.get("enabled", True))
    rec_id = str(rec_stage.get("algorithm_id", "adaface_ir50_webface4m"))
    rec_dim = 512
    if rec_enabled:
        if rec_id not in reg or reg[rec_id].category != "recognition":
            errors.append(f"Unknown recognition model '{rec_id}'.")
        else:
            rec_spec = reg[rec_id]
            rec_dim = int(rec_spec.output_schema.get("embedding_dim", 512))
            if rec_spec.availability_status != "AVAILABLE":
                if bool(rec_stage.get("strict_weights", True)):
                    errors.append(
                        f"Recognizer '{rec_id}' weights are missing locally (`{rec_spec.checkpoint_path}`). "
                        "Select an available research model or disable `strict_weights` to observe explicit BLOCKED status."
                    )
                else:
                    warnings.append(
                        f"Recognizer '{rec_id}' has no local weights; pipeline will emit explicit `BLOCKED_MISSING_REFERENCE_RESNET50_WEIGHTS`."
                    )
            rec_params = rec_stage.get("params", {})
            r_rt = str(rec_params.get("runtime", rec_spec.supported_runtimes[0])).lower()
            r_dev = str(rec_params.get("device", rec_spec.supported_devices[0])).lower()
            r_prec = str(rec_params.get("precision", rec_spec.supported_precisions[0])).lower()
            r_bs = int(rec_params.get("batch_size", 1 if not rec_spec.dynamic_batch_supported else 8))
            r_pol = str(rec_params.get("batch_policy", "dynamic_unpadded"))

            if r_rt not in rec_spec.supported_runtimes:
                errors.append(f"Recognizer '{rec_id}' does not support runtime='{r_rt}' (supported: {rec_spec.supported_runtimes}).")
            if r_rt == "opencv_dnn" and r_dev.startswith("cuda"):
                errors.append("OpenCV DNN `FaceRecognizerSF` supports `device='cpu'` only.")
            if r_prec not in rec_spec.supported_precisions:
                errors.append(f"Recognizer '{rec_id}' does not support precision='{r_prec}' (supported: {rec_spec.supported_precisions}).")
            if r_prec == "fp16" and r_dev == "cpu":
                errors.append("FP16 recognition inference requires `device='cuda'`; CPU FP16 is unsupported.")
            if not rec_spec.dynamic_batch_supported and r_bs != 1:
                errors.append(
                    f"Recognizer '{rec_id}' has a static `[1, 3, 112, 112]` ONNX graph (`dynamic_batch_supported=False`) and cannot accept `batch_size={r_bs}`."
                )
            if not rec_spec.dynamic_batch_supported and r_pol == "guarded_fixed_padded" and r_bs > 1:
                errors.append(f"`GuardedFixedBatchEmbedder` rejects static B=1 model '{rec_id}' for `batch_size={r_bs}`.")
            req_wh = rec_spec.input_requirements.get("crop_size_wh", [112, 112])
            if [cw, ch] != list(req_wh):
                warnings.append(
                    f"Alignment crop size `({cw}×{ch})` differs from '{rec_id}' native input `{req_wh}`; preprocessor will bilinearly resize to `{req_wh}`."
                )
            if bool(rec_params.get("allow_experimental_preprocess_override", False)):
                warnings.append(
                    "Experimental preprocessing override is active — recognition accuracy and calibrated thresholds are intentionally altered."
                )

    # 5. Retrieval compatibility checks
    if bool(ret_stage.get("enabled", False)):
        if not rec_enabled:
            errors.append("Retrieval index stage requires `face_recognition` to be enabled.")
        idx_id = str(ret_stage.get("algorithm_id", "exact_flat_index"))
        idx_params = ret_stage.get("params", {})
        if idx_id not in reg or reg[idx_id].category != "retrieval":
            errors.append(f"Unknown retrieval index '{idx_id}'.")
        elif idx_id == "ivf_pq_index":
            m_sub = int(idx_params.get("m_subspaces", 32))
            if rec_dim % m_sub != 0:
                errors.append(
                    f"`IVF-PQ` requires `embedding_dim ({rec_dim})` to be divisible by `m_subspaces ({m_sub})`."
                )
        elif idx_id == "hnsw_index":
            if not bool(norm_stage.get("enabled", True)):
                errors.append("`HNSWIndex` inner-product distance `1 - <u, v>` requires unit L2-normalized embeddings.")

    # 6. Quality ordering warning
    if bool(qual_stage.get("enabled", True)):
        q_metric = str(qual_stage.get("params", {}).get("primary_quality_metric", "learned_embedding_l2_norm"))
        if q_metric == "learned_embedding_l2_norm" and not rec_enabled:
            warnings.append("`learned_embedding_l2_norm` quality proxy requires `face_recognition` enabled; falling back to classical sharpness.")

    return PipelineValidationResult(
        valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        normalized_config=pipeline_cfg,
    )


def get_full_registry_payload() -> dict[str, Any]:
    """Return the complete JSON-serializable registry payload for `/api/v1/registry`."""
    reg = build_algorithm_registry()
    categories = [
        {"id": "detection", "label": "Face Detection"},
        {"id": "landmark_alignment", "label": "Landmark & Alignment"},
        {"id": "quality", "label": "Face Quality Assessment"},
        {"id": "recognition", "label": "Face Recognition"},
        {"id": "matching", "label": "Pairwise Matching"},
        {"id": "retrieval", "label": "1:N Search & Retrieval"},
    ]
    return {
        "schema_version": "1.0.0",
        "categories": categories,
        "algorithms": [spec.to_dict() for spec in reg.values()],
        "canonical_landmarks_112": {
            "order": [
                "right_eye_subject",
                "left_eye_subject",
                "nose_tip",
                "right_mouth_subject",
                "left_mouth_subject",
            ],
            "viewer_labels": [
                "Viewer-Left Eye (Subj R)",
                "Viewer-Right Eye (Subj L)",
                "Nose Tip",
                "Viewer-Left Mouth (Subj R)",
                "Viewer-Right Mouth (Subj L)",
            ],
            "points_112": ARCFACE_5PT_112.tolist(),
        },
        "stage_definitions": PIPELINE_STAGE_DEFINITIONS,
    }


def export_registry_schema_json(output_rel_path: str = "configs/workbench_registry_schema.json") -> str:
    """Persist canonical registry & parameter schema JSON for documentation and tests."""
    out_path = resolve_project_path(output_rel_path)
    payload = get_full_registry_payload()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_path)
