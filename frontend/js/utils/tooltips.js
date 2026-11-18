/**
 * Configuration-Driven Scientific Tooltip, Contextual Help & Busy-Guard Engine (`frontend/js/utils/tooltips.js`).
 *
 * Features:
 *  - Central technical glossary verified against `src/` implementations (Detection, Alignment,
 *    Quality, Recognition, Matching, Retrieval, and Biometric Evaluation metrics).
 *  - Renders into a single fixed `#wb-tooltip-portal` (`role="tooltip"`) outside `.wb-panel`
 *    containers so `overflow: hidden` never clips explanations.
 *  - Supports mouse hover (`120 ms` delay), keyboard focus (`focusin`), touch/click toggle on
 *    `.wb-help-trigger` (`?`) buttons, `aria-describedby`, and `Escape` dismissal.
 *  - Provides `withBusyButton(btn, fn)` to prevent duplicate jobs on rapid repeated clicks.
 */

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export const TECHNICAL_GLOSSARY = {
  // ---------------- DETECTION ----------------
  score_threshold: {
    title: "Detection Confidence Threshold (score_threshold)",
    meaning: "Minimum detector confidence score (objectness × face classification) required to emit a candidate bounding box.",
    effect: "Higher values reduce false positives (e.g., 0.80 suppresses Apollo 12 spacesuit hardware) but can miss tiny or occluded faces; lower values increase recall and false alarms.",
    limitation: "Detector ranking score in [0, 1] — NOT a calibrated statistical probability of a human face.",
  },
  nms_threshold: {
    title: "Hard-NMS IoU Threshold (nms_threshold)",
    meaning: "Intersection-over-Union (IoU) overlap ceiling above which lower-scoring overlapping boxes are discarded during greedy Non-Maximum Suppression.",
    effect: "Lower values (e.g., 0.25–0.30) aggressively merge duplicate boxes on one face; higher values (0.50+) retain closely packed crowd faces at the risk of duplicate boxes.",
    limitation: "Active only when NMS Mode is set to Hard Greedy NMS.",
  },
  nms_mode: {
    title: "NMS Suppression Mode (nms_mode)",
    meaning: "Selects standard greedy IoU suppression (`hard_greedy`) or Bodla et al. continuous score decay (`gaussian_soft_nms`).",
    effect: "Gaussian Soft-NMS decays overlapping scores smoothly instead of hard-zeroing neighbors in dense crowds.",
    limitation: "For YuNet, Soft-NMS runs post-hoc after permissive internal OpenCV candidate extraction (0.95).",
  },
  soft_nms_sigma: {
    title: "Gaussian Soft-NMS Sigma (soft_nms_sigma)",
    meaning: "Width parameter σ in the Gaussian penalty s_i ← s_i · exp(−IoU² / σ) applied to overlapping candidate boxes.",
    effect: "Smaller σ penalizes overlapping boxes more sharply; larger σ preserves nearby detections in tight groups.",
    limitation: "Candidates whose decayed score falls below score_threshold are pruned.",
  },
  resize_mode: {
    title: "Input Resize Policy (resize_mode)",
    meaning: "Controls how source frames are scaled before YuNet inference (`preserve_aspect_max_dim`, `native`, or `fixed`).",
    effect: "Preserving aspect ratio prevents facial geometric stretching and avoids upscaling small images unnecessarily.",
    limitation: "`fixed` (e.g., 320×320) distorts non-square aspect ratios and drops <32 px small-face recall.",
  },
  max_input_dim: {
    title: "Maximum Input Resolution (max_input_dim / max_dim)",
    meaning: "Longest-edge pixel cap for aspect-preserving detector input tensors.",
    effect: "Higher resolution (1280–1600 px) recovers small <32 px faces (100% recall on Ceremony) at quadratic O(H·W) latency cost; 320–640 px runs faster for close-up portraits.",
    limitation: "SCRFD requires multiples of stride 32 (e.g., 640, 960, 1280).",
  },
  max_dim: {
    title: "SCRFD Maximum Input Dimension (max_dim)",
    meaning: "Longest-edge pixel target (stride-32 multiple) for SCRFD multi-stride [8, 16, 32] feature pyramid inference.",
    effect: "Increasing from 640 to 1280 boosts small-face recall dramatically while increasing tensor FLOPs.",
    limitation: "Must be a positive multiple of 32 (>= 64 px).",
  },
  canvas_mode: {
    title: "SCRFD Canvas Construction Mode (canvas_mode)",
    meaning: "Controls whether resized non-square images are placed onto a square `max_dim × max_dim` zero-padded tensor (`square_padded`) or Tight Stride-32 canvas (`stride32_tight`).",
    effect: "`stride32_tight` eliminates ~32.5% empty zero-padding on 3:2 frames (1280×864 vs. 1280×1280), reducing GPU compute without hurting accuracy.",
    limitation: "Dynamic spatial shapes trigger brief one-time ONNX Runtime kernel compilation on first distinct shape.",
  },
  inverse_scale_mode: {
    title: "Coordinate Inverse-Scale Mapping (inverse_scale_mode)",
    meaning: "Controls how detected boxes/landmarks map back to native image pixels (`exact_dual_xy` vs. legacy `single_scalar`).",
    effect: "`exact_dual_xy` divides X and Y by independent `resized_w/orig_w` and `resized_h/orig_h` ratios, eliminating up to ~0.65 px integer-rounding drift (`P2-AUD-08`).",
    limitation: "`single_scalar` is retained strictly to reproduce historical P2 baseline behavior.",
  },
  min_face_size_px: {
    title: "Minimum Face Size Filter (min_face_size_px)",
    meaning: "Post-detection spatial gate discarding boxes with `min(width, height) < min_face_size_px` in original image pixels.",
    effect: "Filters tiny background faces unsuitable for 112×112 recognition; 0 disables filtering.",
    limitation: "Applied in native image pixel coordinates after inverse scaling.",
  },
  top_k: {
    title: "Pre-NMS Candidate Cap (top_k)",
    meaning: "Maximum highest-scoring anchor predictions retained prior to Non-Maximum Suppression.",
    effect: "Higher values prevent dropping true faces in ultra-dense crowds; lower values reduce post-processing overhead.",
    limitation: "Valid range: [50 .. 10000].",
  },

  // ---------------- ALIGNMENT ----------------
  transform_mode: {
    title: "Geometric Alignment Transformation (4-DoF vs. 6-DoF vs. +10% Zoom-Out)",
    meaning: "Maps 5 detected facial landmarks to the 112×112 ArcFace canonical template (`ARCFACE_5PT_112`).",
    effect: "`similarity_umeyama_4dof` (scale, rotation, translation) preserves exact facial angles (`anisotropy = 0`); `affine_least_squares_6dof` adds shear/aspect stretch to reduce 5-pt fit residual (`~1.84 px` vs `~3.31 px`) but slightly degrades LFW accuracy; `similarity_zoom_out_10pct` retains 10% wider context.",
    limitation: "With 5 2D landmarks (10 equations), both 4-DoF and 6-DoF are overdetermined least-squares fits.",
  },
  canonical_template: {
    title: "Canonical 5-Point Template (ARCFACE_5PT_112)",
    meaning: "Standardized 112×112 target coordinates for eyes ([38.29, 51.70], [73.53, 51.50]), nose ([56.03, 71.74]), and mouth corners ([41.55, 92.37], [70.73, 92.20]).",
    effect: "Ensures every aligned face crop presents eyes, nose, and mouth at the exact spatial locations expected by IR-50, MobileFaceNet, and SFace CNNs.",
    limitation: "Overlays show geometric landmark correspondence to this template — never CNN keypoint descriptors.",
  },
  interpolation: {
    title: "Warp Pixel Interpolation Kernel (interpolation)",
    meaning: "Resampling kernel used by `cv2.warpAffine` (`nearest`, `linear`, `cubic`, `lanczos4`).",
    effect: "`linear` (bilinear) is the canonical training standard; `cubic`/`lanczos4` preserve higher-frequency edges when downsampling large faces; `nearest` introduces blocky aliasing.",
    limitation: "Changing interpolation slightly alters high-frequency crop texture and raw embedding norm.",
  },
  zoom_factor: {
    title: "Template Zoom Factor (zoom_factor)",
    meaning: "Radial scale multiplier around crop center (56, 56) applied to the canonical 5-point template.",
    effect: "`< 1.0` zooms out (more forehead/chin/background context); `> 1.0` zooms tightly into central facial features.",
    limitation: "Deviating far from `1.0` breaks spatial alignment with pretrained recognition backbones.",
  },
  jitter_sigma_px: {
    title: "Synthetic Landmark Gaussian Jitter σ (jitter_sigma_px)",
    meaning: "Controlled zero-mean Gaussian noise (in original image pixels) added to detected 5-point coordinates before alignment.",
    effect: "Simulates detector landmark localization error (`0 px` = unperturbed; `2–4 px` = moderate localization noise).",
    limitation: "Seeded deterministically via `seed` for reproducible sensitivity experiments.",
  },

  // ---------------- QUALITY ----------------
  sharpness_laplacian_var: {
    title: "Laplacian Variance Sharpness (Var(∇²I))",
    meaning: "Variance of the 2D second-derivative Laplacian response on the grayscale aligned 112×112 face crop.",
    effect: "Higher values (> 100) indicate crisp focus and high-frequency edge detail; blur collapses variance toward 0.",
    limitation: "Deterministic pixel statistic; sensitive to strong sensor noise or high-contrast glasses frames.",
  },
  sharpness_tenengrad: {
    title: "Tenengrad Sobel Gradient Energy",
    meaning: "Mean squared Sobel horizontal and vertical gradient magnitude (Gx² + Gy²) across the aligned face crop.",
    effect: "Complements Laplacian variance by measuring directional edge strength; decreases monotonically under Gaussian blur and downsampling.",
    limitation: "Raw pixel measurement (not normalized across lighting extremes).",
  },
  learned_embedding_l2_norm: {
    title: "Deep Embedding L2 Norm (||z_raw||₂)",
    meaning: "Euclidean magnitude of the raw pre-normalization embedding vector produced by the recognition CNN.",
    effect: "In AdaFace (trained with quality-adaptive margin), `||z_raw||₂` (~18–28) correlates with recognizable facial fidelity; severe blur reduces `||z_raw||₂`.",
    limitation: "Model-dependent uncalibrated proxy — magnitudes cannot be compared directly across AdaFace, MobileFaceNet, and SFace.",
  },
  classical_composite_score: {
    title: "Classical Composite Quality Score",
    meaning: "Bounded [0, 1] heuristic combining sharpness, exposure balance, contrast, inter-ocular resolution, and frontal symmetry.",
    effect: "Used to categorize crops (`HIGH >= 0.72`, `MEDIUM >= 0.50`, `LOW < 0.50`) and weight multi-shot templates.",
    limitation: "Does not include a dedicated neural face-occlusion classifier (disclosed as NOT_IMPLEMENTED).",
  },
  blur_sigma: {
    title: "Synthetic Gaussian Blur σ (blur_sigma)",
    meaning: "Standard deviation (in pixels) of isotropic Gaussian smoothing applied to the aligned 112×112 probe crop.",
    effect: "`0.0` = unmodified original crop; increasing σ (e.g., `1.5 – 3.5`) simulates out-of-focus optics and degrades self-recognition cosine.",
    limitation: "Synthetic probe perturbation for research ablation — never modifies dataset files on disk.",
  },
  effective_res: {
    title: "Effective Spatial Resolution (effective_res)",
    meaning: "Downsamples the 112×112 crop to `effective_res × effective_res` and upsamples back to 112×112 to simulate distant tiny faces.",
    effect: "`112×112` = native crop; `16×16` or `24×24` destroys fine ocular/nasal detail and tests low-resolution recognition breakdown.",
    limitation: "Synthetic degradation applied in memory only.",
  },
  gamma: {
    title: "Photometric Gamma Perturbation (γ)",
    meaning: "Non-linear power-law luminance mapping I_out = 255 · (I_in / 255)^(1/γ) on the aligned crop.",
    effect: "`1.0` = unmodified; `< 1.0` darkens midtones (underexposure); `> 1.5` washes out highlights (overexposure).",
    limitation: "Tests illumination invariance of CNN preprocessing.",
  },
  jpeg_quality: {
    title: "Synthetic JPEG Compression Quality (10–100)",
    meaning: "Re-encodes the aligned crop at the specified JPEG quality factor in memory.",
    effect: "`100` = uncompressed; `10–25` introduces 8×8 DCT blockiness artifacts (`jpeg_blockiness_ratio` rises).",
    limitation: "Applied in memory only to evaluate compression robustness.",
  },
  reject_quantile: {
    title: "EVRC Low-Quality Reject Fraction (reject_quantile)",
    meaning: "Fraction of lowest-quality faces filtered out under Error-versus-Reject Characteristic (EVRC) analysis.",
    effect: "Increasing from `0.00` to `0.20` rejects the bottom 20% lowest-quality crops to inspect verification error reduction on retained faces.",
    limitation: "Rejecting faces trades operational coverage (hit rate) for higher precision on retained samples.",
  },

  // ---------------- RECOGNITION & MATCHING ----------------
  embedding: {
    title: "Face Recognition Embedding Vector",
    meaning: "Compact d-dimensional feature vector (512-D for AdaFace/MobileFaceNet; 128-D for SFace) representing facial appearance on a hypersphere.",
    effect: "Faces of the same person map to nearby directions (high cosine similarity); different people map to near-orthogonal directions (~0.0 cosine).",
    limitation: "Global CNN representation — does not encode explicit local landmark keypoint descriptors.",
  },
  cosine: {
    title: "Cosine Similarity <u, v>",
    meaning: "Inner product between L2-normalized unit embedding vectors u and v in [−1.0, +1.0].",
    effect: "Higher values indicate stronger directional agreement (`> 0.2169` LFW TAR@FAR=1e-3 calibrated operating point for AdaFace FP16).",
    limitation: "RAW GEOMETRIC SIMILARITY SCORE — NOT a calibrated match probability (e.g., 0.75 does NOT mean '75% probability of same person').",
  },
  normalized_l2: {
    title: "Negated Normalized Euclidean Distance (−||u − v||₂)",
    meaning: "Negated Euclidean chord distance between unit-normalized embeddings (`−sqrt(2 − 2·cos(u,v))`) in [−2.0, 0.0].",
    effect: "Monotonically equivalent to cosine similarity on unit vectors (0.0 = identical; more negative = further apart).",
    limitation: "Negated so higher scores consistently mean stronger match (`score >= τ`) across all workbench gauges.",
  },
  unnormalized_l2: {
    title: "Negated Raw Euclidean Distance (−||z_a − z_b||₂)",
    meaning: "Negated L2 distance computed directly on unnormalized CNN output vectors prior to unit-sphere projection.",
    effect: "Conflates angular identity similarity with magnitude (`||z||₂` image quality), typically degrading EER relative to cosine.",
    limitation: "Scale depends on backbone output norm (~12 for SFace vs. ~25 for AdaFace).",
  },
  decision_threshold: {
    title: "Verification Decision Threshold (τ)",
    meaning: "Operating-point cutoff comparing geometric score `s` against `τ` (`s >= τ` → MATCH; `s < τ` → NON-MATCH).",
    effect: "Raising `τ` lowers False Accept Rate (FAR) at the cost of higher False Reject Rate (FRR); adjusting `τ` here updates verdicts instantaneously from cached embeddings (`0` re-inference).",
    limitation: "Thresholds are model- and metric-specific; never transfer a cosine threshold to L2 or across incompatible models.",
  },
  flip_tta: {
    title: "Horizontal Flip Test-Time Augmentation (Flip-TTA)",
    meaning: "Extracts embeddings for both the aligned crop and its horizontal mirror, sums/averages them, and L2-normalizes the result.",
    effect: "Improves pose/illumination symmetry robustness at exactly 2× recognition inference compute.",
    limitation: "Doubles raw pre-normalized vector sum magnitude before unit L2 normalization.",
  },
  precision: {
    title: "Inference Numerical Precision (FP32 vs. FP16)",
    meaning: "Selects 32-bit IEEE float (`fp32`) vs. 16-bit half-precision Tensor Core execution (`fp16` on CUDA).",
    effect: "AdaFace ORT FP16 halves model memory (87 MiB vs. 174 MiB) and boosts throughput with < 3×10⁻⁶ cosine drift vs. FP32.",
    limitation: "`fp16` requires CUDA GPU; CPU execution supports `fp32` only; MobileFaceNet/SFace ship `fp32` ONNX weights.",
  },
  batch_policy: {
    title: "Batch Execution Policy (Guarded Fixed vs. Dynamic Unpadded)",
    meaning: "`guarded_fixed_padded` pads partial tail batches to a constant `[B, 3, 112, 112]` shape and slices valid outputs; `dynamic_unpadded` passes variable tail sizes directly.",
    effect: "`guarded_fixed_padded` prevents ONNX Runtime / cuDNN Tensor Core engine recompilation spikes on tail batches (reaching ~293 faces/s at B=8).",
    limitation: "SFace has a static `[1, 3, 112, 112]` ONNX graph (`static_b1`) and rejects `batch_size > 1`.",
  },
  l2_normalize: {
    title: "Unit Hypersphere L2 Normalization (u = z / ||z||₂)",
    meaning: "Projects raw d-dimensional embeddings onto the unit hypersphere S^(d−1) so `||u||₂ = 1.0`.",
    effect: "Decouples angular identity direction from image-quality norm magnitude; essential for cosine and inner-product search.",
    limitation: "Disabling L2 normalization intentionally degrades cosine/IP matching for ablation study.",
  },
  template_aggregation: {
    title: "Multi-Shot Template Aggregation",
    meaning: "Combines multiple face crops of a subject into a single unit template via uniform mean, classical quality weighting, or `||z_raw||₂` norm weighting.",
    effect: "Quality/norm weighting down-weights blurry or off-angle frames before final L2 normalization.",
    limitation: "Previewed across the active working set in Quality Lab.",
  },
  alpha_blend: {
    title: "Aligned Crop Alpha-Blend Slider (A ↔ B)",
    meaning: "Linearly blends aligned 112×112 Face A (`0%`) and Face B (`100%`) in canonical template space.",
    effect: "At `50%`, eyes, nose, and mouth overlap directly, making geometric pose residuals vs. genuine structural differences visually obvious.",
    limitation: "Visual inspection aid only — does not alter extracted embeddings.",
  },
  overlay_opacity: {
    title: "Annotation Overlay Opacity (0% – 100%)",
    meaning: "Controls continuous alpha transparency of rendered canvas annotation overlays (face boxes, confidence tags, 5-point landmarks, correspondences, and ground-truth boxes).",
    effect: "`0%` hides all overlays to reveal unobstructed native pixels on tiny `<32 px` faces; `50%` renders translucent annotations; `100%` reproduces full WB2 opacity.",
    limitation: "Affects ONLY the annotation layer — underlying source image pixels, coordinates, zoom/pan, hit-testing, and crop alpha-blending remain 100% unchanged.",
  },

  // ---------------- RETRIEVAL ----------------
  nlist: {
    title: "IVF Coarse Voronoi Partitions (nlist)",
    meaning: "Number of spherical K-Means coarse centroids partitioning the gallery embedding space in `IVF-Flat` and `IVF-PQ`.",
    effect: "Higher `nlist` creates smaller inverted lists (faster search per probed cell) but increases coarse quantizer training time.",
    limitation: "Automatically clamped to `min(nlist, gallery_size)` on small galleries.",
  },
  nprobe: {
    title: "IVF Probed Voronoi Cells (nprobe)",
    meaning: "Number of nearest coarse Voronoi partitions scanned per query during `IVF-Flat` or `IVF-PQ` search.",
    effect: "Increasing `nprobe` (e.g., `1 → 16`) raises `Recall@K` toward ExactFlat `1.000` while linearly increasing candidate scan latency.",
    limitation: "Setting `nprobe = nlist` degrades IVF into an exhaustive scan.",
  },
  m_subspaces: {
    title: "PQ Subvector Partitions (M / m_subspaces)",
    meaning: "Splits each 512-D residual vector into `M` orthogonal subvectors of dimension `512 / M` for Product Quantization.",
    effect: "`M = 32` (16-D subspaces, 8-bit codes) compresses each 2,048-byte FP32 vector into 32 bytes (`64×` code compression) with high ADC recall.",
    limitation: "Must evenly divide embedding dimension `D` (e.g., 8, 16, 32, 64).",
  },
  nbits: {
    title: "PQ Bits per Subquantizer (nbits)",
    meaning: "Codebook bit-width per subspace (`2^nbits` centroids per subspace; `8` bits = 256 centroids).",
    effect: "Higher `nbits` lowers quantization distortion and boosts `Recall@1` at higher codebook training cost.",
    limitation: "Valid range: [4 .. 8] bits.",
  },
  m: {
    title: "HNSW Bi-Directional Graph Degree (M)",
    meaning: "Maximum outgoing neighbor links per node per layer in the Hierarchical Navigable Small World graph.",
    effect: "Higher `M` (e.g., 16–32) improves graph navigability and recall on high-dimensional 512-D embeddings while increasing index RAM.",
    limitation: "Pure-Python HNSW implementation is capped at `N <= 15,000` vectors on laptop hardware.",
  },
  ef_construction: {
    title: "HNSW Build Candidate Queue Width (ef_construction)",
    meaning: "Dynamic candidate list size maintained when linking each new vector during HNSW graph construction.",
    effect: "Higher values build higher-quality neighbor graphs (better recall) at slower index build time.",
    limitation: "Affects index construction only, not query-time search speed.",
  },
  ef_search: {
    title: "HNSW Query Beam Width (ef_search)",
    meaning: "Dynamic priority-queue width explored during greedy multi-layer HNSW query traversal.",
    effect: "Increasing `ef_search` (`16 → 96+`) monotonically improves `Recall@K` at higher per-query latency.",
    limitation: "Should be `>= top_k`.",
  },
  product_quantization: {
    title: "Residual Product Quantization with Asymmetric Distance Computation (IVF-PQ ADC)",
    meaning: "Quantizes coarse-centroid residuals into `M` compact byte codes and scores queries via precomputed uncompressed query-to-subcentroid lookup tables.",
    effect: "Achieves `64×` vector code compression (`~4.8 MiB` vs `195 MiB` at `N=100K`) while preserving ~95–99% `Recall@5`.",
    limitation: "ADC scores approximate inner product and exhibit small quantization bias relative to ExactFlat cosine.",
  },
  recall_at_k: {
    title: "Nearest-Neighbor Recall@K (ANN vs. ExactFlat)",
    meaning: "Fraction of queries for which the true ExactFlat Top-1 nearest neighbor appears within the ANN index's returned Top-K list.",
    effect: "`Recall@1 = 1.00` means 100% agreement with exhaustive exact search.",
    limitation: "Measures ANN index fidelity against ExactFlat — NOT closed-set biometric subject identification accuracy when synthetic distractors are present.",
  },

  // ---------------- BIOMETRIC EVALUATION ----------------
  tar: {
    title: "True Accept Rate (TAR = 1 − FRR)",
    meaning: "Proportion of genuine same-subject pairs correctly scoring at or above decision threshold τ (`TAR = TP / (TP + FN)`).",
    effect: "Reported at fixed calibrated False Accept Rates (e.g., `TAR @ FAR = 10⁻³`) to compare models fairly.",
    limitation: "Must always be paired with its corresponding FAR and dataset split.",
  },
  far: {
    title: "False Accept Rate (FAR / FMR)",
    meaning: "Proportion of impostor (different-subject) pairs erroneously scoring `>= τ` (`FAR = FP / (FP + TN)`), also termed False Match Rate (FMR).",
    effect: "Raising threshold `τ` reduces FAR; operational security targets typically require `FAR <= 10⁻³` or `10⁻⁴`.",
    limitation: "Minimum resolvable non-zero FAR is `1 / N_impostor` (`1/111` on NASA smoke test vs. `1/1,500` on LFW Test).",
  },
  frr: {
    title: "False Reject Rate (FRR / FNMR = 1 − TAR)",
    meaning: "Proportion of genuine same-subject pairs erroneously scoring `< τ` (`FRR = FN / (TP + FN)`), equivalent to False Non-Match Rate (FNMR) when all faces are enrolled.",
    effect: "Plotted against FAR on DET curves; raising `τ` increases FRR.",
    limitation: "Depends strongly on pose, blur, illumination, and detector crop alignment.",
  },
  fnmr: {
    title: "False Non-Match Rate (FNMR)",
    meaning: "ISO/IEC 19795 biometric matcher error rate for genuine comparisons scoring below threshold τ (`FNMR = 1 − TAR` on extracted pairs).",
    effect: "Lower FNMR at fixed FMR indicates superior genuine/impostor score separation.",
    limitation: "Distinct from Failure-to-Acquire (FTA) when a detector misses a face entirely.",
  },
  eer: {
    title: "Equal Error Rate (EER %)",
    meaning: "Operating point on the ROC/DET curve where False Accept Rate equals False Reject Rate (`FAR(τ*) = FRR(τ*)`).",
    effect: "Lower EER indicates better overall score separability (`0.27%` EER for AdaFace FP16 on LFW Held-Out Folds 6–10).",
    limitation: "Summary statistic at `FAR = FRR`; high-security deployments operate at much lower FAR (`10⁻³`–`10⁻⁶`) than EER.",
  },
  roc_auc: {
    title: "Area Under the ROC Curve (ROC AUC)",
    meaning: "Threshold-independent probability that a randomly chosen genuine pair outscores a randomly chosen impostor pair (`1.0` = perfect separation; `0.5` = random).",
    effect: "Summarizes global ranking separation across all thresholds.",
    limitation: "Saturates near `1.000` on easy splits (`NASA 120-pair` smoke test); inspect `TAR @ FAR=1e-3` and `DET` curves for fine discrimination.",
  },
  calibration_threshold: {
    title: "Identity-Disjoint Calibration Threshold (τ_calib)",
    meaning: "Decision threshold estimated exclusively on Calibration Folds 1–5 (`2,086` subjects) and evaluated once without tuning on Held-Out Test Folds 6–10 (`2,195` subjects).",
    effect: "Prevents optimistic test-set threshold leakage (`P3` & `P3.1` methodology).",
    limitation: "Realized test FAR on finite samples can deviate slightly from target calibration FAR.",
  },
  identity_disjoint_split: {
    title: "Identity-Disjoint Evaluation Protocol",
    meaning: "Guarantees zero person-identity overlap (`0` shared subjects) between threshold calibration (`Folds 1–5`: 2,086 subjects) and held-out evaluation (`Folds 6–10`: 2,195 subjects).",
    effect: "Ensures reported verification metrics generalize to unseen identities rather than memorizing subject templates.",
    limitation: "LFW pairs are restricted strictly to non-commercial academic/algorithmic research.",
  },
};

export function helpBadgeHtml(glossaryKey, labelOverride = null) {
  const entry = TECHNICAL_GLOSSARY[glossaryKey];
  if (!entry) return "";
  const aria = labelOverride || `Explain ${entry.title}`;
  return `<button type="button" class="wb-help-trigger" data-tip-key="${esc(glossaryKey)}" aria-label="${esc(
    aria
  )}">?</button>`;
}

export function tipAttr(glossaryKey) {
  if (!TECHNICAL_GLOSSARY[glossaryKey]) return "";
  return `data-tip-key="${esc(glossaryKey)}"`;
}

let portalEl = null;
let hoverTimer = null;
let activeTarget = null;
let pinnedByClick = false;

function ensurePortal() {
  if (!portalEl) {
    portalEl = document.getElementById("wb-tooltip-portal");
    if (!portalEl) {
      portalEl = document.createElement("div");
      portalEl.id = "wb-tooltip-portal";
      portalEl.className = "wb-tooltip-portal hidden";
      portalEl.setAttribute("role", "tooltip");
      document.body.appendChild(portalEl);
    }
  }
  return portalEl;
}

function resolveTipData(el) {
  if (!el) return null;
  const key = el.getAttribute("data-tip-key");
  if (key && TECHNICAL_GLOSSARY[key]) {
    return TECHNICAL_GLOSSARY[key];
  }
  const customTitle = el.getAttribute("data-tip-title");
  const customBody = el.getAttribute("data-tip-body");
  if (customTitle || customBody) {
    return {
      title: customTitle || "Parameter / Control Help",
      meaning: customBody || "",
      effect: el.getAttribute("data-tip-effect") || "",
      limitation: el.getAttribute("data-tip-note") || "",
    };
  }
  return null;
}

export function showTooltipForElement(targetEl, pin = false) {
  const data = resolveTipData(targetEl);
  if (!data) return;
  const portal = ensurePortal();
  activeTarget = targetEl;
  pinnedByClick = pin;

  portal.innerHTML = `
    <div class="wb-tip-header">${esc(data.title)}</div>
    ${data.meaning ? `<div class="wb-tip-row">${esc(data.meaning)}</div>` : ""}
    ${data.effect ? `<div class="wb-tip-row"><strong>Effect:</strong> ${esc(data.effect)}</div>` : ""}
    ${data.limitation ? `<div class="wb-tip-note"><strong>Note:</strong> ${esc(data.limitation)}</div>` : ""}
  `;
  portal.classList.remove("hidden");
  targetEl.setAttribute("aria-describedby", "wb-tooltip-portal");

  // Viewport-clamped positioning (never clipped by panel boundaries or window edges)
  const rect = targetEl.getBoundingClientRect();
  const pw = Math.min(360, window.innerWidth - 16);
  portal.style.maxWidth = `${pw}px`;
  const pRect = portal.getBoundingClientRect();

  let left = rect.left + rect.width / 2 - pRect.width / 2;
  left = Math.max(8, Math.min(left, window.innerWidth - pRect.width - 8));

  let top = rect.bottom + 8;
  if (top + pRect.height > window.innerHeight - 8) {
    top = Math.max(8, rect.top - pRect.height - 8);
  }

  portal.style.left = `${Math.round(left)}px`;
  portal.style.top = `${Math.round(top)}px`;
}

export function hideTooltip(force = false) {
  if (hoverTimer) {
    clearTimeout(hoverTimer);
    hoverTimer = null;
  }
  if (pinnedByClick && !force) return;
  pinnedByClick = false;
  if (activeTarget) {
    activeTarget.removeAttribute("aria-describedby");
    activeTarget = null;
  }
  if (portalEl) {
    portalEl.classList.add("hidden");
  }
}

let initialized = false;
export function initTooltipManager() {
  if (initialized) return;
  initialized = true;
  ensurePortal();

  const findTrigger = (t) => (t && t.closest ? t.closest("[data-tip-key], [data-tip-title], [data-tip-body]") : null);

  document.addEventListener("pointerover", (e) => {
    const trig = findTrigger(e.target);
    if (!trig) return;
    if (pinnedByClick && activeTarget !== trig) return;
    if (hoverTimer) clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => showTooltipForElement(trig, false), 110);
  });

  document.addEventListener("pointerout", (e) => {
    const trig = findTrigger(e.target);
    if (!trig) return;
    const related = e.relatedTarget;
    if (related && trig.contains(related)) return;
    hideTooltip(false);
  });

  document.addEventListener("focusin", (e) => {
    const trig = findTrigger(e.target);
    if (trig) showTooltipForElement(trig, false);
  });

  document.addEventListener("focusout", () => {
    hideTooltip(false);
  });

  document.addEventListener("click", (e) => {
    const helpBtn = e.target && e.target.closest ? e.target.closest(".wb-help-trigger") : null;
    if (helpBtn) {
      e.preventDefault();
      e.stopPropagation();
      if (activeTarget === helpBtn && pinnedByClick) {
        hideTooltip(true);
      } else {
        showTooltipForElement(helpBtn, true);
      }
      return;
    }
    if (pinnedByClick && portalEl && !portalEl.contains(e.target)) {
      hideTooltip(true);
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hideTooltip(true);
    }
  });

  window.addEventListener("scroll", () => hideTooltip(true), true);
}

/**
 * Prevents duplicate concurrent submissions when a researcher clicks an action button repeatedly.
 */
export async function withBusyButton(btn, fn, busyLabel = "Running…") {
  if (!btn) return fn();
  if (btn.dataset.wbBusy === "1" || btn.disabled) return undefined;
  const origHtml = btn.innerHTML;
  const origTitle = btn.getAttribute("title");
  btn.dataset.wbBusy = "1";
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  btn.setAttribute("title", "Execution in progress — please wait for completion.");
  btn.textContent = busyLabel;
  try {
    return await fn();
  } finally {
    delete btn.dataset.wbBusy;
    if (btn.isConnected) {
      btn.disabled = false;
      btn.removeAttribute("aria-busy");
      if (origTitle) btn.setAttribute("title", origTitle);
      else btn.removeAttribute("title");
      btn.innerHTML = origHtml;
    }
  }
}

/**
 * Clamps and validates a numeric input element, returning `{ value, clamped, invalid }`.
 */
export function sanitizeNumericInput(rawStr, { min = -Infinity, max = Infinity, isInt = false, fallback = 0 } = {}) {
  const trimmed = String(rawStr ?? "").trim();
  if (trimmed === "") {
    return { value: fallback, clamped: false, invalid: true, reason: "Empty value restored to valid setting." };
  }
  const num = Number(trimmed);
  if (!Number.isFinite(num)) {
    return { value: fallback, clamped: false, invalid: true, reason: "Non-numeric input rejected." };
  }
  let v = num;
  let roundedInt = false;
  if (isInt && !Number.isInteger(v)) {
    v = Math.round(v);
    roundedInt = true;
  }
  let clamped = false;
  if (v < min) {
    v = min;
    clamped = true;
  } else if (v > max) {
    v = max;
    clamped = true;
  }
  return {
    value: v,
    clamped: clamped || roundedInt,
    invalid: false,
    reason: clamped
      ? `Clamped to permitted range [${min} .. ${max}].`
      : roundedInt
      ? `Rounded decimal input to integer (${v}).`
      : null,
  };
}
