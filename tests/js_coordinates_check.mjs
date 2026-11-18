/**
 * Automated JavaScript Syntax & Visualization Coordinate-Space Verification (`tests/js_coordinates_check.mjs`).
 *
 * Verifies:
 *   1. Every ES2022 module under `frontend/js/` parses without syntax errors (`node --check`).
 *   2. Original-pixel <-> viewport-screen coordinate roundtrips (`origToScreen` / `screenToOrig`),
 *      fit-to-screen aspect preservation, 1:1 native transform, cursor-anchored zoom invariance,
 *      bounding-box / landmark scaling, and crop-local vs image-global separation hold to < 1e-9 px.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  computeFitTransform,
  computeNative1to1Transform,
  hitTestFaceBoxes,
  origToScreen,
  projectBoxToScreen,
  projectPointsToScreen,
  screenToOrig,
  zoomAroundScreenPoint,
} from "../frontend/js/utils/coordinates.js";
import {
  areParamValuesEqual,
  compareTwoExperimentConfigs,
  computeCrossStageDelta,
  computeSingleStageDelta,
  isParamVisibleForValues,
} from "../frontend/js/components/config_delta.js";
import {
  classifyPairsAtThreshold,
  computeEmpiricalOperatingPoint,
  computeEmpiricalRocCurve,
  computeScoreHistograms,
} from "../frontend/js/components/threshold_explorer.js";
import { computeOverlayAlphaComposite } from "../frontend/js/components/image_viewer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const jsRoot = path.join(rootDir, "frontend", "js");

function collectJsFiles(dir) {
  const out = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...collectJsFiles(full));
    else if (ent.name.endsWith(".js")) out.push(full);
  }
  return out;
}

// 1. Syntax check every frontend ES module
const jsFiles = collectJsFiles(jsRoot);
assert(jsFiles.length >= 16, `Expected >= 16 frontend JS modules; found ${jsFiles.length}`);
for (const file of jsFiles) {
  execFileSync(process.execPath, ["--check", file], { stdio: "pipe" });
}

// 2. Fit-to-screen aspect ratio & roundtrip on non-square image (1920x1271 into 800x440 viewport)
const fitT = computeFitTransform(1920, 1271, 800, 440, 16);
assert(fitT.scale > 0 && fitT.scale < 1.0);
const testLandmarks = [
  [312.45, 408.12],
  [389.9, 405.75],
  [351.02, 455.33],
  [322.1, 502.8],
  [381.4, 500.2],
];
for (const [ox, oy] of testLandmarks) {
  const scr = origToScreen(ox, oy, fitT);
  const back = screenToOrig(scr.x, scr.y, fitT);
  assert(Math.abs(back.x - ox) < 1e-9, `X roundtrip drift: ${back.x} vs ${ox}`);
  assert(Math.abs(back.y - oy) < 1e-9, `Y roundtrip drift: ${back.y} vs ${oy}`);
  assert.equal(back.inBounds, true);
}

// 3. Cursor-anchored zoom must keep the anchored original pixel fixed at the exact screen point!
const anchorScreen = origToScreen(635.0, 235.0, fitT);
const zoomedT = zoomAroundScreenPoint(fitT, 3.2, anchorScreen.x, anchorScreen.y);
const afterZoomOrig = screenToOrig(anchorScreen.x, anchorScreen.y, zoomedT);
assert(Math.abs(afterZoomOrig.x - 635.0) < 1e-9);
assert(Math.abs(afterZoomOrig.y - 235.0) < 1e-9);

// 4. Pan + zoom box & landmark scaling verification
const pannedT = { ...zoomedT, offsetX: zoomedT.offsetX - 75.5, offsetY: zoomedT.offsetY + 42.25 };
const origBox = { x: 240.0, y: 165.0, w: 86.0, h: 106.0 };
const scrBox = projectBoxToScreen(origBox, pannedT);
assert(Math.abs(scrBox.w - origBox.w * pannedT.scale) < 1e-9);
assert(Math.abs(scrBox.h - origBox.h * pannedT.scale) < 1e-9);
const tlBack = screenToOrig(scrBox.x, scrBox.y, pannedT);
assert(Math.abs(tlBack.x - origBox.x) < 1e-9);
assert(Math.abs(tlBack.y - origBox.y) < 1e-9);

// 5. Native 1:1 resolution transform preserves exact 1.0 pixel step
const natT = computeNative1to1Transform(1280, 1024, 900, 500);
assert.equal(natT.scale, 1.0);
const p0 = origToScreen(100, 200, natT);
const p1 = origToScreen(101, 201, natT);
assert(Math.abs(p1.x - p0.x - 1.0) < 1e-12);
assert(Math.abs(p1.y - p0.y - 1.0) < 1e-12);

// 6. Crop-local [0..112] vs original-image coordinate separation + hit-testing
const cropT = computeFitTransform(112, 112, 370, 370, 16);
const arcfacePts = projectPointsToScreen(
  [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
  ],
  cropT
);
assert(arcfacePts[1][0] > arcfacePts[0][0]);

const hitIdx = hitTestFaceBoxes(600, 200, [
  { bounding_box: { x: 100, y: 100, w: 50, h: 50 } },
  { bounding_box: { x: 594, y: 166, w: 84, h: 106 } },
]);
assert.equal(hitIdx, 1);

// =========================================================================
// 7. WB3 P-1: Global Cross-Stage Configuration Delta & Two-Run Comparison
// =========================================================================
const registryFile = path.join(rootDir, "configs", "workbench_registry_schema.json");
const fallbackAlgos = [
  {
    algorithm_id: "yunet_2023mar",
    name: "YuNet (OpenCV Zoo 2023-Mar)",
    parameters: [
      { name: "score_threshold", label: "Score Threshold", value_type: "float", default: 0.6 },
      { name: "nms_mode", label: "NMS Mode", value_type: "string", default: "hard_greedy" },
      { name: "soft_nms_sigma", label: "Soft-NMS Sigma", value_type: "float", default: 0.5, visible_when: { nms_mode: "gaussian_soft_nms" } },
      { name: "top_k", label: "Top-K", value_type: "int", default: 5000 },
    ],
  },
  {
    algorithm_id: "scrfd_500m_research",
    name: "SCRFD-500M",
    parameters: [
      { name: "score_threshold", label: "Score Threshold", value_type: "float", default: 0.5 },
      { name: "max_dim", label: "Max Input Dim", value_type: "int", default: 1280 },
    ],
  },
  {
    algorithm_id: "five_point_canonical_aligner",
    name: "5-Point Canonical Aligner",
    parameters: [
      { name: "transform_mode", label: "Transform Mode", value_type: "string", default: "similarity_umeyama_4dof" },
      { name: "interpolation", label: "Interpolation", value_type: "string", default: "linear" },
    ],
  },
  {
    algorithm_id: "hybrid_classical_and_norm_quality",
    name: "Hybrid Classical + Norm Quality",
    parameters: [
      { name: "reject_quantile", label: "Reject Quantile", value_type: "float", default: 0.0 },
    ],
  },
  {
    algorithm_id: "adaface_ir50_webface4m",
    name: "AdaFace IR-50",
    parameters: [
      { name: "precision", label: "Precision", value_type: "string", default: "fp32" },
      { name: "batch_size", label: "Batch Size", value_type: "int", default: 8 },
      { name: "flip_tta", label: "Flip TTA", value_type: "bool", default: false },
      { name: "locked_preprocessing_summary", label: "Locked Preprocessing", value_type: "string", default: "RGB [-1,1]", read_only: true },
    ],
  },
  {
    algorithm_id: "ivf_pq_index",
    name: "IVF-PQ Index",
    parameters: [
      { name: "nprobe", label: "nprobe", value_type: "int", default: 16 },
    ],
  },
  {
    algorithm_id: "exact_flat_index",
    name: "Exact Flat Index",
    parameters: [
      { name: "backend", label: "Backend", value_type: "string", default: "numpy_cpu" },
    ],
  },
];
const regPayload = fs.existsSync(registryFile)
  ? JSON.parse(fs.readFileSync(registryFile, "utf-8"))
  : { algorithms: fallbackAlgos };
const algosById = Object.fromEntries(regPayload.algorithms.map((a) => [a.algorithm_id, a]));

// 7A. Default configuration -> 0 active modifications across all 5 stages
const defaultExpState = {
  detectorId: "yunet_2023mar",
  detectorParamsByModel: { yunet_2023mar: {}, scrfd_500m_research: {} },
  alignmentParams: {},
  qualityParams: {},
  recognizerId: "adaface_ir50_webface4m",
  recognizerParamsByModel: { adaface_ir50_webface4m: {}, mobilefacenet_w600k_research: {} },
  retrievalIndexId: "ivf_pq_index",
  retrievalParamsByIndex: { ivf_pq_index: {} },
};
const cleanDelta = computeCrossStageDelta(algosById, defaultExpState, { scope: "global" });
assert.equal(cleanDelta.totalModified, 0);
assert.equal(cleanDelta.stages.length, 5);
// Locked parameter (`locked_preprocessing_summary`) must have status === "Locked", never "Modified"
const lockedRows = cleanDelta.allRows.filter((r) => r.status === "Locked");
assert(lockedRows.length >= 1, "Expected locked preprocessing row in Recognition stage");

// 7B. Modify 1 parameter in each of the 5 stages -> totalModified === 5, stageCounts = 1 each
const fiveStageModState = {
  detectorId: "yunet_2023mar",
  detectorParamsByModel: { yunet_2023mar: { score_threshold: 0.78 } },
  alignmentParams: { transform_mode: "affine_least_squares_6dof" },
  qualityParams: { reject_quantile: 0.15 },
  recognizerId: "adaface_ir50_webface4m",
  recognizerParamsByModel: { adaface_ir50_webface4m: { flip_tta: true } },
  retrievalIndexId: "ivf_pq_index",
  retrievalParamsByIndex: { ivf_pq_index: { nprobe: 32 } },
};
const mod5Delta = computeCrossStageDelta(algosById, fiveStageModState, { scope: "global" });
assert.equal(mod5Delta.totalModified, 5);
for (const sk of ["detection", "alignment", "quality", "recognition", "retrieval"]) {
  assert.equal(mod5Delta.stageCounts[sk], 1, `Expected 1 modification in stage ${sk}`);
}

// 7C. Conditional parameter visibility (`soft_nms_sigma` hidden under `hard_greedy` must NOT count!)
const condHidden = computeSingleStageDelta({
  stageKey: "detection",
  stageLabel: "Detection",
  algoSpec: algosById["yunet_2023mar"],
  currentValues: { nms_mode: "hard_greedy", soft_nms_sigma: 0.25 },
});
assert.equal(condHidden.modifiedCount, 0, "Hidden soft_nms_sigma must not count as active modification");
const condVisible = computeSingleStageDelta({
  stageKey: "detection",
  stageLabel: "Detection",
  algoSpec: algosById["yunet_2023mar"],
  currentValues: { nms_mode: "gaussian_soft_nms", soft_nms_sigma: 0.25 },
});
assert.equal(condVisible.modifiedCount, 2, "nms_mode + visible soft_nms_sigma must both count when Soft-NMS active");

// 7D. Disabled stage in Pipeline scope -> status === "Not Applicable", modifiedCount === 0
const pipeDisabledDelta = computeCrossStageDelta(
  algosById,
  {
    pipelineA: {
      detection: { algorithm_id: "scrfd_500m_research", params: { max_dim: 640 } },
      alignment: { algorithm_id: "five_point_canonical_aligner", params: {} },
      quality: { enabled: false, params: { reject_quantile: 0.2 } },
      recognition: { enabled: true, algorithm_id: "adaface_ir50_webface4m", params: {} },
      normalization: { enabled: true },
      retrieval: { enabled: false, algorithm_id: "exact_flat_index", params: {} },
    },
  },
  { scope: "pipeline" }
);
assert.equal(pipeDisabledDelta.stageCounts.detection, 1);
assert.equal(pipeDisabledDelta.stageCounts.quality, 0);
assert.equal(pipeDisabledDelta.stages.find((s) => s.stageKey === "quality").rows[0].status, "Not Applicable");

// 7E. Two-Run Configuration Comparison with stage/model context & dataset mismatch warning
const cmpTwo = compareTwoExperimentConfigs(
  {
    experiment_id: "EXP-WB-A",
    category: "pipeline",
    dataset_id: "lfw_research_split",
    dataset_split: "Folds 6-10",
    parameters: {
      detection: { algorithm_id: "yunet_2023mar", params: { score_threshold: 0.6 } },
      recognition: { algorithm_id: "adaface_ir50_webface4m", params: { precision: "fp16", batch_size: 8 } },
    },
  },
  {
    experiment_id: "EXP-WB-B",
    category: "pipeline",
    dataset_id: "nasa_p11_verification",
    dataset_split: "120 smoke",
    parameters: {
      detection: { algorithm_id: "scrfd_500m_research", params: { score_threshold: 0.5 } },
      recognition: { algorithm_id: "adaface_ir50_webface4m", params: { precision: "fp32", batch_size: 8 } },
    },
  },
  algosById
);
assert.equal(cmpTwo.datasetMismatch, true);
assert.equal(cmpTwo.modelFamilyMismatch, true);
assert(cmpTwo.warnings.length >= 2);
assert(cmpTwo.counts.changed >= 3);
assert(cmpTwo.counts.added >= 1); // SCRFD-only params added in Run B
assert(cmpTwo.counts.absent >= 1); // YuNet-only params absent in Run B

// =========================================================================
// 8. WB3 P-2: Score Distribution, Boundary/Tie Math & Zero-Denominator Safety
// =========================================================================
const genFixture = [0.20, 0.35, 0.50, 0.72, 0.88];
const impFixture = [-0.10, 0.05, 0.20, 0.35, 0.60];

// Tie test (`tau = 0.35` -> genuine >= 0.35 gives 4/5 = 0.80; impostor >= 0.35 gives 2/5 = 0.40)
const opTie = computeEmpiricalOperatingPoint(genFixture, impFixture, 0.35, true);
assert.equal(opTie.tp, 4);
assert.equal(opTie.fn, 1);
assert.equal(opTie.fp, 2);
assert.equal(opTie.tn, 3);
assert.equal(opTie.tar, 0.8);
assert.equal(opTie.frr, 0.2);
assert.equal(opTie.far, 0.4);
assert.equal(opTie.tnr, 0.6);

// All accepted (`tau = -1.0`) & All rejected (`tau = 1.0`)
const opAllAcc = computeEmpiricalOperatingPoint(genFixture, impFixture, -1.0, true);
assert.equal(opAllAcc.tar, 1.0);
assert.equal(opAllAcc.far, 1.0);
assert.equal(opAllAcc.frr, 0.0);
assert.equal(opAllAcc.tnr, 0.0);

const opAllRej = computeEmpiricalOperatingPoint(genFixture, impFixture, 1.0, true);
assert.equal(opAllRej.tar, 0.0);
assert.equal(opAllRej.far, 0.0);
assert.equal(opAllRej.frr, 1.0);
assert.equal(opAllRej.tnr, 1.0);

// Zero-denominator handling (never NaN or Infinity)
const opNoGen = computeEmpiricalOperatingPoint([], impFixture, 0.3, true);
assert.equal(opNoGen.tar, null);
assert.equal(opNoGen.frr, null);
assert.equal(opNoGen.far, 0.4);
const opNoImp = computeEmpiricalOperatingPoint(genFixture, [], 0.3, true);
assert.equal(opNoImp.far, null);
assert.equal(opNoImp.tnr, null);
assert.equal(opNoImp.tar, 0.8);

// Histogram population conservation
const histCheck = computeScoreHistograms(genFixture, impFixture, { binCount: 12 });
assert.equal(histCheck.genuineSum, genFixture.length);
assert.equal(histCheck.impostorSum, impFixture.length);

// Pair error classification & ROC monotonicity
const pairRefs = [
  { pair_id: "P1", is_same_subject: true, image_a: "a1.jpg", image_b: "b1.jpg" },
  { pair_id: "P2", is_same_subject: true, image_a: "a2.jpg", image_b: "b2.jpg" },
  { pair_id: "P3", is_same_subject: false, image_a: "a3.jpg", image_b: "b3.jpg" },
  { pair_id: "P4", is_same_subject: false, image_a: "a4.jpg", image_b: "b4.jpg" },
];
const clsCheck = classifyPairsAtThreshold(pairRefs, [0.82, 0.18, 0.44, 0.02], 0.30, true);
assert.equal(clsCheck.trueAccepts.length, 1);
assert.equal(clsCheck.falseRejects.length, 1);
assert.equal(clsCheck.falseAccepts.length, 1);
assert.equal(clsCheck.trueRejects.length, 1);
assert.equal(clsCheck.falseAccepts[0].pair_id, "P3");
assert.equal(clsCheck.falseRejects[0].pair_id, "P2");

const rocCheck = computeEmpiricalRocCurve(genFixture, impFixture, true);
assert(rocCheck.points.length >= 6);
for (let i = 1; i < rocCheck.points.length; i++) {
  assert(rocCheck.points[i].far >= rocCheck.points[i - 1].far);
}

// =========================================================================
// 9. WB3 P-3: Continuous Overlay Opacity Alpha Compositing (0, 25, 50, 75, 100%)
// =========================================================================
const basePixel = [40, 120, 200];
const overlayPixel = [220, 20, 60];
assert.deepEqual(computeOverlayAlphaComposite(basePixel, overlayPixel, 0), [40, 120, 200]);
assert.deepEqual(computeOverlayAlphaComposite(basePixel, overlayPixel, 25), [85, 95, 165]);
assert.deepEqual(computeOverlayAlphaComposite(basePixel, overlayPixel, 50), [130, 70, 130]);
assert.deepEqual(computeOverlayAlphaComposite(basePixel, overlayPixel, 75), [175, 45, 95]);
assert.deepEqual(computeOverlayAlphaComposite(basePixel, overlayPixel, 100), [220, 20, 60]);

console.log(`JS_COORDINATE_AND_SYNTAX_CHECKS_PASSED (${jsFiles.length} modules verified)`);
