/**
 * Persistent Reactive State Store (`frontend/js/state.js`).
 *
 * Strictly separates:
 *   - `ui`: Active workspace, theme, overlay toggles, comparison mode
 *   - `exp`: Algorithm selections, parameter maps, selected images/pairs, and cached results
 * Navigating across any of the 12 workspaces preserves the complete experiment state.
 */

const DEFAULT_PRIMARY_IMAGE = "datasets/evaluation/detection/det_group_sts120_crew.jpg";
const DEFAULT_IMAGE_A = "datasets/evaluation/verification_pairs/ver_armstrong_official.jpg";
const DEFAULT_IMAGE_B = "datasets/evaluation/verification_pairs/ver_armstrong_pose.jpg";

export const store = {
  registry: null,
  algorithmsById: {},
  datasets: [],
  experiments: [],
  systemStatus: null,

  ui: {
    activeWorkspace: "dashboard",
    theme: "light",
    overlayOpacityPct: 100,
    configDeltaDrawerOpen: false,
    overlays: {
      bboxes: true,
      gtBoxes: true,
      landmarks: true,
      landmarkLabels: false,
      correspondenceLines: true,
      confidenceLabels: true,
      qualityBadges: true,
      alignmentBoundary: true,
    },
  },

  exp: {
    // Explorer & Single-Image Labs
    explorerImages: [
      DEFAULT_PRIMARY_IMAGE,
      "datasets/evaluation/detection/det_group_apollo12_crew.jpg",
      "datasets/samples/dev_single_sally_ride.jpg",
      DEFAULT_IMAGE_A,
      DEFAULT_IMAGE_B,
      "datasets/evaluation/verification_pairs/ver_aldrin_portrait.jpg",
    ],
    selectedImageRef: DEFAULT_PRIMARY_IMAGE,
    selectedSecondaryImageRef: "datasets/evaluation/detection/det_group_apollo12_crew.jpg",
    selectedFaceIndex: 0,

    // Detection Lab
    detectorId: "yunet_2023mar",
    detectorParamsByModel: {},
    lastDetectionResult: null,
    lastDetectionSweep: null,

    // Alignment Lab
    landmarkSource: "yunet_2023mar",
    compareBothDetectors: true,
    alignmentParams: {},
    lastAlignmentResult: null,

    // Quality Lab
    qualityParams: {},
    qualityDegradation: { blur_sigma: 0.0, effective_res: 112, gamma: 1.0, jpeg_quality: 100 },
    lastQualityResult: null,

    // Recognition & 1:1 / 1:N Lab
    imageARef: DEFAULT_IMAGE_A,
    imageBRef: DEFAULT_IMAGE_B,
    faceIndexA: 0,
    faceIndexB: 0,
    recognizerId: "adaface_ir50_webface4m",
    recognizerParamsByModel: {},
    matchingMetric: "cosine",
    threshold: 0.2169,
    lastVerificationResult: null,
    lastIdentificationResult: null,

    // 1:N Retrieval Lab
    retrievalIndexId: "ivf_pq_index",
    retrievalParamsByIndex: {},
    retrievalGallerySize: 10000,
    retrievalQueryCount: 150,
    lastRetrievalBenchmark: null,

    // Pipeline Experiment Builder (Pipeline A & Comparison Pipeline B)
    pipelineA: {
      name: "Pipeline A — YuNet 1280 + AdaFace IR-50 FP16",
      dataset_id: "nasa_p11_multiscale",
      pair_limit: 120,
      detection: { algorithm_id: "yunet_2023mar", enable_landmarks: true, params: {} },
      alignment: { algorithm_id: "five_point_canonical_aligner", params: {} },
      quality: { enabled: true, params: {} },
      recognition: { enabled: true, algorithm_id: "adaface_ir50_webface4m", strict_weights: true, params: {} },
      normalization: { enabled: true },
      matching: { metric: "cosine", threshold: 0.2169 },
      retrieval: { enabled: false, algorithm_id: "exact_flat_index", params: {} },
    },
    savedPipelines: [],
    lastPipelineResult: null,

    // Comparison Workspace
    selectedExperimentIds: [
      "HIST-P3-EXP02-ADAFACE_IR50_FP16-YUNET_5PT_LANDMARKS",
      "HIST-P3-EXP02-ADAFACE_IR50_FP16-SCRFD_5PT_LANDMARKS",
      "HIST-P3-EXP06-ADAFACE_IR50_FP16-FLIP_TTA_CALIB_TEST",
      "HIST-P3-EXP02-MOBILEFACENET_W600K_FP32-YUNET_5PT_LANDMARKS",
    ],
  },

  listeners: new Set(),

  initFromRegistry(registryPayload) {
    this.registry = registryPayload;
    this.algorithmsById = {};
    for (const algo of registryPayload.algorithms || []) {
      this.algorithmsById[algo.algorithm_id] = algo;
      const defaults = {};
      for (const p of algo.parameters || []) {
        defaults[p.name] = p.default;
      }
      if (algo.category === "detection") {
        this.exp.detectorParamsByModel[algo.algorithm_id] = {
          ...defaults,
          ...(this.exp.detectorParamsByModel[algo.algorithm_id] || {}),
        };
      } else if (algo.category === "landmark_alignment") {
        this.exp.alignmentParams = { ...defaults, ...this.exp.alignmentParams };
      } else if (algo.category === "quality") {
        this.exp.qualityParams = { ...defaults, ...this.exp.qualityParams };
      } else if (algo.category === "recognition") {
        this.exp.recognizerParamsByModel[algo.algorithm_id] = {
          ...defaults,
          ...(this.exp.recognizerParamsByModel[algo.algorithm_id] || {}),
        };
      } else if (algo.category === "retrieval") {
        this.exp.retrievalParamsByIndex[algo.algorithm_id] = {
          ...defaults,
          ...(this.exp.retrievalParamsByIndex[algo.algorithm_id] || {}),
        };
      }
    }
    this.exp.pipelineA.detection.params = { ...(this.exp.detectorParamsByModel["yunet_2023mar"] || {}) };
    this.exp.pipelineA.alignment.params = { ...this.exp.alignmentParams };
    this.exp.pipelineA.quality.params = { ...this.exp.qualityParams };
    this.exp.pipelineA.recognition.params = { ...(this.exp.recognizerParamsByModel["adaface_ir50_webface4m"] || {}) };
  },

  getDefaultParams(algorithmId) {
    const algo = this.algorithmsById[algorithmId];
    if (!algo) return {};
    const out = {};
    for (const p of algo.parameters || []) {
      out[p.name] = p.default;
    }
    return out;
  },

  resetAllActiveStagesToDefaults() {
    for (const algo of this.registry?.algorithms || []) {
      const defs = this.getDefaultParams(algo.algorithm_id);
      if (algo.category === "detection") {
        this.exp.detectorParamsByModel[algo.algorithm_id] = { ...defs };
      } else if (algo.category === "landmark_alignment") {
        this.exp.alignmentParams = { ...defs };
      } else if (algo.category === "quality") {
        this.exp.qualityParams = { ...defs };
      } else if (algo.category === "recognition") {
        this.exp.recognizerParamsByModel[algo.algorithm_id] = { ...defs };
      } else if (algo.category === "retrieval") {
        this.exp.retrievalParamsByIndex[algo.algorithm_id] = { ...defs };
      }
    }
    this.exp.qualityDegradation = { blur_sigma: 0.0, effective_res: 112, gamma: 1.0, jpeg_quality: 100 };
    if (this.exp.pipelineA) {
      const detId = this.exp.pipelineA.detection?.algorithm_id || "yunet_2023mar";
      const recId = this.exp.pipelineA.recognition?.algorithm_id || "adaface_ir50_webface4m";
      const retId = this.exp.pipelineA.retrieval?.algorithm_id || "exact_flat_index";
      this.exp.pipelineA.detection.params = { ...this.getDefaultParams(detId) };
      this.exp.pipelineA.alignment.params = { ...this.getDefaultParams("five_point_canonical_aligner") };
      this.exp.pipelineA.quality.enabled = true;
      this.exp.pipelineA.quality.params = { ...this.getDefaultParams("hybrid_classical_and_norm_quality") };
      this.exp.pipelineA.recognition.enabled = true;
      this.exp.pipelineA.recognition.params = { ...this.getDefaultParams(recId) };
      this.exp.pipelineA.normalization.enabled = true;
      this.exp.pipelineA.retrieval.params = { ...this.getDefaultParams(retId) };
    }
    this.notify("config_reset");
  },

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  },

  notify(topic = "update") {
    for (const fn of this.listeners) {
      try {
        fn(topic, this);
      } catch (err) {
        console.error("Store listener error:", err);
      }
    }
  },
};

export function showToast(message, isError = false) {
  const container = document.getElementById("wb-toast-container");
  if (!container) return;
  const el = document.createElement("div");
  el.className = `wb-toast ${isError ? "wb-toast-error" : ""}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.remove();
  }, 4800);
}
