/**
 * Typed HTTP/JSON Client for Face Biometric Research Workbench Backend (`/api/v1/*`).
 */

export class ApiError extends Error {
  constructor(message, code = "API_ERROR", status = 400, details = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

async function requestJson(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let data = null;
  try {
    data = await resp.json();
  } catch (_err) {
    if (!resp.ok) {
      throw new ApiError(`HTTP ${resp.status} ${resp.statusText}`, "HTTP_ERROR", resp.status);
    }
  }
  if (!resp.ok) {
    const errObj = (data && data.error) || {};
    throw new ApiError(
      errObj.message || `HTTP ${resp.status}`,
      errObj.code || "HTTP_ERROR",
      resp.status,
      errObj.details || {}
    );
  }
  return data;
}

export const api = {
  rawImageUrl(imageRef) {
    return `/api/v1/images/raw?ref=${encodeURIComponent(imageRef)}`;
  },
  exportUrl(experimentId, fmt = "json") {
    return `/api/v1/experiments/${encodeURIComponent(experimentId)}/export?format=${encodeURIComponent(fmt)}`;
  },
  getSystemStatus() {
    return requestJson("/api/v1/system/status");
  },
  getRegistry() {
    return requestJson("/api/v1/registry");
  },
  loadModel(payload) {
    return requestJson("/api/v1/models/load", { method: "POST", body: JSON.stringify(payload) });
  },
  unloadModels(payload) {
    return requestJson("/api/v1/models/unload", { method: "POST", body: JSON.stringify(payload) });
  },
  listDatasets() {
    return requestJson("/api/v1/datasets");
  },
  getDatasetItems(datasetId, { kind = "images", split = "all", offset = 0, limit = 50 } = {}) {
    const q = new URLSearchParams({ kind, split, offset: String(offset), limit: String(limit) });
    return requestJson(`/api/v1/datasets/${encodeURIComponent(datasetId)}/items?${q.toString()}`);
  },
  inspectImage(imageRef) {
    return requestJson("/api/v1/images/inspect", { method: "POST", body: JSON.stringify({ image_ref: imageRef }) });
  },
  scanDirectory(directoryPath, limit = 120) {
    return requestJson("/api/v1/images/scan-directory", {
      method: "POST",
      body: JSON.stringify({ directory_path: directoryPath, limit }),
    });
  },
  uploadLocalImage(filename, dataBase64) {
    return requestJson("/api/v1/images/upload", {
      method: "POST",
      body: JSON.stringify({ filename, data_base64: dataBase64 }),
    });
  },
  runDetection(payload) {
    return requestJson("/api/v1/detect/run", { method: "POST", body: JSON.stringify(payload) });
  },
  runAlignment(payload) {
    return requestJson("/api/v1/align/run", { method: "POST", body: JSON.stringify(payload) });
  },
  runQuality(payload) {
    return requestJson("/api/v1/quality/run", { method: "POST", body: JSON.stringify(payload) });
  },
  runVerification(payload) {
    return requestJson("/api/v1/verify/run", { method: "POST", body: JSON.stringify(payload) });
  },
  runIdentification(payload) {
    return requestJson("/api/v1/identify/search", { method: "POST", body: JSON.stringify(payload) });
  },
  validatePipeline(payload) {
    return requestJson("/api/v1/pipeline/validate", { method: "POST", body: JSON.stringify(payload) });
  },
  submitJob(payload) {
    return requestJson("/api/v1/jobs/submit", { method: "POST", body: JSON.stringify(payload) });
  },
  listJobs() {
    return requestJson("/api/v1/jobs");
  },
  getJob(jobId) {
    return requestJson(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
  },
  cancelJob(jobId) {
    return requestJson(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", body: "{}" });
  },
  listExperiments(includeHistorical = true) {
    return requestJson(`/api/v1/experiments?include_historical=${includeHistorical ? "true" : "false"}`);
  },
  getExperiment(experimentId) {
    return requestJson(`/api/v1/experiments/${encodeURIComponent(experimentId)}`);
  },
  saveNamedConfig(payload) {
    return requestJson("/api/v1/experiments/save-config", { method: "POST", body: JSON.stringify(payload) });
  },
};
