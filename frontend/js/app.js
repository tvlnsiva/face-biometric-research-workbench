/**
 * Main Application Bootstrapper & 12-Workspace Router (`frontend/js/app.js`).
 */

import { api } from "./api.js";
import { computeCrossStageDelta, renderConfigDeltaPanel } from "./components/config_delta.js";
import { showToast, store } from "./state.js";
import { hideTooltip, initTooltipManager } from "./utils/tooltips.js";
import { renderAlignmentLab } from "./workspaces/alignment_lab.js";
import { renderComparisonWorkspace } from "./workspaces/comparison_workspace.js";
import { renderDashboard } from "./workspaces/dashboard.js";
import { renderDatasetExplorer } from "./workspaces/dataset_explorer.js";
import { renderDetectionLab } from "./workspaces/detection_lab.js";
import { renderExperimentHistory } from "./workspaces/experiment_history.js";
import { renderImageExplorer } from "./workspaces/image_explorer.js";
import { renderPipelineBuilder } from "./workspaces/pipeline_builder.js";
import { renderQualityLab } from "./workspaces/quality_lab.js";
import { renderRecognitionLab } from "./workspaces/recognition_lab.js";
import { renderRegistryBrowser } from "./workspaces/registry_browser.js";
import { renderRetrievalLab } from "./workspaces/retrieval_lab.js";

const WORKSPACE_RENDERERS = {
  dashboard: renderDashboard,
  image_explorer: renderImageExplorer,
  detection_lab: renderDetectionLab,
  alignment_lab: renderAlignmentLab,
  quality_lab: renderQualityLab,
  recognition_lab: renderRecognitionLab,
  retrieval_lab: renderRetrievalLab,
  pipeline_builder: renderPipelineBuilder,
  comparison: renderComparisonWorkspace,
  datasets: renderDatasetExplorer,
  history: renderExperimentHistory,
  registry: renderRegistryBrowser,
};

let activeJobId = null;

function updateTopTelemetryBar(sys) {
  if (!sys) return;
  const hw = sys.hardware_and_env || {};
  const gpu = hw.gpu || {};
  const loadedCount = (hw.loaded_models || []).length;
  const gpuEl = document.getElementById("wb-gpu-text");
  if (gpuEl) {
    gpuEl.textContent = gpu.available
      ? `${gpu.name.replace("NVIDIA GeForce ", "")} · VRAM ${gpu.vram_device_used_mib}/${gpu.vram_total_mib} MiB · ${loadedCount} Loaded`
      : `CPU Mode · RAM ${hw.ram_used_mib}/${hw.ram_total_mib} MiB · ${loadedCount} Loaded`;
  }

  const ctxEl = document.getElementById("wb-context-summary");
  if (ctxEl) {
    const delta = computeCrossStageDelta(store.algorithmsById, store.exp, { scope: "global" });
    ctxEl.textContent = `Det: ${store.exp.detectorId} · Rec: ${store.exp.recognizerId} · ${delta.totalModified} mod`;
    const cntEl = document.getElementById("wb-config-delta-count");
    if (cntEl) cntEl.textContent = String(delta.totalModified);
  }

  const jobPill = document.getElementById("wb-active-job-pill");
  const jobText = document.getElementById("wb-job-status-text");
  const cancelBtn = document.getElementById("wb-job-cancel-btn");
  if (sys.active_job) {
    activeJobId = sys.active_job.job_id;
    jobPill.classList.add("wb-pill-running");
    jobText.textContent = `${sys.active_job.progress_pct}% · ${sys.active_job.stage_message}`;
    cancelBtn.classList.remove("hidden");
  } else {
    activeJobId = null;
    jobPill.classList.remove("wb-pill-running");
    jobText.textContent = "Queue Idle";
    cancelBtn.classList.add("hidden");
  }
}

export async function waitJobCompletion(jobId) {
  while (true) {
    const j = await api.getJob(jobId);
    const sys = await api.getSystemStatus();
    store.systemStatus = sys;
    updateTopTelemetryBar(sys);

    if (j.status === "completed") {
      const expResp = await api.listExperiments(true);
      store.experiments = expResp.experiments;
      showToast(`Experiment completed${j.experiment_id ? ` (${j.experiment_id})` : ""}.`);
      return j;
    }
    if (j.status === "cancelled") {
      showToast(`Job ${jobId} cancelled.`);
      return j;
    }
    if (j.status === "failed") {
      showToast(`Job failed: ${j.error?.message || j.stage_message}`, true);
      return j;
    }
    await new Promise((r) => setTimeout(r, 450));
  }
}

export async function navigateTo(workspaceId) {
  if (!WORKSPACE_RENDERERS[workspaceId]) return;
  hideTooltip(true);
  store.ui.activeWorkspace = workspaceId;

  document.querySelectorAll(".wb-nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-workspace") === workspaceId);
  });

  const mainEl = document.getElementById("wb-main-viewport");
  try {
    await WORKSPACE_RENDERERS[workspaceId](mainEl, navigateTo, waitJobCompletion);
    const sys = await api.getSystemStatus();
    store.systemStatus = sys;
    updateTopTelemetryBar(sys);
  } catch (err) {
    console.error("Workspace render error:", err);
    showToast(`Failed to render workspace '${workspaceId}': ${err.message}`, true);
  }
}

async function initWorkbench() {
  initTooltipManager();
  try {
    const [reg, dsResp, expResp, sys] = await Promise.all([
      api.getRegistry(),
      api.listDatasets(),
      api.listExperiments(true),
      api.getSystemStatus(),
    ]);

    store.initFromRegistry(reg);
    store.datasets = dsResp.datasets || [];
    store.experiments = expResp.experiments || [];
    store.systemStatus = sys;
    updateTopTelemetryBar(sys);

    // Navigation bindings
    document.querySelectorAll(".wb-nav-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        navigateTo(btn.getAttribute("data-workspace"));
      });
    });

    // Theme toggle (Light default, optional Dark)
    document.getElementById("wb-theme-toggle")?.addEventListener("click", (e) => {
      store.ui.theme = store.ui.theme === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", store.ui.theme);
      e.target.textContent = `Theme: ${store.ui.theme === "light" ? "Light" : "Dark"}`;
    });

    document.getElementById("wb-gpu-pill")?.addEventListener("click", () => {
      navigateTo("registry");
    });

    document.getElementById("wb-release-models-btn")?.addEventListener("click", async () => {
      const res = await api.unloadModels({ unload_all: true });
      store.systemStatus = await api.getSystemStatus();
      updateTopTelemetryBar(store.systemStatus);
      showToast(`Released ${res.unloaded_keys.length} cached model(s).`);
    });

    document.getElementById("wb-job-cancel-btn")?.addEventListener("click", async () => {
      if (activeJobId) {
        await api.cancelJob(activeJobId);
        showToast(`Cancellation requested for ${activeJobId}.`);
      }
    });

    // Global P-1 Active Configuration Delta Drawer
    const deltaBtn = document.getElementById("wb-config-delta-btn");
    const deltaDrawer = document.getElementById("wb-global-delta-drawer");
    let globalDeltaHandle = null;

    const syncGlobalDeltaBadgeAndDrawer = () => {
      const delta = computeCrossStageDelta(store.algorithmsById, store.exp, { scope: "global" });
      const cntEl = document.getElementById("wb-config-delta-count");
      if (cntEl) cntEl.textContent = String(delta.totalModified);
      const ctxEl = document.getElementById("wb-context-summary");
      if (ctxEl) {
        ctxEl.textContent = `Det: ${store.exp.detectorId} · Rec: ${store.exp.recognizerId} · ${delta.totalModified} mod`;
      }
      if (store.ui.configDeltaDrawerOpen && deltaDrawer) {
        globalDeltaHandle = renderConfigDeltaPanel(deltaDrawer, {
          algorithmsById: store.algorithmsById,
          expState: store.exp,
          scope: "global",
          title: "Active Configuration Delta (Global 5-Stage Laboratory State)",
          onResetConfirmed: async () => {
            store.resetAllActiveStagesToDefaults();
            showToast("Reset all 5 active stages to frozen registry defaults (historical records preserved).");
            await navigateTo(store.ui.activeWorkspace);
          },
          onClose: () => {
            store.ui.configDeltaDrawerOpen = false;
            deltaDrawer.classList.add("hidden");
            deltaBtn?.setAttribute("aria-expanded", "false");
          },
        });
      }
    };

    deltaBtn?.addEventListener("click", () => {
      store.ui.configDeltaDrawerOpen = !store.ui.configDeltaDrawerOpen;
      if (store.ui.configDeltaDrawerOpen) {
        deltaDrawer?.classList.remove("hidden");
        deltaBtn.setAttribute("aria-expanded", "true");
        syncGlobalDeltaBadgeAndDrawer();
      } else {
        deltaDrawer?.classList.add("hidden");
        deltaBtn.setAttribute("aria-expanded", "false");
      }
    });

    store.subscribe((topic) => {
      if (topic === "config_change" || topic === "config_reset" || topic === "update") {
        syncGlobalDeltaBadgeAndDrawer();
      }
    });
    syncGlobalDeltaBadgeAndDrawer();

    await navigateTo("dashboard");
  } catch (err) {
    const mainEl = document.getElementById("wb-main-viewport");
    mainEl.innerHTML = `<div class="wb-banner wb-banner-nonmatch">Workbench Initialization Failed: ${err.message}</div>`;
  }
}

initWorkbench();
