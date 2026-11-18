/**
 * Workspace 1 — Research Laboratory Dashboard (`frontend/js/workspaces/dashboard.js`).
 *
 * Every summary metric explicitly cites its dataset, split, model, configuration, and source experiment.
 */

import { api } from "../api.js";
import { showToast, store } from "../state.js";
import { helpBadgeHtml } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderDashboard(container, navigateTo) {
  const sys = store.systemStatus || (await api.getSystemStatus());
  store.systemStatus = sys;
  const hw = sys.hardware_and_env || {};
  const gpu = hw.gpu || {};
  const algos = store.registry?.algorithms || [];
  const exps = store.experiments || [];
  const datasets = store.datasets || [];

  const detModels = algos.filter((a) => a.category === "detection");
  const recModels = algos.filter((a) => a.category === "recognition");

  // Locate provenance-tagged reference metrics from historical + workbench records
  const p3Det = exps.find((e) => e.experiment_id === "HIST-P3-EXP01-DET-SWEEP");
  const p3Ver = exps.find((e) => e.experiment_id === "HIST-P3-EXP02-ADAFACE_IR50_FP16-SCRFD_5PT_LANDMARKS");
  const p3Batch = exps.find((e) => e.experiment_id === "HIST-P3-EXP07-BATCH-SWEEP");
  const p3Ret = exps.find((e) => e.experiment_id === "HIST-P3-EXP08-RETRIEVAL-SCALING");

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>01 · Facial Recognition Research Laboratory Dashboard</h2>
        <p class="wb-workspace-desc">
          Algorithmic research, controlled parameter ablation, and reproducible benchmark inspection.
          All metrics report dataset, split, configuration, and experiment provenance — never universal rankings.
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="dash-refresh-btn" type="button" title="Poll live CPU, RAM, CUDA VRAM, and resident model cache status">Refresh Telemetry</button>
        <button class="wb-btn wb-btn-sm" data-goto="image_explorer" type="button" title="Open Workspace 02 to inspect images, folders, and face crops">Open Image Explorer →</button>
      </div>
    </div>

    <div class="wb-grid-6" style="margin-bottom:12px;">
      <div class="wb-metric-card" data-tip-title="Recorded Experiments Ledger" data-tip-body="Total historical milestone benchmarks (read-only) plus interactive Workbench runs persisted in reports/workbench_runs/.">
        <div class="wb-metric-label"><span>Experiments</span><span class="wb-badge">READ-ONLY</span></div>
        <div class="wb-metric-value">${sys.counts.total_experiments}</div>
        <div class="wb-metric-prov">${sys.counts.completed_experiments} completed · ${sys.counts.failed_jobs_session} session failures</div>
      </div>

      <div class="wb-metric-card" data-tip-title="Registered Algorithm Catalog" data-tip-body="Total detection, alignment, quality, recognition, matching, and 1:N retrieval implementations exposed from src/.">
        <div class="wb-metric-label"><span>Algorithms</span><span class="wb-badge">REGISTRY</span></div>
        <div class="wb-metric-value">${algos.length}</div>
        <div class="wb-metric-prov">${detModels.length} detectors · ${recModels.length} recognizers · 4 indexes</div>
      </div>

      <div class="wb-metric-card" data-tip-key="score_threshold">
        <div class="wb-metric-label"><span>Best Detection Recall</span><span class="wb-badge">PROVENANCE</span></div>
        <div class="wb-metric-value">${p3Det ? "100.0% (F1 1.000)" : "—"}</div>
        <div class="wb-metric-prov">Dataset: NASA P1.1 9-img/86-face (YuNet-seeded caveat) · SCRFD-500M @ 1280 sq · Src: HIST-P3-EXP01</div>
      </div>

      <div class="wb-metric-card" data-tip-key="eer">
        <div class="wb-metric-label"><span>Best Verification EER ${helpBadgeHtml("eer")}</span><span class="wb-badge">PROVENANCE</span></div>
        <div class="wb-metric-value">${p3Ver ? `${p3Ver.metrics.test_eer_pct}% EER` : "—"}</div>
        <div class="wb-metric-prov">Dataset: LFW Held-Out Folds 6–10 (3,000 pairs) · AdaFace IR-50 FP16 + SCRFD 5pt · Src: HIST-P3-EXP02</div>
      </div>

      <div class="wb-metric-card" data-tip-key="batch_policy">
        <div class="wb-metric-label"><span>Peak Rec Throughput ${helpBadgeHtml("batch_policy")}</span><span class="wb-badge">PROVENANCE</span></div>
        <div class="wb-metric-value">${p3Batch ? `${p3Batch.metrics.peak_throughput_faces_per_sec} faces/s` : "—"}</div>
        <div class="wb-metric-prov">Dataset: NASA 86-face stream · AdaFace FP16 GuardedFixedBatch B=8 · Src: HIST-P3-EXP07</div>
      </div>

      <div class="wb-metric-card" data-tip-key="recall_at_k">
        <div class="wb-metric-label"><span>1:N Search Latency ${helpBadgeHtml("recall_at_k")}</span><span class="wb-badge">PROVENANCE</span></div>
        <div class="wb-metric-value">${p3Ret ? `${p3Ret.metrics.cuda_fp16_100k_q1_p50_ms} ms` : "—"}</div>
        <div class="wb-metric-prov">Gallery: N=100K (13.2K LFW + syn) · ExactFlat CUDA FP16 (37,110 QPS @ Q=64) · Src: HIST-P3-EXP08</div>
      </div>
    </div>

    <div class="wb-grid-2">
      <!-- Hardware, Environment & Loaded Models -->
      <div class="wb-panel">
        <div class="wb-panel-header">
          <span>Hardware, Runtime Environment &amp; On-Demand Model Cache</span>
          <span class="wb-badge">${hw.loaded_models?.length || 0} Loaded in Memory</span>
        </div>
        <div class="wb-panel-body">
          <table class="wb-table">
            <tbody>
              <tr><th>GPU Device</th><td class="wb-mono">${esc(gpu.name || "CPU Only")} ${gpu.available ? "(CUDA Active)" : ""}</td></tr>
              <tr><th>GPU VRAM Budget</th><td class="wb-mono">${
                gpu.vram_total_mib
                  ? `Used ${gpu.vram_device_used_mib} / ${gpu.vram_total_mib} MiB (Free ${gpu.vram_free_mib} MiB · PyTorch Alloc ${gpu.torch_allocated_mib} MiB)`
                  : "N/A"
              }</td></tr>
              <tr><th>CPU &amp; System RAM</th><td class="wb-mono">${hw.cpu_count_physical}C/${hw.cpu_count_logical}T · CPU ${hw.cpu_percent}% · RAM ${hw.ram_used_mib} / ${hw.ram_total_mib} MiB (Workbench RSS ${hw.process_rss_mib} MiB)</td></tr>
              <tr><th>Software Stack</th><td class="wb-mono">Python ${esc(hw.python_version)} · PyTorch ${esc(hw.torch_version)} · ORT ${esc(hw.onnxruntime_version)}</td></tr>
              <tr><th>ORT Execution Providers</th><td class="wb-mono">${esc((hw.onnxruntime_providers || []).join(", "))}</td></tr>
            </tbody>
          </table>

          <div style="margin-top:10px;font-weight:600;font-size:12px;">Currently Loaded Model Sessions</div>
          ${
            (hw.loaded_models || []).length === 0
              ? `<div class="wb-help" style="margin-top:4px;">Zero models currently occupy RAM/VRAM (models instantiate lazily on demand).</div>`
              : `<div class="wb-table-wrap" style="margin-top:6px;">
                  <table class="wb-table">
                    <thead><tr><th>Model</th><th>Runtime / Device</th><th>Prec</th><th>Init (ms)</th><th>Action</th></tr></thead>
                    <tbody>
                      ${hw.loaded_models
                        .map(
                          (m) => `<tr>
                            <td class="wb-mono">${esc(m.model_id)}</td>
                            <td class="wb-mono">${esc(m.runtime)} / ${esc(m.device)}</td>
                            <td class="wb-mono">${esc(m.precision)}</td>
                            <td class="wb-mono">${m.init_time_ms}</td>
                            <td><button class="wb-btn wb-btn-secondary wb-btn-sm" data-unload="${esc(m.key)}" type="button">Unload</button></td>
                          </tr>`
                        )
                        .join("")}
                    </tbody>
                  </table>
                </div>`
          }
        </div>
      </div>

      <!-- Available Datasets & Quick Launch -->
      <div class="wb-panel">
        <div class="wb-panel-header">
          <span>Permitted Local Research Datasets (${datasets.length})</span>
          <button class="wb-btn wb-btn-secondary wb-btn-sm" data-goto="datasets" type="button">Explore Datasets</button>
        </div>
        <div class="wb-panel-body">
          <div class="wb-table-wrap">
            <table class="wb-table">
              <thead><tr><th>Dataset</th><th>Images / Pairs</th><th>GT</th><th>License</th></tr></thead>
              <tbody>
                ${datasets
                  .map(
                    (d) => `<tr>
                      <td><strong>${esc(d.name)}</strong><div class="wb-muted-xs wb-mono">${esc(d.evaluation_split)}</div></td>
                      <td class="wb-mono">${d.num_images.toLocaleString()} imgs${
                      d.has_verification_pairs ? ` · ${(d.positive_pairs + d.negative_pairs).toLocaleString()} pairs` : ""
                    }</td>
                      <td>${d.has_detection_gt ? '<span class="wb-badge wb-badge-perf">BBox GT</span>' : ""}${
                      d.has_verification_pairs ? '<span class="wb-badge wb-badge-acc">Pair GT</span>' : ""
                    }</td>
                      <td style="font-size:11px;">${esc(d.license)}</td>
                    </tr>`
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent & Historical Experiments -->
    <div class="wb-panel">
      <div class="wb-panel-header">
        <span>Recent Workbench Runs &amp; Frozen Reference Benchmarks</span>
        <div class="wb-toolbar-actions">
          <button class="wb-btn wb-btn-secondary wb-btn-sm" data-goto="comparison" type="button">Compare Runs</button>
          <button class="wb-btn wb-btn-secondary wb-btn-sm" data-goto="history" type="button">Full History (${exps.length})</button>
        </div>
      </div>
      <div class="wb-panel-body">
        <div class="wb-table-wrap">
          <table class="wb-table">
            <thead>
              <tr>
                <th>Experiment ID</th>
                <th>Name</th>
                <th>Origin</th>
                <th>Dataset &amp; Split</th>
                <th>Provider / Prec</th>
                <th>Headline Metrics (Context-Specific)</th>
              </tr>
            </thead>
            <tbody>
              ${exps
                .slice(0, 8)
                .map((e) => {
                  const mPairs = Object.entries(e.metrics || {})
                    .slice(0, 3)
                    .map(([k, v]) => `${k}: ${typeof v === "number" ? Number(v.toFixed(4)) : v}`)
                    .join(" · ");
                  return `<tr>
                    <td class="wb-mono">${esc(e.experiment_id)}</td>
                    <td><strong>${esc(e.name)}</strong></td>
                    <td><span class="wb-badge">${esc(e.source_origin === "historical_readonly" ? `FROZEN ${e.milestone}` : "WORKBENCH")}</span></td>
                    <td style="font-size:11px;">${esc(e.dataset_id)}<div class="wb-muted-xs">${esc(e.dataset_split)}</div></td>
                    <td class="wb-mono">${esc(e.execution_provider)} (${esc(e.precision)})</td>
                    <td class="wb-mono" style="font-size:11px;">${esc(mPairs || "Configuration Record")}</td>
                  </tr>`;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  container.querySelector("#dash-refresh-btn")?.addEventListener("click", async () => {
    store.systemStatus = await api.getSystemStatus();
    const expResp = await api.listExperiments(true);
    store.experiments = expResp.experiments;
    await renderDashboard(container, navigateTo);
    showToast("Telemetry and experiment inventory refreshed.");
  });

  container.querySelectorAll("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => navigateTo(btn.getAttribute("data-goto")));
  });

  container.querySelectorAll("[data-unload]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const key = btn.getAttribute("data-unload");
      await api.unloadModels({ key });
      store.systemStatus = await api.getSystemStatus();
      await renderDashboard(container, navigateTo);
      showToast(`Unloaded model session ${key}`);
    });
  });
}
