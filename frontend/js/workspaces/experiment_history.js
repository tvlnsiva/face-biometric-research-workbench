/**
 * Workspace 11 — Reproducible Experiment History & Export Manager (`frontend/js/workspaces/experiment_history.js`).
 *
 * Displays full provenance records for both immutable historical P1.1/P2/P3 benchmarks and
 * newly executed Workbench runs, with JSON/CSV/YAML export, duplication, and rerun actions.
 */

import { api } from "../api.js";
import { showToast, store } from "../state.js";
import { withBusyButton } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderExperimentHistory(container, navigateTo, waitJobFn) {
  const expResp = await api.listExperiments(true);
  if (!container.isConnected) return;
  store.experiments = expResp.experiments;
  let selectedId = store.experiments[0]?.experiment_id || null;
  let searchQuery = "";
  let filterCat = "all";
  let sortMode = "newest";

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>11 · Experiment History &amp; Reproducibility Ledger</h2>
        <p class="wb-workspace-desc">
          Every run records dataset split, SHA-256 digests, full parameter configuration, software/hardware
          environment, and metrics. Historical P1/P1.1/P2/P3 records are strictly read-only.
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="hist-refresh-btn" type="button"
          data-tip-title="Refresh Experiment Ledger"
          data-tip-body="Reload experiment records from reports/workbench/experiments/ and immutable historical baselines.">Refresh Ledger</button>
      </div>
    </div>

    <div class="wb-split-layout">
      <div class="wb-panel">
        <div class="wb-panel-header"><span id="hist-count-label">Recorded Experiments (${store.experiments.length})</span></div>
        <div style="padding:8px 10px;border-bottom:1px solid var(--border);display:grid;grid-template-columns:1fr 1fr;gap:6px;">
          <div style="grid-column:1 / -1;">
            <label for="hist-search" class="wb-muted-xs">Search ID / Name / Dataset</label>
            <input id="hist-search" type="search" class="wb-input" placeholder="e.g., AdaFace, SCRFD, HIST-P3, WB-…" />
          </div>
          <div>
            <label for="hist-filter-cat" class="wb-muted-xs">Filter Origin / Category</label>
            <select id="hist-filter-cat" class="wb-select">
              <option value="all">All Origins &amp; Categories</option>
              <option value="workbench">Workbench Interactive Runs Only</option>
              <option value="historical_readonly">Immutable Historical Baselines</option>
              <option value="cat:detection">Category: Detection</option>
              <option value="cat:recognition">Category: Recognition / 1:1</option>
              <option value="cat:retrieval">Category: 1:N Retrieval</option>
              <option value="cat:pipeline">Category: Full Pipeline</option>
            </select>
          </div>
          <div>
            <label for="hist-sort" class="wb-muted-xs">Sort Order</label>
            <select id="hist-sort" class="wb-select">
              <option value="newest">Newest Timestamp First</option>
              <option value="oldest">Oldest Timestamp First</option>
              <option value="name_asc">Name (A → Z)</option>
              <option value="id_asc">Experiment ID (A → Z)</option>
            </select>
          </div>
        </div>
        <div class="wb-panel-body" style="max-height:540px;overflow-y:auto;padding:0;">
          <table class="wb-table" id="hist-list-table">
            <thead><tr><th>ID / Origin</th><th>Name</th><th>Status</th></tr></thead>
            <tbody id="hist-list-tbody"></tbody>
          </table>
        </div>
      </div>

      <div class="wb-panel">
        <div class="wb-panel-header" style="flex-wrap:wrap;gap:6px;">
          <span>Full Reproducibility Record &amp; Export Actions</span>
          <div class="wb-toolbar-actions" id="hist-detail-actions"></div>
        </div>
        <div class="wb-panel-body" id="hist-detail-body"></div>
      </div>
    </div>
  `;

  function getFilteredAndSorted() {
    const q = searchQuery.trim().toLowerCase();
    const filtered = store.experiments.filter((e) => {
      if (filterCat === "workbench" && e.source_origin === "historical_readonly") return false;
      if (filterCat === "historical_readonly" && e.source_origin !== "historical_readonly") return false;
      if (filterCat.startsWith("cat:") && e.category !== filterCat.slice(4)) return false;
      if (!q) return true;
      const hay = `${e.experiment_id} ${e.name} ${e.dataset_id} ${e.category} ${e.milestone}`.toLowerCase();
      return hay.includes(q);
    });
    filtered.sort((a, b) => {
      if (sortMode === "oldest") return String(a.timestamp_utc || "").localeCompare(String(b.timestamp_utc || ""));
      if (sortMode === "name_asc") return String(a.name || "").localeCompare(String(b.name || ""));
      if (sortMode === "id_asc") return String(a.experiment_id || "").localeCompare(String(b.experiment_id || ""));
      return String(b.timestamp_utc || "").localeCompare(String(a.timestamp_utc || ""));
    });
    return filtered;
  }

  function renderTableRows() {
    const rows = getFilteredAndSorted();
    const countLbl = container.querySelector("#hist-count-label");
    countLbl.textContent = `Recorded Experiments (${rows.length} / ${store.experiments.length})`;
    const tbody = container.querySelector("#hist-list-tbody");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="3" class="wb-help" style="padding:12px;">No experiments match the active search/filter criteria.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows
      .map(
        (e) => `
        <tr data-exp-row="${esc(e.experiment_id)}" class="${
          e.experiment_id === selectedId ? "selected-row" : ""
        }" style="cursor:pointer;">
          <td class="wb-mono" style="font-size:10.5px;">
            <div>${esc(e.experiment_id)}</div>
            <span class="wb-badge">${
              e.source_origin === "historical_readonly" ? `IMMUTABLE ${esc(e.milestone)}` : esc(e.source_origin)
            }</span>
          </td>
          <td style="font-size:11.5px;"><strong>${esc(e.name)}</strong><div class="wb-muted-xs">${esc(
          e.timestamp_utc
        )}</div></td>
          <td class="wb-mono" style="font-size:11px;">${esc(e.status)}</td>
        </tr>`
      )
      .join("");

    tbody.querySelectorAll("[data-exp-row]").forEach((tr) => {
      tr.addEventListener("click", () => {
        selectedId = tr.getAttribute("data-exp-row");
        tbody.querySelectorAll("[data-exp-row]").forEach((r) => {
          r.classList.toggle("selected-row", r.getAttribute("data-exp-row") === selectedId);
        });
        renderDetail(selectedId);
      });
    });
  }

  async function renderDetail(expId) {
    if (!expId) return;
    const rec = await api.getExperiment(expId);
    if (!container.isConnected) return;
    const actBar = container.querySelector("#hist-detail-actions");
    const isReadonly = rec.source_origin === "historical_readonly";
    actBar.innerHTML = `
      <a class="wb-btn wb-btn-secondary wb-btn-sm" href="${api.exportUrl(expId, "json")}" download style="text-decoration:none;"
        data-tip-title="Export Reproducibility JSON"
        data-tip-body="Download full provenance JSON including parameters, SHA-256 hashes, hardware environment, and metrics.">Export JSON</a>
      <a class="wb-btn wb-btn-secondary wb-btn-sm" href="${api.exportUrl(expId, "csv")}" download style="text-decoration:none;"
        data-tip-title="Export Flat Metrics CSV"
        data-tip-body="Download tabular summary CSV suitable for spreadsheets or external notebooks.">Export CSV</a>
      <a class="wb-btn wb-btn-secondary wb-btn-sm" href="${api.exportUrl(expId, "yaml")}" download style="text-decoration:none;"
        data-tip-title="Export Configuration YAML"
        data-tip-body="Download reproducible YAML configuration manifest.">Export YAML</a>
      <button class="wb-btn wb-btn-secondary wb-btn-sm" id="hist-add-cmp" type="button"
        data-tip-title="Add to Comparison Matrix"
        data-tip-body="Select this experiment in Workspace 09 for side-by-side metric comparison.">+ Compare</button>
      ${
        !isReadonly
          ? `<button class="wb-btn wb-btn-secondary wb-btn-sm" id="hist-load-cfg-btn" type="button"
               data-tip-title="Load Configuration Into Lab"
               data-tip-body="Restore this experiment's exact algorithm and parameter choices into the corresponding workspace.">Load Config Into State</button>
             <button class="wb-btn wb-btn-sm" id="hist-rerun-btn" type="button"
               data-tip-title="Reproduce Experiment Run"
               data-tip-body="Re-execute this exact configuration against the backend and record a new timestamped verification run.">↻ Reproduce Run</button>`
          : `<button class="wb-btn wb-btn-secondary wb-btn-sm" type="button" disabled
               title="Immutable P1/P1.1/P2/P3 baseline record — direct overwrite/rerun disabled to preserve historical benchmark integrity."
               data-tip-title="Immutable Historical Baseline"
               data-tip-body="Frozen P1/P1.1/P2/P3 benchmark records are read-only so historical numbers are never overwritten.">Read-Only Baseline</button>`
      }
    `;

    container.querySelector("#hist-detail-body").innerHTML = `
      <div class="wb-grid-2" style="margin-bottom:10px;">
        <table class="wb-table">
          <tbody>
            <tr><th>Experiment ID</th><td class="wb-mono">${esc(rec.experiment_id)}</td></tr>
            <tr><th>Source Origin</th><td><span class="wb-badge">${esc(rec.source_origin)}</span> (${esc(rec.milestone)})</td></tr>
            <tr><th>Timestamp (UTC)</th><td class="wb-mono">${esc(rec.timestamp_utc)}</td></tr>
            <tr><th>Dataset &amp; Split</th><td class="wb-mono">${esc(rec.dataset_id)} · ${esc(rec.dataset_split)}</td></tr>
            <tr><th>Manifest SHA-256</th><td class="wb-mono" style="font-size:10.5px;word-break:break-all;">${esc(
              rec.dataset_manifest_sha256 || "—"
            )}</td></tr>
          </tbody>
        </table>
        <table class="wb-table">
          <tbody>
            <tr><th>Execution Provider</th><td class="wb-mono">${esc(rec.execution_provider)}</td></tr>
            <tr><th>Precision / Batch / Seed</th><td class="wb-mono">${esc(rec.precision)} · B=${rec.batch_size} · seed=${
      rec.seed ?? "—"
    }</td></tr>
            <tr><th>Algorithms</th><td class="wb-mono" style="font-size:11px;">${esc(JSON.stringify(rec.algorithms))}</td></tr>
            <tr><th>Model Hashes (SHA-256)</th><td class="wb-mono" style="font-size:10px;word-break:break-all;">${esc(
              JSON.stringify(rec.model_hashes || {})
            )}</td></tr>
            <tr><th>Software Env</th><td class="wb-mono" style="font-size:10.5px;">${esc(
              JSON.stringify(rec.software_environment || {})
            )}</td></tr>
            <tr><th>Artifacts</th><td class="wb-mono" style="font-size:10.5px;">${esc(
              (rec.artifact_refs || []).join(", ")
            )}</td></tr>
            <tr><th>Notes</th><td style="font-size:11px;">${esc(rec.notes || "—")}</td></tr>
          </tbody>
        </table>
      </div>

      <div class="wb-grid-2">
        <div>
          <div style="font-weight:600;margin-bottom:4px;">Measured Metrics</div>
          <pre class="wb-mono" style="background:var(--bg-metric);padding:8px;border-radius:4px;max-height:240px;overflow:auto;font-size:11px;">${esc(
            JSON.stringify(rec.metrics, null, 2)
          )}</pre>
        </div>
        <div>
          <div style="font-weight:600;margin-bottom:4px;">Full Parameter Configuration</div>
          <pre class="wb-mono" style="background:var(--bg-metric);padding:8px;border-radius:4px;max-height:240px;overflow:auto;font-size:11px;">${esc(
            JSON.stringify(rec.parameters, null, 2)
          )}</pre>
        </div>
      </div>
    `;

    container.querySelector("#hist-add-cmp")?.addEventListener("click", () => {
      if (!store.exp.selectedExperimentIds.includes(expId)) {
        store.exp.selectedExperimentIds.push(expId);
      }
      showToast(`Added ${expId} to Algorithm Comparison.`);
      navigateTo("comparison");
    });

    container.querySelector("#hist-load-cfg-btn")?.addEventListener("click", () => {
      const pCopy = JSON.parse(JSON.stringify(rec.parameters || {}));
      if (rec.category === "pipeline" || rec.category === "pipeline_builder") {
        store.exp.pipelineA = pCopy;
        if (pCopy.detection?.algorithm_id) {
          store.exp.detectorId = pCopy.detection.algorithm_id;
          store.exp.detectorParamsByModel[pCopy.detection.algorithm_id] = {
            ...store.getDefaultParams(pCopy.detection.algorithm_id),
            ...(pCopy.detection.params || {}),
          };
        }
        if (pCopy.alignment?.params) {
          store.exp.alignmentParams = { ...store.getDefaultParams("five_point_canonical_aligner"), ...pCopy.alignment.params };
        }
        if (pCopy.quality?.params) {
          store.exp.qualityParams = { ...store.getDefaultParams("hybrid_classical_and_norm_quality"), ...pCopy.quality.params };
        }
        if (pCopy.recognition?.algorithm_id) {
          store.exp.recognizerId = pCopy.recognition.algorithm_id;
          store.exp.recognizerParamsByModel[pCopy.recognition.algorithm_id] = {
            ...store.getDefaultParams(pCopy.recognition.algorithm_id),
            ...(pCopy.recognition.params || {}),
          };
        }
        if (pCopy.retrieval?.algorithm_id) {
          store.exp.retrievalIndexId = pCopy.retrieval.algorithm_id;
          store.exp.retrievalParamsByIndex[pCopy.retrieval.algorithm_id] = {
            ...store.getDefaultParams(pCopy.retrieval.algorithm_id),
            ...(pCopy.retrieval.params || {}),
          };
        }
        store.notify("config_change");
        showToast(`Loaded pipeline configuration from ${expId}`);
        navigateTo("pipeline_builder");
      } else if (rec.category === "detection" && rec.algorithms?.detector) {
        store.exp.detectorId = rec.algorithms.detector;
        if (pCopy.sweep_param) {
          store.exp.detectorParamsByModel[rec.algorithms.detector] = {
            ...store.getDefaultParams(rec.algorithms.detector),
            ...(pCopy.base_params || {}),
          };
        } else {
          store.exp.detectorParamsByModel[rec.algorithms.detector] = {
            ...store.getDefaultParams(rec.algorithms.detector),
            ...pCopy,
          };
        }
        store.notify("config_change");
        showToast(`Loaded detection configuration from ${expId}`);
        navigateTo("detection_lab");
      } else if (rec.category === "recognition") {
        if (pCopy.image_a_ref) store.exp.imageARef = pCopy.image_a_ref;
        if (pCopy.image_b_ref) store.exp.imageBRef = pCopy.image_b_ref;
        if (pCopy.recognizer_id) {
          store.exp.recognizerId = pCopy.recognizer_id;
          store.exp.recognizerParamsByModel[pCopy.recognizer_id] = {
            ...store.getDefaultParams(pCopy.recognizer_id),
            ...(pCopy.recognizer_params || {}),
          };
        }
        store.notify("config_change");
        showToast(`Loaded 1:1 verification configuration from ${expId}`);
        navigateTo("recognition_lab");
      } else if (rec.category === "retrieval" && pCopy.index_id) {
        store.exp.retrievalIndexId = pCopy.index_id;
        store.exp.retrievalParamsByIndex[pCopy.index_id] = {
          ...store.getDefaultParams(pCopy.index_id),
          ...(pCopy.index_params || {}),
        };
        if (pCopy.gallery_size) store.exp.retrievalGallerySize = pCopy.gallery_size;
        store.notify("config_change");
        showToast(`Loaded retrieval configuration from ${expId}`);
        navigateTo("retrieval_lab");
      }
    });

    const rerunBtn = container.querySelector("#hist-rerun-btn");
    rerunBtn?.addEventListener("click", () =>
      withBusyButton(
        rerunBtn,
        async () => {
          const cfgCopy = JSON.parse(JSON.stringify(rec.parameters || {}));
          if (rec.category === "pipeline") {
            store.exp.pipelineA = cfgCopy;
            const sub = await api.submitJob({
              job_type: "pipeline_run",
              name: `${rec.name} (Reproduced)`,
              config: cfgCopy,
              save_experiment: true,
            });
            showToast(`Queued reproduced pipeline run ${sub.job_id}`);
            await waitJobFn(sub.job_id);
          } else if (rec.category === "retrieval" && cfgCopy.index_id) {
            const sub = await api.submitJob({
              job_type: "retrieval_benchmark",
              name: `${rec.name} (Reproduced)`,
              config: cfgCopy,
              save_experiment: true,
            });
            showToast(`Queued reproduced retrieval run ${sub.job_id}`);
            await waitJobFn(sub.job_id);
          } else if (rec.category === "recognition" && cfgCopy.image_a_ref) {
            await api.runVerification({
              ...cfgCopy,
              save_experiment: true,
              experiment_name: `${rec.name} (Reproduced)`,
            });
            showToast("Reproduced 1:1 verification run.");
          } else if (rec.category === "detection" && rec.dataset_id === "single_image") {
            await api.runDetection({
              image_ref: rec.dataset_split,
              detector_id: rec.algorithms?.detector || "yunet_2023mar",
              params: cfgCopy,
              save_experiment: true,
              experiment_name: `${rec.name} (Reproduced)`,
            });
            showToast("Reproduced single-image detection run.");
          } else {
            showToast("Loaded configuration; open the corresponding Lab to run sweep.", false);
            return;
          }
          if (container.isConnected) {
            await renderExperimentHistory(container, navigateTo, waitJobFn);
          }
        },
        "Reproducing…"
      )
    );
  }

  container.querySelector("#hist-search").addEventListener("input", (e) => {
    searchQuery = e.target.value;
    renderTableRows();
  });
  container.querySelector("#hist-filter-cat").addEventListener("change", (e) => {
    filterCat = e.target.value;
    renderTableRows();
  });
  container.querySelector("#hist-sort").addEventListener("change", (e) => {
    sortMode = e.target.value;
    renderTableRows();
  });

  container.querySelector("#hist-refresh-btn").addEventListener("click", () =>
    renderExperimentHistory(container, navigateTo, waitJobFn)
  );

  renderTableRows();
  if (selectedId) await renderDetail(selectedId);
}
