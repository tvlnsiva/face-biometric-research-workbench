/**
 * Workspace 8 — Visual Pipeline Experiment Builder (`frontend/js/workspaces/pipeline_builder.js`).
 *
 * Configures the 8-stage processing pipeline, validates cross-stage compatibility prior to execution,
 * supports saving/duplicating/comparing two pipelines, and renders the live execution graph.
 */

import { api } from "../api.js";
import { renderConfigDeltaPanel } from "../components/config_delta.js";
import { renderParameterEditor } from "../components/param_editor.js";
import { renderScoreDistributionExplorer } from "../components/threshold_explorer.js";
import { showToast, store } from "../state.js";
import { helpBadgeHtml, sanitizeNumericInput, withBusyButton } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderPipelineBuilder(container, navigateTo, waitJobFn) {
  const pipe = store.exp.pipelineA;
  const algos = store.registry?.algorithms || [];
  const detAlgos = algos.filter((a) => a.category === "detection");
  const recAlgos = algos.filter((a) => a.category === "recognition");
  const retAlgos = algos.filter((a) => a.category === "retrieval");

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>08 · Visual Pipeline Experiment Builder</h2>
        <p class="wb-workspace-desc">
          Assemble, validate, save, duplicate, and execute end-to-end multi-stage pipelines
          (Image Input → Detection → Alignment → Quality → Recognition → Normalization → Matching/Retrieval → Evaluation).
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="pb-validate-btn" type="button" title="Check cross-stage compatibility, landmark dependencies, and local model weight availability">✓ Validate Compatibility</button>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="pb-save-cfg-btn" type="button" title="Save this 8-stage pipeline configuration into Experiment History as a reusable preset">Save Pipeline Config</button>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="pb-dup-btn" type="button" title="Clone current pipeline settings into the side-by-side snapshot comparison table below">Duplicate Snapshot</button>
        <button class="wb-btn wb-btn-sm" id="pb-run-btn" type="button" title="Validate and submit this end-to-end 8-stage pipeline to the serialized execution queue">▶ Execute Pipeline</button>
      </div>
    </div>

    <div id="pb-validation-banner"></div>

    <!-- P-1: Inline Pipeline Active Configuration Delta -->
    <div id="pb-config-delta-slot"></div>

    <!-- Top Metadata & Dataset Selector -->
    <div class="wb-panel">
      <div class="wb-panel-body wb-filter-grid-pb">
        <div>
          <label class="wb-label" for="pb-name">Pipeline Experiment Name</label>
          <input id="pb-name" class="wb-input" value="${esc(pipe.name)}" title="Human-readable experiment label recorded in Experiment History" />
        </div>
        <div data-tip-key="identity_disjoint_split">
          <label class="wb-label" for="pb-dataset">Evaluation Dataset Split ${helpBadgeHtml("identity_disjoint_split")}</label>
          <select id="pb-dataset" class="wb-select" data-tip-key="identity_disjoint_split">
            <option value="nasa_p11_multiscale" ${pipe.dataset_id === "nasa_p11_multiscale" ? "selected" : ""}>NASA P1.1 Detection Split (9 Imgs / 86 Faces)</option>
            <option value="nasa_p1_detection" ${pipe.dataset_id === "nasa_p1_detection" ? "selected" : ""}>NASA P1 Detection Split (6 Imgs / 15 Faces)</option>
            <option value="nasa_p11_verification" ${pipe.dataset_id === "nasa_p11_verification" ? "selected" : ""}>NASA P1.1 Verification (120 Pairs — Smoke Test)</option>
            <option value="lfw_research_split" ${pipe.dataset_id === "lfw_research_split" ? "selected" : ""}>LFW Identity-Disjoint Sample Pairs</option>
          </select>
        </div>
        <div>
          <label class="wb-label" for="pb-pair-limit">LFW Pair Cap</label>
          <input type="number" id="pb-pair-limit" class="wb-input wb-mono" min="20" max="600" value="${pipe.pair_limit}" title="Total stratified pairs (50% Calib Folds 1–5 + 50% Held-Out Test Folds 6–10) [20..600]" />
        </div>
        <div>
          <label class="wb-label" for="pb-order-mode">Stage Ordering (Quality ↔ Recognition)</label>
          <select id="pb-order-mode" class="wb-select" title="Evaluate classical pixel quality before CNN inference, or after CNN inference alongside ||z_raw||_2">
            <option value="standard">Standard: Quality (#4) → Recognition (#5)</option>
            <option value="post_rec_quality">Reordered: Recognition (#4) → Quality + ||z||_2 (#5)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Stage Configuration Grid -->
    <div class="wb-grid-3">
      <!-- Stage 2: Detection -->
      <div class="wb-panel">
        <div class="wb-panel-header">
          <span>Stage 2 · Face &amp; Landmark Detection</span>
          <label style="font-size:11px;" title="5-point landmarks are required by Stage 3 Alignment"><input type="checkbox" id="pb-lm-enable" ${pipe.detection.enable_landmarks ? "checked" : ""}/> Landmarks</label>
        </div>
        <div class="wb-panel-body">
          <div class="wb-field">
            <label class="wb-label" for="pb-det-id">Detector</label>
            <select id="pb-det-id" class="wb-select" title="Select face & 5-point landmark detector for Stage 2">
              ${detAlgos
                .map(
                  (a) =>
                    `<option value="${esc(a.algorithm_id)}" ${
                      a.algorithm_id === pipe.detection.algorithm_id ? "selected" : ""
                    }>${esc(a.name)}</option>`
                )
                .join("")}
            </select>
          </div>
          <div id="pb-det-params"></div>
        </div>
      </div>

      <!-- Stage 3 & 4: Alignment + Optional Quality -->
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Stage 3 · 5-Point Alignment ${helpBadgeHtml("transform_mode")}</span></div>
          <div class="wb-panel-body" id="pb-align-params"></div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header">
            <span>Stage 4 · Quality Assessment (Optional)</span>
            <label style="font-size:11px;" title="Enable or skip Stage 4 quality assessment"><input type="checkbox" id="pb-qual-enable" ${pipe.quality.enabled ? "checked" : ""}/> Enabled</label>
          </div>
          <div class="wb-panel-body">
            <div class="wb-help" style="margin-bottom:6px;">Computes P1 classical warnings and P3 sharpness/exposure metrics on aligned crops.</div>
            <details>
              <summary style="cursor:pointer;font-size:11.5px;font-weight:600;color:var(--accent);">Stage 4 Quality Parameters</summary>
              <div id="pb-qual-params" style="margin-top:8px;"></div>
            </details>
          </div>
        </div>
      </div>

      <!-- Stage 5, 6, 7: Recognition, Normalization & Matching/Retrieval -->
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header">
            <span>Stage 5 · Face Recognition ${helpBadgeHtml("embedding")}</span>
            <label style="font-size:11px;" title="Enable or skip Stage 5 embedding extraction"><input type="checkbox" id="pb-rec-enable" ${pipe.recognition.enabled ? "checked" : ""}/> Enabled</label>
          </div>
          <div class="wb-panel-body">
            <div class="wb-field">
              <label class="wb-label" for="pb-rec-id">Recognizer Model</label>
              <select id="pb-rec-id" class="wb-select" title="Select deep face embedding backbone for Stage 5">
                ${recAlgos
                  .map(
                    (a) =>
                      `<option value="${esc(a.algorithm_id)}" ${
                        a.algorithm_id === pipe.recognition.algorithm_id ? "selected" : ""
                      }>${esc(a.name)}</option>`
                  )
                  .join("")}
              </select>
            </div>
            <label style="display:inline-flex;align-items:center;gap:6px;font-size:11.5px;margin-bottom:8px;" title="Fail pre-validation if selected model weights are missing on disk (recommended)">
              <input type="checkbox" id="pb-strict-weights" ${pipe.recognition.strict_weights ? "checked" : ""} />
              Strict Local Weights Check
            </label>
            <div id="pb-rec-params"></div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header">
            <span>Stages 6–7 · Normalization, Matching &amp; Retrieval</span>
          </div>
          <div class="wb-panel-body">
            <label style="display:flex;align-items:center;gap:6px;margin-bottom:8px;" data-tip-key="l2_normalize">
              <input type="checkbox" id="pb-norm-enable" ${pipe.normalization.enabled ? "checked" : ""} />
              <strong>Stage 6: Enable Unit Hypersphere L2 Normalization</strong> ${helpBadgeHtml("l2_normalize")}
            </label>
            <div class="wb-field" data-tip-key="cosine">
              <label class="wb-label" for="pb-metric">Verification Metric ${helpBadgeHtml("cosine")}</label>
              <select id="pb-metric" class="wb-select" data-tip-key="cosine">
                <option value="cosine" ${pipe.matching.metric === "cosine" ? "selected" : ""}>Cosine Similarity</option>
                <option value="normalized_l2" ${pipe.matching.metric === "normalized_l2" ? "selected" : ""}>Normalized L2</option>
                <option value="unnormalized_l2" ${pipe.matching.metric === "unnormalized_l2" ? "selected" : ""}>Unnormalized L2</option>
              </select>
            </div>
            <label style="display:flex;align-items:center;gap:6px;margin-bottom:6px;" title="Also construct and query a Stage 7 1:N retrieval index over extracted embeddings">
              <input type="checkbox" id="pb-ret-enable" ${pipe.retrieval.enabled ? "checked" : ""} />
              <strong>Enable Stage 7 1:N Retrieval Index Build</strong>
            </label>
            <label class="wb-label" for="pb-ret-id" style="font-size:11px;">Stage 7 Retrieval Index</label>
            <select id="pb-ret-id" class="wb-select" title="1:N vector index constructed when Stage 7 Retrieval is enabled">
              ${retAlgos
                .map(
                  (a) =>
                    `<option value="${esc(a.algorithm_id)}" ${
                      a.algorithm_id === pipe.retrieval.algorithm_id ? "selected" : ""
                    }>${esc(a.name)}</option>`
                )
                .join("")}
            </select>
            <details style="margin-top:8px;">
              <summary style="cursor:pointer;font-size:11.5px;font-weight:600;color:var(--accent);">Stage 7 Index Parameters</summary>
              <div id="pb-ret-params" style="margin-top:8px;"></div>
            </details>
          </div>
        </div>
      </div>
    </div>

    <!-- Execution Graph & Two-Pipeline Snapshot Comparison -->
    <div class="wb-panel">
      <div class="wb-panel-header">
        <span>Live Stage Execution Graph &amp; Output Telemetry</span>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="pb-goto-cmp" type="button">Compare in Workspace 09 →</button>
      </div>
      <div class="wb-panel-body" id="pb-exec-graph-box">
        <div class="wb-help">Validate or execute the pipeline to inspect per-stage status, tensor shapes, face counts, and latency.</div>
      </div>
    </div>

    <!-- P-2: Post-Run Genuine / Impostor Score Distribution & Operating Point Explorer -->
    <div id="pb-p2-explorer-panel" class="wb-panel hidden">
      <div class="wb-panel-header"><span>Genuine / Impostor Score Distribution &amp; Operating Point (Completed Pipeline Run)</span></div>
      <div class="wb-panel-body" id="pb-p2-explorer-body"></div>
    </div>

    <div class="wb-panel">
      <div class="wb-panel-header"><span>Duplicated / Saved Pipeline Snapshots (Side-by-Side Diff)</span></div>
      <div class="wb-panel-body" id="pb-snapshots-box"></div>
    </div>
  `;

  let pbDeltaHandle = null;
  function refreshPipelineDeltaPanel() {
    pbDeltaHandle = renderConfigDeltaPanel(container.querySelector("#pb-config-delta-slot"), {
      algorithmsById: store.algorithmsById,
      expState: store.exp,
      scope: "pipeline",
      title: "Active Configuration Delta (Pipeline Builder Stages vs. Frozen Registry Defaults)",
      onResetConfirmed: () => {
        store.resetAllActiveStagesToDefaults();
        showToast("Reset pipeline and active stages to frozen registry defaults.");
        renderPipelineBuilder(container, navigateTo, waitJobFn);
      },
    });
  }

  function mountStageEditors() {
    renderParameterEditor(
      container.querySelector("#pb-det-params"),
      store.algorithmsById[pipe.detection.algorithm_id],
      pipe.detection.params,
      () => refreshPipelineDeltaPanel(),
      { showHeader: false }
    );
    renderParameterEditor(
      container.querySelector("#pb-align-params"),
      store.algorithmsById["five_point_canonical_aligner"],
      pipe.alignment.params,
      () => refreshPipelineDeltaPanel(),
      { showHeader: false }
    );
    renderParameterEditor(
      container.querySelector("#pb-qual-params"),
      store.algorithmsById["hybrid_classical_and_norm_quality"],
      pipe.quality.params,
      () => refreshPipelineDeltaPanel(),
      { showHeader: false }
    );
    renderParameterEditor(
      container.querySelector("#pb-rec-params"),
      store.algorithmsById[pipe.recognition.algorithm_id],
      pipe.recognition.params,
      () => refreshPipelineDeltaPanel(),
      { showHeader: false }
    );
    renderParameterEditor(
      container.querySelector("#pb-ret-params"),
      store.algorithmsById[pipe.retrieval.algorithm_id],
      pipe.retrieval.params,
      () => refreshPipelineDeltaPanel(),
      { showHeader: false }
    );
    refreshPipelineDeltaPanel();
  }

  function syncTopInputs() {
    pipe.name = container.querySelector("#pb-name").value.trim() || "Pipeline Run";
    pipe.dataset_id = container.querySelector("#pb-dataset").value;
    pipe.pair_limit = parseInt(container.querySelector("#pb-pair-limit").value, 10) || 120;
    pipe.detection.enable_landmarks = container.querySelector("#pb-lm-enable").checked;
    pipe.quality.enabled = container.querySelector("#pb-qual-enable").checked;
    pipe.recognition.enabled = container.querySelector("#pb-rec-enable").checked;
    pipe.recognition.strict_weights = container.querySelector("#pb-strict-weights").checked;
    pipe.normalization.enabled = container.querySelector("#pb-norm-enable").checked;
    pipe.matching.metric = container.querySelector("#pb-metric").value;
    pipe.retrieval.enabled = container.querySelector("#pb-ret-enable").checked;
    pipe.retrieval.algorithm_id = container.querySelector("#pb-ret-id").value;

    const orderMode = container.querySelector("#pb-order-mode").value;
    pipe.stage_order =
      orderMode === "post_rec_quality"
        ? ["image_input", "face_detection", "alignment", "face_recognition", "quality_assessment", "embedding_normalization", "matching_retrieval", "evaluation"]
        : ["image_input", "face_detection", "alignment", "quality_assessment", "face_recognition", "embedding_normalization", "matching_retrieval", "evaluation"];
  }

  async function runValidation() {
    syncTopInputs();
    const res = await api.validatePipeline(pipe);
    const slot = container.querySelector("#pb-validation-banner");
    if (res.valid) {
      slot.innerHTML = `<div class="wb-banner wb-banner-match">✓ Pipeline configuration is compatible.${
        res.warnings.length ? ` Warnings: ${esc(res.warnings.join(" | "))}` : ""
      }</div>`;
    } else {
      slot.innerHTML = `<div class="wb-banner wb-banner-nonmatch">✕ Incompatible Pipeline: ${esc(res.errors.join(" | "))}</div>`;
    }
    return res;
  }

  function renderStageGraph(res) {
    if (!res) return;
    const nodesHtml = (res.stage_graph || [])
      .map(
        (st) => `
        <div class="wb-stage-node ${st.status === "completed" ? "completed" : st.status === "skipped" ? "skipped" : "blocked"}">
          <div style="font-weight:700;font-size:11.5px;">${esc(st.stage_id)}</div>
          <div class="wb-mono wb-muted-xs">${esc(st.algorithm_id || st.status.toUpperCase())}</div>
          <div class="wb-mono" style="margin-top:4px;font-size:11px;">Time: <strong>${st.time_ms ?? 0} ms</strong></div>
          ${st.input_shape ? `<div class="wb-muted-xs wb-mono">In: ${esc(st.input_shape)}</div>` : ""}
          ${st.output_shape ? `<div class="wb-muted-xs wb-mono">Out: ${esc(st.output_shape)}</div>` : ""}
          ${st.face_count !== undefined ? `<div class="wb-muted-xs wb-mono">Items: ${st.face_count}</div>` : ""}
        </div>`
      )
      .join(`<span style="font-weight:700;color:var(--text-muted);">→</span>`);

    const mEntries = Object.entries(res.metrics || {})
      .map(([k, v]) => `<span class="wb-pill wb-mono">${esc(k)}: <strong>${v ?? "N/A"}</strong></span>`)
      .join(" ");

    container.querySelector("#pb-exec-graph-box").innerHTML = `
      <div style="margin-bottom:10px;display:flex;flex-wrap:wrap;gap:6px;">${mEntries}</div>
      <div class="wb-stage-flow">${nodesHtml}</div>
    `;

    const p2Panel = container.querySelector("#pb-p2-explorer-panel");
    const p2Body = container.querySelector("#pb-p2-explorer-body");
    if (res.curves?.score_distributions && p2Panel && p2Body) {
      p2Panel.classList.remove("hidden");
      renderScoreDistributionExplorer(p2Body, {
        experimentId: res.curves.score_distributions.experiment_id || "LIVE-PIPELINE-RUN",
        experimentName: pipe.name,
        scoreDistributions: res.curves.score_distributions,
        sanitizeNumericFn: sanitizeNumericInput,
        onToast: showToast,
        onInspectPair: ({ pair, threshold, metric, recognizerId, detectorId }) => {
          store.exp.imageARef = pair.image_a;
          store.exp.imageBRef = pair.image_b;
          store.exp.faceIndexA = 0;
          store.exp.faceIndexB = 0;
          if (threshold !== undefined) store.exp.threshold = threshold;
          if (metric) store.exp.matchingMetric = metric;
          if (recognizerId) store.exp.recognizerId = recognizerId;
          if (detectorId) store.exp.detectorId = detectorId;
          navigateTo("recognition_lab");
        },
      });
    } else if (p2Panel) {
      p2Panel.classList.add("hidden");
    }
  }

  function renderSnapshots() {
    const snaps = store.exp.savedPipelines;
    const box = container.querySelector("#pb-snapshots-box");
    if (!snaps.length) {
      box.innerHTML = `<div class="wb-help">Click "Duplicate Snapshot" to store multiple pipeline variants and compare their stage configurations side by side.</div>`;
      return;
    }
    box.innerHTML = `
      <div class="wb-table-wrap">
        <table class="wb-table">
          <thead><tr><th>Snapshot Name</th><th>Dataset</th><th>Detector</th><th>Recognizer</th><th>L2 Norm</th><th>Action</th></tr></thead>
          <tbody>
            ${snaps
              .map(
                (s, idx) => `<tr>
                  <td><strong>${esc(s.name)}</strong></td>
                  <td class="wb-mono">${esc(s.dataset_id)}</td>
                  <td class="wb-mono">${esc(s.detection.algorithm_id)}</td>
                  <td class="wb-mono">${esc(s.recognition.algorithm_id)} (${esc(s.recognition.params.precision || "auto")}, B=${
                  s.recognition.params.batch_size || 1
                })</td>
                  <td class="wb-mono">${s.normalization.enabled ? "True" : "False"}</td>
                  <td><button class="wb-btn wb-btn-secondary wb-btn-sm" data-load-snap="${idx}" type="button">Load Into Builder</button></td>
                </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
    box.querySelectorAll("[data-load-snap]").forEach((b) => {
      b.addEventListener("click", () => {
        const snap = JSON.parse(JSON.stringify(snaps[Number(b.getAttribute("data-load-snap"))]));
        store.exp.pipelineA = snap;
        renderPipelineBuilder(container, navigateTo, waitJobFn);
      });
    });
  }

  container.querySelector("#pb-det-id").addEventListener("change", (e) => {
    pipe.detection.algorithm_id = e.target.value;
    pipe.detection.params = { ...(store.exp.detectorParamsByModel[e.target.value] || {}) };
    mountStageEditors();
    store.notify("config_change");
  });
  container.querySelector("#pb-rec-id").addEventListener("change", (e) => {
    pipe.recognition.algorithm_id = e.target.value;
    pipe.recognition.params = { ...(store.exp.recognizerParamsByModel[e.target.value] || {}) };
    mountStageEditors();
    store.notify("config_change");
  });
  container.querySelector("#pb-ret-id").addEventListener("change", (e) => {
    pipe.retrieval.algorithm_id = e.target.value;
    pipe.retrieval.params = { ...(store.exp.retrievalParamsByIndex[e.target.value] || {}) };
    mountStageEditors();
    store.notify("config_change");
  });
  ["#pb-qual-enable", "#pb-rec-enable", "#pb-norm-enable", "#pb-ret-enable"].forEach((sel) => {
    container.querySelector(sel)?.addEventListener("change", () => {
      syncTopInputs();
      refreshPipelineDeltaPanel();
      store.notify("config_change");
    });
  });

  container.querySelector("#pb-validate-btn").addEventListener("click", () => runValidation());

  container.querySelector("#pb-dup-btn").addEventListener("click", () => {
    syncTopInputs();
    const clone = JSON.parse(JSON.stringify(pipe));
    clone.name = `${pipe.name} (Copy #${store.exp.savedPipelines.length + 1})`;
    store.exp.savedPipelines.push(clone);
    renderSnapshots();
    showToast(`Duplicated pipeline snapshot "${clone.name}"`);
  });

  container.querySelector("#pb-pair-limit").addEventListener("change", (e) => {
    const check = sanitizeNumericInput(e.target.value, { min: 20, max: 600, isInt: true, fallback: pipe.pair_limit });
    if (check.invalid || check.clamped) {
      showToast(`LFW Pair Cap: ${check.reason}`, true);
    }
    pipe.pair_limit = check.value;
    e.target.value = String(check.value);
  });

  const pbRunBtn = container.querySelector("#pb-run-btn");
  const pbSaveCfgBtn = container.querySelector("#pb-save-cfg-btn");

  pbSaveCfgBtn.addEventListener("click", () =>
    withBusyButton(pbSaveCfgBtn, async () => {
      syncTopInputs();
      const saved = await api.saveNamedConfig({
        name: pipe.name,
        workspace: "pipeline",
        config: pipe,
        description: "Saved from Pipeline Experiment Builder",
      });
      const expResp = await api.listExperiments(true);
      store.experiments = expResp.experiments;
      showToast(`Saved reproducible pipeline configuration ${saved.experiment_id}`);
    }, "Saving…")
  );

  pbRunBtn.addEventListener("click", () =>
    withBusyButton(pbRunBtn, async () => {
      const val = await runValidation();
      if (!val.valid) {
        showToast("Fix validation errors before running pipeline.", true);
        return;
      }
      const sub = await api.submitJob({
        job_type: "pipeline_run",
        name: pipe.name,
        config: pipe,
        save_experiment: true,
      });
      showToast(`Submitted pipeline job ${sub.job_id}`);
      const done = await waitJobFn(sub.job_id);
      if (done?.result && container.querySelector("#pb-exec-graph-box")) {
        if (done.experiment_id && done.result.curves?.score_distributions) {
          done.result.curves.score_distributions.experiment_id = done.experiment_id;
        }
        store.exp.lastPipelineResult = done.result;
        renderStageGraph(done.result);
      }
    }, "Executing…")
  );

  container.querySelector("#pb-goto-cmp").addEventListener("click", () => navigateTo("comparison"));

  mountStageEditors();
  renderSnapshots();
  if (store.exp.lastPipelineResult) renderStageGraph(store.exp.lastPipelineResult);
}
