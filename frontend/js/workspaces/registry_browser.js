/**
 * Workspace 12 — Model, Algorithm & Parameter Schema Registry (`frontend/js/workspaces/registry_browser.js`).
 *
 * Displays complete 5-layer provenance, SHA-256 verification hashes, input/output contracts,
 * parameter schemas, and live On-Demand Model Loading / VRAM Release controls.
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

export async function renderRegistryBrowser(container) {
  const sys = await api.getSystemStatus();
  if (!container.isConnected) return;
  store.systemStatus = sys;
  const loaded = sys.hardware_and_env?.loaded_models || [];
  const algos = store.registry?.algorithms || [];
  let activeId = algos[0]?.algorithm_id || null;

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>12 · Model, Algorithm &amp; Parameter Schema Registry</h2>
        <p class="wb-workspace-desc">
          Authoritative registry metadata driving dynamic parameter generation across the Workbench,
          paired with on-demand GPU/CPU model loading and explicit VRAM release controls.
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="reg-unload-all-btn" type="button"
          data-tip-title="Release All Cached Models &amp; Flush VRAM"
          data-tip-body="Evicts all loaded detector and recognizer sessions from memory and runs gc.collect() + torch.cuda.empty_cache().">Release All Loaded Models (${loaded.length})</button>
      </div>
    </div>

    <div class="wb-split-layout">
      <div class="wb-panel">
        <div class="wb-panel-header"><span>Registered Algorithms (${algos.length})</span></div>
        <div class="wb-panel-body" style="padding:0;">
          <table class="wb-table">
            <thead><tr><th>Category</th><th>Algorithm</th><th>Status</th></tr></thead>
            <tbody>
              ${algos
                .map(
                  (a) => `<tr data-reg-id="${esc(a.algorithm_id)}" class="${
                    a.algorithm_id === activeId ? "selected-row" : ""
                  }" style="cursor:pointer;">
                    <td><span class="wb-badge">${esc(a.category)}</span></td>
                    <td><strong>${esc(a.name)}</strong><div class="wb-muted-xs wb-mono">${esc(a.algorithm_id)}</div></td>
                    <td><span class="wb-badge ${a.availability_status === "AVAILABLE" ? "wb-badge-perf" : ""}">${esc(
                    a.availability_status
                  )}</span></td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>

      <div class="wb-panel">
        <div class="wb-panel-header">
          <span>Algorithm Contract, Provenance &amp; Parameter Schema</span>
          <div id="reg-model-actions" class="wb-toolbar-actions"></div>
        </div>
        <div class="wb-panel-body" id="reg-detail-box"></div>
      </div>
    </div>
  `;

  function renderAlgoDetail(algoId) {
    const a = store.algorithmsById[algoId];
    if (!a) return;

    const actBox = container.querySelector("#reg-model-actions");
    if ((a.category === "detection" || a.category === "recognition") && a.availability_status === "AVAILABLE") {
      actBox.innerHTML = `
        <button class="wb-btn wb-btn-sm" id="reg-load-default" type="button"
          data-tip-title="Load Model Into Active Session Cache"
          data-tip-body="Instantiates ${esc(a.name)} on the default execution provider and reports initialization time and VRAM delta.">Load Model On-Demand</button>
      `;
      const loadBtn = container.querySelector("#reg-load-default");
      loadBtn.addEventListener("click", () =>
        withBusyButton(
          loadBtn,
          async () => {
            try {
              const res = await api.loadModel({ model_id: a.algorithm_id });
              store.systemStatus = await api.getSystemStatus();
              showToast(`Loaded ${a.algorithm_id} in ${res.entry.init_time_ms} ms (${res.entry.device})`);
              if (container.isConnected) await renderRegistryBrowser(container);
            } catch (err) {
              showToast(err.message, true);
            }
          },
          "Loading…"
        )
      );
    } else if (a.availability_status !== "AVAILABLE") {
      actBox.innerHTML = `
        <button class="wb-btn wb-btn-secondary wb-btn-sm" type="button" disabled
          title="Cannot load ${esc(a.algorithm_id)}: ${esc(a.availability_status)} (automatic internet downloads are disabled)."
          data-tip-title="Model Unavailable (${esc(a.availability_status)})"
          data-tip-body="Local checkpoint weights are missing from models/. Offline governance prohibits automatic remote downloads.">Load Blocked (${esc(
            a.availability_status
          )})</button>
      `;
    } else {
      actBox.innerHTML = `<span class="wb-muted-xs" data-tip-title="Stateless Algorithmic Stage" data-tip-body="This stage executes directly in NumPy/OpenCV/SciPy without persistent neural network weights.">Stateless Module (No GPU Weights)</span>`;
    }

    container.querySelector("#reg-detail-box").innerHTML = `
      <p style="margin-top:0;">${esc(a.description)}</p>
      <table class="wb-table" style="margin-bottom:12px;">
        <tbody>
          <tr><th>Algorithm Identifier</th><td class="wb-mono">${esc(a.algorithm_id)}</td></tr>
          <tr><th>Paper / Reference</th><td>${esc(a.paper_reference)}</td></tr>
          <tr><th>Python Implementation</th><td class="wb-mono">${esc(a.source_implementation)}</td></tr>
          <tr><th>Checkpoint &amp; SHA-256</th><td class="wb-mono" style="word-break:break-all;">${esc(
            a.checkpoint_path || "N/A (Algorithmic Module)"
          )}${a.model_sha256 ? `<br/>SHA-256: ${esc(a.model_sha256)}` : ""}</td></tr>
          <tr><th>Supported Runtimes / Devices / Prec</th><td class="wb-mono">Runtimes: ${esc(
            a.supported_runtimes.join(", ")
          )} · Devices: ${esc(a.supported_devices.join(", "))} · Precision: ${esc(
      a.supported_precisions.join(", ")
    )} · Dynamic Batch: ${a.dynamic_batch_supported}</td></tr>
          <tr><th>Input Contract</th><td class="wb-mono" style="font-size:11px;">${esc(JSON.stringify(a.input_requirements))}</td></tr>
          <tr><th>Output Contract</th><td class="wb-mono" style="font-size:11px;">${esc(JSON.stringify(a.output_schema))}</td></tr>
          <tr><th>Licensing &amp; Provenance</th><td class="wb-mono" style="font-size:11px;">${esc(
            JSON.stringify(a.license_provenance)
          )}</td></tr>
        </tbody>
      </table>

      <div style="font-weight:700;margin-bottom:6px;">Registered Dynamic Parameter Schema (${a.parameters.length})</div>
      ${
        a.parameters.length === 0
          ? `<div class="wb-help">No configurable parameters exposed.</div>`
          : `<div class="wb-table-wrap">
              <table class="wb-table">
                <thead><tr><th>Parameter</th><th>Control / Type</th><th>Default</th><th>Constraints</th><th>Impact</th><th>Description</th></tr></thead>
                <tbody>
                  ${a.parameters
                    .map(
                      (p) => `<tr>
                        <td class="wb-mono"><strong>${esc(p.name)}</strong></td>
                        <td class="wb-mono">${esc(p.control_type)} (${esc(p.value_type)})</td>
                        <td class="wb-mono">${esc(p.default)}</td>
                        <td class="wb-mono" style="font-size:10.5px;">${
                          p.min_value !== null ? `[${p.min_value}..${p.max_value}]` : p.options ? `${p.options.length} opts` : "—"
                        }</td>
                        <td><span class="wb-badge">${esc(p.impact)}</span></td>
                        <td style="font-size:11px;">${esc(p.description)}</td>
                      </tr>`
                    )
                    .join("")}
                </tbody>
              </table>
            </div>`
      }
    `;
  }

  container.querySelectorAll("[data-reg-id]").forEach((tr) => {
    tr.addEventListener("click", () => {
      activeId = tr.getAttribute("data-reg-id");
      container.querySelectorAll("[data-reg-id]").forEach((r) => {
        r.classList.toggle("selected-row", r.getAttribute("data-reg-id") === activeId);
      });
      renderAlgoDetail(activeId);
    });
  });

  const unloadBtn = container.querySelector("#reg-unload-all-btn");
  unloadBtn.addEventListener("click", () =>
    withBusyButton(
      unloadBtn,
      async () => {
        const r = await api.unloadModels({ unload_all: true });
        showToast(`Released ${r.unloaded_keys.length} model session(s) and flushed CUDA cache.`);
        if (container.isConnected) await renderRegistryBrowser(container);
      },
      "Releasing…"
    )
  );

  if (activeId) renderAlgoDetail(activeId);
}
