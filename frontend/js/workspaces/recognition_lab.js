/**
 * Workspace 6 — Recognition & Matching Laboratory (`frontend/js/workspaces/recognition_lab.js`).
 *
 * Implements:
 *  - 1:1 Verification with multi-face image selection & synchronized zoom/pan
 *  - Instant zero-reinference threshold & metric updates when embeddings are cached
 *  - Side-by-side crops, interactive alpha-blend slider, pixel difference heatmap, and
 *    5-point anatomical landmark correspondence SVG (never fabricating local descriptor matches)
 *  - Integrated 1:N Gallery Probe Identification with click-to-inspect candidate into Image B
 */

import { api } from "../api.js";
import { renderSimilarityGauge } from "../components/charts.js";
import { ScientificImageViewer } from "../components/image_viewer.js";
import { renderParameterEditor } from "../components/param_editor.js";
import { showToast, store } from "../state.js";
import { helpBadgeHtml, sanitizeNumericInput, withBusyButton } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderRecognitionLab(container) {
  const recognizers = (store.registry?.algorithms || []).filter((a) => a.category === "recognition");
  const indexes = (store.registry?.algorithms || []).filter((a) => a.category === "retrieval");
  let syncZoom = true;
  const candidateImages = Array.from(
    new Set(
      [
        store.exp.imageARef,
        store.exp.imageBRef,
        ...store.exp.explorerImages,
        "datasets/evaluation/verification_pairs/ver_armstrong_official.jpg",
        "datasets/evaluation/verification_pairs/ver_armstrong_pose.jpg",
        "datasets/evaluation/verification_pairs/ver_aldrin_portrait.jpg",
      ].filter(Boolean)
    )
  );

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>06 · Face Recognition, 1:1 Verification &amp; 1:N Probe Identification Lab</h2>
        <p class="wb-workspace-desc">
          Compare AdaFace IR-50, MobileFaceNet W600K, and SFace. Adjusting the decision threshold updates
          Match / Non-Match decisions instantaneously without re-running embedding inference.
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <label style="display:inline-flex;align-items:center;gap:5px;font-size:12px;" title="Synchronize zoom and pan across Image A and Image B viewports">
          <input type="checkbox" id="rec-sync-zoom" checked /> Sync A↔B Zoom/Pan
        </label>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="rec-save-btn" type="button" title="Save this 1:1 verification run with SHA-256 provenance into Experiment History">Save 1:1 Run</button>
        <button class="wb-btn wb-btn-sm" id="rec-run-btn" type="button" title="Detect, align, extract embeddings, and compute similarity between Selected Face A and Face B">▶ Extract &amp; Verify 1:1</button>
      </div>
    </div>

    <div class="wb-split-layout">
      <!-- Left: Model, Preprocessing Lock, Metric & Instant Threshold -->
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Pair &amp; Recognition Model ${helpBadgeHtml("embedding")}</span></div>
          <div class="wb-panel-body">
            <div class="wb-field">
              <label class="wb-label" for="rec-img-a">Image A (Enrollment / Reference)</label>
              <select id="rec-img-a" class="wb-select" title="Reference/enrollment photograph (click any face box in Viewer A if multiple faces exist)">
                ${candidateImages
                  .map(
                    (r) =>
                      `<option value="${esc(r)}" ${r === store.exp.imageARef ? "selected" : ""}>${esc(
                        r.split("/").pop()
                      )}</option>`
                  )
                  .join("")}
              </select>
            </div>
            <div class="wb-field">
              <label class="wb-label" for="rec-img-b">Image B (Probe / Query)</label>
              <select id="rec-img-b" class="wb-select" title="Probe/comparison photograph (click any face box in Viewer B if multiple faces exist)">
                ${candidateImages
                  .map(
                    (r) =>
                      `<option value="${esc(r)}" ${r === store.exp.imageBRef ? "selected" : ""}>${esc(
                        r.split("/").pop()
                      )}</option>`
                  )
                  .join("")}
              </select>
            </div>
            <div class="wb-field">
              <label class="wb-label" for="rec-det-select">Detector / Landmark Source</label>
              <select id="rec-det-select" class="wb-select" title="Detector providing bounding boxes and 5-point landmarks for crop alignment">
                <option value="yunet_2023mar" ${store.exp.detectorId === "yunet_2023mar" ? "selected" : ""}>YuNet (2023mar)</option>
                <option value="scrfd_500m_research" ${store.exp.detectorId === "scrfd_500m_research" ? "selected" : ""}>SCRFD-500M</option>
              </select>
            </div>
            <div class="wb-field" style="margin-bottom:0;" data-tip-key="embedding">
              <label class="wb-label" for="rec-model-select">Recognition Embedding Model</label>
              <select id="rec-model-select" class="wb-select" data-tip-key="embedding">
                ${recognizers
                  .map(
                    (m) =>
                      `<option value="${esc(m.algorithm_id)}" ${
                        m.algorithm_id === store.exp.recognizerId ? "selected" : ""
                      }>${esc(m.name)}</option>`
                  )
                  .join("")}
              </select>
            </div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Model-Specific Controls</span></div>
          <div class="wb-panel-body" id="rec-param-editor"></div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Instant Threshold &amp; Metric (Zero Re-Inference)</span></div>
          <div class="wb-panel-body">
            <div class="wb-field" data-tip-key="cosine">
              <label class="wb-label" for="rec-metric-select">Comparison Metric ${helpBadgeHtml("cosine")} ${helpBadgeHtml("normalized_l2")}</label>
              <select id="rec-metric-select" class="wb-select" data-tip-key="cosine">
                <option value="cosine" ${store.exp.matchingMetric === "cosine" ? "selected" : ""}>Cosine Similarity &lt;u, v&gt;</option>
                <option value="normalized_l2" ${store.exp.matchingMetric === "normalized_l2" ? "selected" : ""}>Negated Normalized L2 (-||u - v||_2)</option>
                <option value="unnormalized_l2" ${store.exp.matchingMetric === "unnormalized_l2" ? "selected" : ""}>Negated Unnormalized L2 (-||z_a - z_b||_2)</option>
                <option value="normalized_l1" ${store.exp.matchingMetric === "normalized_l1" ? "selected" : ""}>Negated Normalized L1 (-||u - v||_1)</option>
              </select>
            </div>
            <div class="wb-field" data-tip-key="decision_threshold">
              <div class="wb-field-top">
                <label class="wb-label" for="rec-th-slider">Decision Threshold (τ) ${helpBadgeHtml("decision_threshold")}</label>
                <span class="wb-mono" id="rec-th-readout">${store.exp.threshold.toFixed(4)}</span>
              </div>
              <div class="wb-slider-row">
                <input type="range" id="rec-th-slider" min="-1.80" max="0.95" step="0.002" value="${store.exp.threshold}" data-tip-key="decision_threshold" />
                <input type="number" id="rec-th-num" aria-label="Decision Threshold numeric value" class="wb-input wb-mono" min="-2.00" max="1.00" step="0.002" value="${store.exp.threshold}" data-tip-key="decision_threshold" />
              </div>
              <div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;" id="rec-preset-th-btns"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Dual Image A/B Viewers, Decision Banner, Visual Crop Comparison & 1:N Search -->
      <div>
        <div id="rec-warn-slot"></div>
        <div id="rec-decision-slot" style="margin-bottom:10px;"></div>
        <div id="rec-gauge-slot" style="margin-bottom:12px;"></div>

        <div class="wb-grid-2" style="margin-bottom:12px;">
          <div id="rec-viewer-a"></div>
          <div id="rec-viewer-b"></div>
        </div>

        <!-- Detailed Crop Visual Comparison (Side-by-Side, Alpha Blend, Diff Heatmap, Landmark Links) -->
        <div class="wb-panel">
          <div class="wb-panel-header">
            <span>Aligned Crop Comparison, Alpha-Blend Overlay &amp; 5-Point Landmark Correspondences</span>
            <div style="display:flex;align-items:center;gap:6px;font-size:11.5px;" data-tip-key="alpha_blend">
              <label for="rec-alpha-slider">Alpha Blend (A↔B):</label>
              <input type="range" id="rec-alpha-slider" min="0" max="100" value="50" data-tip-key="alpha_blend" style="width:110px;" />
              <span class="wb-mono" id="rec-alpha-lbl">50%</span>
              ${helpBadgeHtml("alpha_blend")}
            </div>
          </div>
          <div class="wb-panel-body" id="rec-crop-comp-body"></div>
        </div>

        <!-- 1:N Probe Identification Section -->
        <div class="wb-panel">
          <div class="wb-panel-header">
            <span>1:N Gallery Probe Identification (Probe = Selected Face in Image A) ${helpBadgeHtml("recall_at_k")}</span>
            <button class="wb-btn wb-btn-sm" id="rec-1ton-btn" type="button" title="Search Selected Face A against the chosen gallery and rank Top-K templates">▶ Search Gallery (Top-K)</button>
          </div>
          <div class="wb-panel-body">
            <div class="wb-grid-4" style="margin-bottom:10px;">
              <div>
                <label class="wb-label" for="idn-gallery">Gallery Dataset</label>
                <select id="idn-gallery" class="wb-select" title="Enrolled research template gallery to search against">
                  <option value="nasa_p11_verification">NASA P1.1 Astronaut Gallery (16 Templates)</option>
                  <option value="lfw_sample">LFW Held-Out Sample Gallery (80 Templates)</option>
                  <option value="custom_refs">Current Working Image Set</option>
                </select>
              </div>
              <div>
                <label class="wb-label" for="idn-index">Retrieval Index</label>
                <select id="idn-index" class="wb-select" title="Exact brute-force or Approximate Nearest Neighbor (ANN) index">
                  ${indexes.map((ix) => `<option value="${esc(ix.algorithm_id)}">${esc(ix.name)}</option>`).join("")}
                </select>
              </div>
              <div>
                <label class="wb-label" for="idn-topk">Top-K Candidates</label>
                <input type="number" id="idn-topk" class="wb-input wb-mono" min="1" max="25" value="8" title="Number of highest-scoring gallery candidates to return [1..25]" />
              </div>
              <div>
                <label class="wb-label" for="idn-cap">Gallery Cap</label>
                <input type="number" id="idn-cap" class="wb-input wb-mono" min="4" max="300" value="80" title="Maximum gallery templates enrolled for interactive search [4..300]" />
              </div>
            </div>
            <div id="idn-results-box">
              <div class="wb-help">Click "Search Gallery (Top-K)" to retrieve ranked templates. Click any returned candidate to inspect it in Image B.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  let viewerA = null;
  let viewerB = null;

  viewerA = new ScientificImageViewer(container.querySelector("#rec-viewer-a"), {
    title: "Image A (Click any face to select Face A)",
    heightPx: 340,
    onSelectFace: (idx) => {
      store.exp.faceIndexA = idx;
      executeVerification();
    },
    onTransformChange: (t) => {
      if (syncZoom && viewerB) viewerB.applySyncedTransform(t);
    },
  });

  viewerB = new ScientificImageViewer(container.querySelector("#rec-viewer-b"), {
    title: "Image B (Click any face to select Face B)",
    heightPx: 340,
    onSelectFace: (idx) => {
      store.exp.faceIndexB = idx;
      executeVerification();
    },
    onTransformChange: (t) => {
      if (syncZoom && viewerA) viewerA.applySyncedTransform(t);
    },
  });

  function mountRecParams() {
    const algo = store.algorithmsById[store.exp.recognizerId];
    const vals = store.exp.recognizerParamsByModel[store.exp.recognizerId] || {};
    renderParameterEditor(container.querySelector("#rec-param-editor"), algo, vals, () => {});
  }

  async function executeVerification({ saveRun = false } = {}) {
    try {
      // Ensure Image A/B are in dropdowns if loaded from pair explorer / 1:N click
      for (const [selId, refVal] of [
        ["#rec-img-a", store.exp.imageARef],
        ["#rec-img-b", store.exp.imageBRef],
      ]) {
        const sel = container.querySelector(selId);
        if (!Array.from(sel.options).some((o) => o.value === refVal)) {
          const opt = document.createElement("option");
          opt.value = refVal;
          opt.textContent = refVal.split("/").pop();
          sel.appendChild(opt);
        }
        sel.value = refVal;
      }

      const recId = store.exp.recognizerId;
      const res = await api.runVerification({
        image_a_ref: store.exp.imageARef,
        image_b_ref: store.exp.imageBRef,
        face_index_a: store.exp.faceIndexA,
        face_index_b: store.exp.faceIndexB,
        detector_id: store.exp.detectorId,
        alignment_params: store.exp.alignmentParams,
        recognizer_id: recId,
        recognizer_params: store.exp.recognizerParamsByModel[recId] || {},
        metric: store.exp.matchingMetric,
        threshold: store.exp.threshold,
        save_experiment: saveRun,
      });
      store.exp.lastVerificationResult = res;
      if (res.saved_experiment_id) {
        const expResp = await api.listExperiments(true);
        store.experiments = expResp.experiments;
        showToast(`Saved verification run ${res.saved_experiment_id}`);
      }

      if (!container.querySelector("#rec-decision-slot")) return;
      await viewerA.loadImageUrl(api.rawImageUrl(store.exp.imageARef), res.image_a.meta, true);
      if (!container.querySelector("#rec-decision-slot")) return;
      viewerA.setScene({
        detections: res.image_a.all_faces.map((f, i) =>
          i === res.image_a.selected_face_index ? { ...f, landmarks: res.image_a.face.landmarks } : f
        ),
        selectedFaceIndex: res.image_a.selected_face_index,
        alignmentQuadOrig: res.image_a.face.alignment.crop_boundary_quad_orig_px,
      });

      await viewerB.loadImageUrl(api.rawImageUrl(store.exp.imageBRef), res.image_b.meta, true);
      if (!container.querySelector("#rec-decision-slot")) return;
      viewerB.setScene({
        detections: res.image_b.all_faces.map((f, i) =>
          i === res.image_b.selected_face_index ? { ...f, landmarks: res.image_b.face.landmarks } : f
        ),
        selectedFaceIndex: res.image_b.selected_face_index,
        alignmentQuadOrig: res.image_b.face.alignment.crop_boundary_quad_orig_px,
      });

      renderCachedDecisionAndVisuals();
    } catch (err) {
      const decSlot = container.querySelector("#rec-decision-slot");
      if (!decSlot) return;
      showToast(`1:1 Verification error: ${err.message}`, true);
      decSlot.innerHTML = `<div class="wb-banner wb-banner-nonmatch">${esc(err.message)}</div>`;
    }
  }

  /**
   * Re-evaluates decision & operating-point display purely from cached embeddings/metrics
   * when threshold or metric changes — ZERO redundant inference!
   */
  function renderCachedDecisionAndVisuals() {
    const res = store.exp.lastVerificationResult;
    if (!res || !container.querySelector("#rec-decision-slot")) return;

    const metric = store.exp.matchingMetric;
    const tau = Number(store.exp.threshold);
    const score = Number(res.metrics[metric]);
    const isMatch = score >= tau;
    const margin = score - tau;

    container.querySelector("#rec-warn-slot").innerHTML = res.preprocessing_warning
      ? `<div class="wb-banner wb-banner-warn">⚠ ${esc(res.preprocessing_warning)}</div>`
      : "";

    const cal = res.decision.calibration_context || {};
    const ops = cal.operating_points || {};
    let opSummary = "Uncalibrated / manual threshold.";
    if (cal.calibrated_reference_available && metric === res.decision.metric) {
      const op1e3 = ops["target_far_0.001"];
      const op1e2 = ops["target_far_0.01"];
      opSummary = `${cal.calibration_source} · Test EER ${cal.test_eer_pct}% · AUC ${cal.test_roc_auc}${
        op1e3
          ? ` · @FAR≤1e-3 (τ=${op1e3.calibrated_threshold}): Realized TAR=${(op1e3.test_realized_tar * 100).toFixed(
              2
            )}%, Realized FAR=${(op1e3.test_realized_far * 100).toFixed(3)}%`
          : ""
      }${op1e2 ? ` · @FAR≤1e-2 (τ=${op1e2.calibrated_threshold})` : ""}`;

      // Preset calibrated threshold quick-apply buttons
      const pSlot = container.querySelector("#rec-preset-th-btns");
      pSlot.innerHTML = [
        cal.calibrated_acc_threshold !== undefined
          ? `<button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-set-th="${cal.calibrated_acc_threshold}">Calib Best Acc (${cal.calibrated_acc_threshold})</button>`
          : "",
        op1e3
          ? `<button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-set-th="${op1e3.calibrated_threshold}">FAR≤10⁻³ (${op1e3.calibrated_threshold})</button>`
          : "",
        op1e2
          ? `<button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-set-th="${op1e2.calibrated_threshold}">FAR≤10⁻² (${op1e2.calibrated_threshold})</button>`
          : "",
      ].join("");
      pSlot.querySelectorAll("[data-set-th]").forEach((b) => {
        b.addEventListener("click", () => {
          applyInstantThreshold(Number(b.getAttribute("data-set-th")));
        });
      });
    }

    container.querySelector("#rec-decision-slot").innerHTML = `
      <div class="wb-banner ${isMatch ? "wb-banner-match" : "wb-banner-nonmatch"}" style="margin-bottom:0;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
          <span style="font-size:14px;">${isMatch ? "✓ DECISION: MATCH" : "✕ DECISION: NON-MATCH"} (Instant Threshold Evaluation)</span>
          <span class="wb-mono">${esc(metric)} = ${score.toFixed(5)} · τ = ${tau.toFixed(5)} · Δ = ${
      margin >= 0 ? "+" : ""
    }${margin.toFixed(5)}</span>
        </div>
        <div style="font-weight:400;font-size:11px;margin-top:4px;opacity:0.92;">
          <strong>Score Semantics:</strong> Raw vector similarity/distance (NOT a calibrated match probability). ${esc(opSummary)}
        </div>
      </div>
    `;

    renderSimilarityGauge(container.querySelector("#rec-gauge-slot"), {
      metric,
      score,
      threshold: tau,
    });

    // Render Crops, Alpha-Blend, Diff Heatmap, Landmark Correspondences, and All Metrics Table
    const ca = res.image_a.face.alignment.crop_data_url;
    const cb = res.image_b.face.alignment.crop_data_url;
    const ptsA = res.image_a.face.alignment.transformed_landmarks_crop_px;
    const ptsB = res.image_b.face.alignment.transformed_landmarks_crop_px;
    const alphaPct = Number(container.querySelector("#rec-alpha-slider").value);

    const svgLines = ptsA
      .map((pa, k) => {
        const pb = ptsB[k];
        return `
          <circle cx="${pa[0]}" cy="${pa[1]}" r="3" fill="#0ea5e9" />
          <circle cx="${pb[0] + 132}" cy="${pb[1]}" r="3" fill="#f97316" />
          <line x1="${pa[0]}" y1="${pa[1]}" x2="${pb[0] + 132}" y2="${pb[1]}" stroke="#0d9488" stroke-width="1.4" stroke-dasharray="3,2" />
        `;
      })
      .join("");

    container.querySelector("#rec-crop-comp-body").innerHTML = `
      <div style="display:grid;grid-template-columns:auto auto auto 1fr;gap:16px;align-items:start;">
        <!-- SVG 5-Point Landmark Correspondence Pair -->
        <div>
          <div style="font-weight:600;font-size:11px;margin-bottom:4px;">5-Pt Landmark Correspondence</div>
          <svg width="244" height="112" viewBox="0 0 244 112" style="border:1px solid var(--border);border-radius:4px;background:var(--bg-canvas);">
            <image href="${ca}" x="0" y="0" width="112" height="112" />
            <image href="${cb}" x="132" y="0" width="112" height="112" />
            ${svgLines}
          </svg>
          <div class="wb-muted-xs" style="max-width:244px;">Anatomical 5-pt landmarks only (no fabricated local descriptor matches).</div>
        </div>

        <!-- Alpha Blend Overlay -->
        <div style="text-align:center;">
          <div style="font-weight:600;font-size:11px;margin-bottom:4px;">Alpha BlendOverlay</div>
          <div style="position:relative;width:112px;height:112px;border:1px solid var(--border-strong);border-radius:4px;overflow:hidden;">
            <img src="${ca}" style="position:absolute;left:0;top:0;width:112px;height:112px;" />
            <img id="rec-alpha-img-b" src="${cb}" style="position:absolute;left:0;top:0;width:112px;height:112px;opacity:${(
      alphaPct / 100
    ).toFixed(2)};" />
          </div>
          <div class="wb-muted-xs wb-mono">A + B Blend</div>
        </div>

        <!-- Pixel Difference Heatmap -->
        <div style="text-align:center;">
          <div style="font-weight:600;font-size:11px;margin-bottom:4px;">Pixel |A − B| Heatmap</div>
          <img src="${res.visual_comparison.crop_diff_heatmap_data_url}" style="width:112px;height:112px;border:1px solid var(--border-strong);border-radius:4px;" />
          <div class="wb-muted-xs wb-mono">MAE ${res.visual_comparison.crop_mean_abs_diff}</div>
        </div>

        <!-- All Metrics + Embedding & Quality Summary -->
        <div>
          <table class="wb-table">
            <thead><tr><th>Metric / Property</th><th>Value</th><th>Face A vs. Face B</th></tr></thead>
            <tbody>
              <tr><th>Cosine Similarity</th><td class="wb-mono"><strong>${res.metrics.cosine}</strong></td><td class="wb-mono">Dim ${res.recognizer_meta.embedding_dim} (${esc(res.recognizer_meta.precision)})</td></tr>
              <tr><th>Normalized L2 (Unit Dist)</th><td class="wb-mono">${res.metrics.normalized_l2} (||u−v||=${res.metrics.positive_unit_l2_distance})</td><td class="wb-mono">||u_A||=${res.image_a.face.unit_embedding_norm}, ||u_B||=${res.image_b.face.unit_embedding_norm}</td></tr>
              <tr><th>Unnormalized L2</th><td class="wb-mono">${res.metrics.unnormalized_l2} (||z_a−z_b||=${res.metrics.positive_raw_l2_distance})</td><td class="wb-mono">||z_A||=${res.image_a.face.raw_embedding_norm}, ||z_B||=${res.image_b.face.raw_embedding_norm}</td></tr>
              <tr><th>Normalized L1</th><td class="wb-mono">${res.metrics.normalized_l1}</td><td class="wb-mono">Rec Inf: ${res.stage_latency_ms.recognition_inference_ms} ms</td></tr>
              <tr><th>Laplacian Sharpness</th><td class="wb-mono">—</td><td class="wb-mono">A: ${res.image_a.face.quality_p1.sharpness_laplacian_var.toFixed(1)} · B: ${res.image_b.face.quality_p1.sharpness_laplacian_var.toFixed(1)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  function applyInstantThreshold(val) {
    if (Number.isNaN(val)) return;
    store.exp.threshold = val;
    container.querySelector("#rec-th-slider").value = val;
    container.querySelector("#rec-th-num").value = val;
    container.querySelector("#rec-th-readout").textContent = val.toFixed(4);
    renderCachedDecisionAndVisuals();
  }

  // Event listeners
  container.querySelector("#rec-sync-zoom").addEventListener("change", (e) => {
    syncZoom = e.target.checked;
  });
  container.querySelector("#rec-img-a").addEventListener("change", (e) => {
    store.exp.imageARef = e.target.value;
    store.exp.faceIndexA = 0;
    executeVerification();
  });
  container.querySelector("#rec-img-b").addEventListener("change", (e) => {
    store.exp.imageBRef = e.target.value;
    store.exp.faceIndexB = 0;
    executeVerification();
  });
  container.querySelector("#rec-det-select").addEventListener("change", (e) => {
    store.exp.detectorId = e.target.value;
    store.notify("config_change");
    executeVerification();
  });
  container.querySelector("#rec-model-select").addEventListener("change", (e) => {
    store.exp.recognizerId = e.target.value;
    mountRecParams();
    store.notify("config_change");
    executeVerification();
  });
  container.querySelector("#rec-metric-select").addEventListener("change", (e) => {
    store.exp.matchingMetric = e.target.value;
    if (e.target.value === "cosine" && store.exp.threshold < -0.5) {
      applyInstantThreshold(0.2169);
    } else if (e.target.value === "normalized_l2" && store.exp.threshold > 0) {
      applyInstantThreshold(-1.25);
    } else {
      renderCachedDecisionAndVisuals();
    }
  });

  container.querySelector("#rec-th-slider").addEventListener("input", (e) => applyInstantThreshold(parseFloat(e.target.value)));
  container.querySelector("#rec-th-num").addEventListener("change", (e) => {
    const check = sanitizeNumericInput(e.target.value, { min: -2.0, max: 1.0, isInt: false, fallback: store.exp.threshold });
    if (check.invalid || check.clamped) {
      showToast(`Decision Threshold (τ): ${check.reason}`, true);
    }
    applyInstantThreshold(check.value);
  });

  container.querySelector("#idn-topk").addEventListener("change", (e) => {
    const check = sanitizeNumericInput(e.target.value, { min: 1, max: 25, isInt: true, fallback: 8 });
    if (check.invalid || check.clamped) showToast(`Top-K Candidates: ${check.reason}`, true);
    e.target.value = String(check.value);
  });
  container.querySelector("#idn-cap").addEventListener("change", (e) => {
    const check = sanitizeNumericInput(e.target.value, { min: 4, max: 300, isInt: true, fallback: 80 });
    if (check.invalid || check.clamped) showToast(`Gallery Cap: ${check.reason}`, true);
    e.target.value = String(check.value);
  });

  container.querySelector("#rec-alpha-slider").addEventListener("input", (e) => {
    const v = Number(e.target.value);
    container.querySelector("#rec-alpha-lbl").textContent = `${v}%`;
    const imgB = container.querySelector("#rec-alpha-img-b");
    if (imgB) imgB.style.opacity = (v / 100).toFixed(2);
  });

  const recRunBtn = container.querySelector("#rec-run-btn");
  const recSaveBtn = container.querySelector("#rec-save-btn");
  const rec1toNBtn = container.querySelector("#rec-1ton-btn");
  recRunBtn.addEventListener("click", () => withBusyButton(recRunBtn, () => executeVerification(), "Verifying…"));
  recSaveBtn.addEventListener("click", () => withBusyButton(recSaveBtn, () => executeVerification({ saveRun: true }), "Saving…"));

  // 1:N Identification Search
  rec1toNBtn.addEventListener("click", () =>
    withBusyButton(rec1toNBtn, async () => {
      const box = container.querySelector("#idn-results-box");
      if (!box) return;
      box.innerHTML = `<div class="wb-help">Extracting / querying gallery templates…</div>`;
      try {
        const galSource = container.querySelector("#idn-gallery").value;
        const idxId = container.querySelector("#idn-index").value;
        const topK = sanitizeNumericInput(container.querySelector("#idn-topk").value, { min: 1, max: 25, isInt: true, fallback: 8 }).value;
        const cap = sanitizeNumericInput(container.querySelector("#idn-cap").value, { min: 4, max: 300, isInt: true, fallback: 80 }).value;
        const recId = store.exp.recognizerId;

        const idRes = await api.runIdentification({
          probe_image_ref: store.exp.imageARef,
          probe_face_index: store.exp.faceIndexA,
          gallery_source: galSource,
          gallery_limit: cap,
          custom_refs: galSource === "custom_refs" ? store.exp.explorerImages : null,
          detector_id: store.exp.detectorId,
          recognizer_id: recId,
          recognizer_params: store.exp.recognizerParamsByModel[recId] || {},
          index_id: idxId,
          index_params: store.exp.retrievalParamsByIndex[idxId] || {},
          top_k: topK,
          threshold: store.exp.threshold,
        });
        store.exp.lastIdentificationResult = idRes;
        if (!container.querySelector("#idn-results-box")) return;

      box.innerHTML = `
        <div class="wb-mono wb-muted-xs" style="margin-bottom:8px;">
          Gallery N=${idRes.gallery_summary.gallery_size} · Index=${esc(idRes.index_summary.index_id)} ·
          Build ${idRes.index_summary.build_time_ms} ms · Search <strong>${idRes.index_summary.search_latency_ms} ms</strong> ·
          Memory ${idRes.index_summary.memory_mib} MiB (Click any candidate card to inspect 1:1 in Image B)
        </div>
        <div class="wb-thumb-grid">
          ${idRes.candidates
            .map(
              (c) => `
              <div class="wb-thumb-card" data-cand-ref="${esc(c.image_ref)}" title="Click to inspect candidate vs Probe A">
                <img src="${c.crop_data_url}" alt="${esc(c.template_id)}" />
                <div class="wb-mono" style="font-size:10.5px;font-weight:700;">#${c.rank} (Exact #${c.exact_rank}) · cos ${c.exact_cosine.toFixed(
                4
              )}</div>
                <div class="wb-muted-xs wb-mono">${esc(c.template_id)}</div>
                <div class="wb-muted-xs">${esc(c.anonymized_template_tag)}</div>
                <span class="wb-badge ${c.above_threshold ? "wb-badge-perf" : ""}">${
                c.above_threshold ? "≥ Threshold" : "< Threshold"
              }</span>
              </div>`
            )
            .join("")}
        </div>
      `;

      box.querySelectorAll("[data-cand-ref]").forEach((card) => {
        card.addEventListener("click", () => {
          store.exp.imageBRef = card.getAttribute("data-cand-ref");
          store.exp.faceIndexB = 0;
          executeVerification();
        });
      });
      } catch (err) {
        if (box.isConnected) {
          box.innerHTML = `<div class="wb-banner wb-banner-nonmatch">${esc(err.message)}</div>`;
        }
      }
    }, "Searching…")
  );

  mountRecParams();
  await executeVerification();
}
