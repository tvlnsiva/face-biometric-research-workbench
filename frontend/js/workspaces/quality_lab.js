/**
 * Workspace 5 — Face Quality Assessment Laboratory (`frontend/js/workspaces/quality_lab.js`).
 *
 * Exposes all implemented P1 classical and P3 extended quality metrics + embedding norm `||z_raw||_2`,
 * controlled synthetic degradations, quality-based filtering/ranking, and multi-shot spherical pooling.
 */

import { api } from "../api.js";
import { renderBarChart, renderLineChart } from "../components/charts.js";
import { ScientificImageViewer } from "../components/image_viewer.js";
import { renderParameterEditor } from "../components/param_editor.js";
import { showToast, store } from "../state.js";
import { helpBadgeHtml, withBusyButton } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderQualityLab(container) {
  const qualSpec = store.algorithmsById["hybrid_classical_and_norm_quality"];
  const deg = store.exp.qualityDegradation;

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>05 · Face Quality Assessment Laboratory</h2>
        <p class="wb-workspace-desc">
          Inspect deterministic pixel measurements alongside deep embedding norm <code>||z_raw||_2</code>.
          Never invents unimplemented metrics (e.g., occlusion classifier explicitly disclosed as unimplemented).
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ql-reset-deg-btn" type="button" title="Reset blur, spatial resolution, gamma, and JPEG quality back to uncorrupted defaults">Reset Degradations</button>
        <button class="wb-btn wb-btn-sm" id="ql-run-btn" type="button" title="Run classical pixel quality + deep embedding norm assessment across working images">▶ Assess Working Image Set</button>
      </div>
    </div>

    <div class="wb-split-layout">
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Models &amp; Controlled Probe Degradations</span></div>
          <div class="wb-panel-body">
            <div class="wb-field" data-tip-key="learned_embedding_l2_norm">
              <label class="wb-label" for="ql-rec-select">Embedding Model (for ||z_raw||_2 Proxy) ${helpBadgeHtml("learned_embedding_l2_norm")}</label>
              <select id="ql-rec-select" class="wb-select" data-tip-key="learned_embedding_l2_norm">
                <option value="adaface_ir50_webface4m" ${store.exp.recognizerId === "adaface_ir50_webface4m" ? "selected" : ""}>AdaFace IR-50 WebFace4M (Quality-Adaptive Margin)</option>
                <option value="mobilefacenet_w600k_research" ${store.exp.recognizerId === "mobilefacenet_w600k_research" ? "selected" : ""}>MobileFaceNet W600K (ArcFace — Uncalibrated Norm)</option>
                <option value="sface_2021dec" ${store.exp.recognizerId === "sface_2021dec" ? "selected" : ""}>SFace 128-D</option>
              </select>
            </div>

            <div class="wb-field" data-tip-key="blur_sigma">
              <div class="wb-field-top"><label class="wb-label" for="ql-blur">Synthetic Gaussian Blur σ ${helpBadgeHtml("blur_sigma")}</label><span class="wb-mono" id="ql-blur-val">${deg.blur_sigma}</span></div>
              <input type="range" id="ql-blur" min="0" max="4.0" step="0.2" value="${deg.blur_sigma}" data-tip-key="blur_sigma" style="width:100%;" />
            </div>
            <div class="wb-field" data-tip-key="effective_res">
              <div class="wb-field-top"><label class="wb-label" for="ql-res">Effective Spatial Resolution (px) ${helpBadgeHtml("effective_res")}</label><span class="wb-mono" id="ql-res-val">${deg.effective_res}×${deg.effective_res}</span></div>
              <input type="range" id="ql-res" min="16" max="112" step="8" value="${deg.effective_res}" data-tip-key="effective_res" style="width:100%;" />
            </div>
            <div class="wb-field" data-tip-key="gamma">
              <div class="wb-field-top"><label class="wb-label" for="ql-gam">Illumination Gamma (γ) ${helpBadgeHtml("gamma")}</label><span class="wb-mono" id="ql-gam-val">${deg.gamma}</span></div>
              <input type="range" id="ql-gam" min="0.4" max="3.5" step="0.1" value="${deg.gamma}" data-tip-key="gamma" style="width:100%;" />
            </div>
            <div class="wb-field" style="margin-bottom:0;" data-tip-key="jpeg_quality">
              <div class="wb-field-top"><label class="wb-label" for="ql-jpg">JPEG Compression Quality ${helpBadgeHtml("jpeg_quality")}</label><span class="wb-mono" id="ql-jpg-val">${deg.jpeg_quality}</span></div>
              <input type="range" id="ql-jpg" min="10" max="100" step="5" value="${deg.jpeg_quality}" data-tip-key="jpeg_quality" style="width:100%;" />
            </div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Quality Thresholds &amp; EVRC Reject Control</span></div>
          <div class="wb-panel-body" id="ql-param-editor"></div>
        </div>
      </div>

      <div>
        <div id="ql-viewer-box" style="margin-bottom:12px;"></div>

        <div class="wb-grid-2" style="margin-bottom:12px;">
          <div class="wb-panel">
            <div class="wb-panel-header"><span>Quality Score Ranking &amp; Filtering</span></div>
            <div class="wb-panel-body" id="ql-rank-chart"></div>
          </div>
          <div class="wb-panel">
            <div class="wb-panel-header"><span>Quality vs. Self-Recognition Cosine / Pooling</span></div>
            <div class="wb-panel-body" id="ql-sim-chart"></div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header">
            <span>Per-Face Quality Measurements (Click Any Row to Inspect in Viewer)</span>
            <span class="wb-badge" id="ql-filter-badge">—</span>
          </div>
          <div class="wb-panel-body" id="ql-table-box"></div>
        </div>
      </div>
    </div>
  `;

  let selectedQualityIdx = 0;
  const qualViewer = new ScientificImageViewer(container.querySelector("#ql-viewer-box"), {
    title: "Selected Face Source Frame & Overlay Inspector",
    heightPx: 320,
    overlays: { gtBoxes: false, correspondenceLines: false },
  });

  renderParameterEditor(container.querySelector("#ql-param-editor"), qualSpec, store.exp.qualityParams, () => {
    executeQuality();
  });

  async function executeQuality() {
    const recSel = container.querySelector("#ql-rec-select");
    if (!recSel) return;
    try {
      const recId = recSel.value;
      const res = await api.runQuality({
        image_refs: store.exp.explorerImages.slice(0, 8),
        detector_id: store.exp.detectorId,
        recognizer_id: recId,
        quality_params: store.exp.qualityParams,
        degradation: store.exp.qualityDegradation,
      });
      store.exp.lastQualityResult = res;
      if (!container.querySelector("#ql-filter-badge")) return;
      renderQualityResults(res);
    } catch (err) {
      showToast(`Quality assessment failed: ${err.message}`, true);
    }
  }

  function renderQualityResults(res) {
    const fs = res.filter_summary || {};
    container.querySelector("#ql-filter-badge").textContent = `Retained ${fs.retained_faces ?? 0}/${
      fs.total_faces ?? 0
    } (${fs.metric || ""})`;

    const items = res.items || [];
    if (!items.length) {
      container.querySelector("#ql-table-box").innerHTML = `<div class="wb-help">No faces detected across selected working images.</div>`;
      return;
    }
    if (selectedQualityIdx >= items.length) selectedQualityIdx = 0;

    async function syncQualityViewerToRow(rowIdx) {
      const target = items[rowIdx];
      if (!target || !container.isConnected) return;
      const sameImgFaces = items.filter((x) => x.image_ref === target.image_ref);
      const localFaceIdx = Math.max(0, sameImgFaces.findIndex((x) => x.face_instance_id === target.face_instance_id));
      await qualViewer.loadImageUrl(api.rawImageUrl(target.image_ref), { format: target.face_instance_id });
      qualViewer.setScene({
        detections: sameImgFaces.map((f) => ({
          bounding_box: f.bounding_box,
          landmarks_points: f.landmarks_points || null,
          confidence: f.confidence,
        })),
        selectedFaceIndex: localFaceIdx,
      });
    }
    syncQualityViewerToRow(selectedQualityIdx);

    renderBarChart(container.querySelector("#ql-rank-chart"), {
      title: `Active Quality Metric: ${fs.metric}`,
      subtitle: fs.cutoff_value ? `EVRC Reject Cutoff = ${fs.cutoff_value}` : "No pairs rejected (0%)",
      categories: items.slice(0, 8).map((it) => it.face_instance_id),
      values: items.slice(0, 8).map((it) => it.active_metric_value),
    });

    const hasSelfCos = items.some((it) => it.clean_vs_degraded_self_cosine !== null);
    if (hasSelfCos) {
      renderLineChart(container.querySelector("#ql-sim-chart"), {
        title: `${fs.metric} vs. Clean↔Degraded Self Cosine`,
        subtitle: "Demonstrates how degradation lowers both quality and embedding similarity",
        xLabel: fs.metric,
        yLabel: "Cosine(clean, degraded)",
        series: [
          {
            name: "Face Instances",
            points: items.map((it) => ({
              x: Number(it.active_metric_value),
              y: Number(it.clean_vs_degraded_self_cosine),
              label: it.face_instance_id,
            })),
          },
        ],
      });
    } else if (res.template_aggregation_preview) {
      const tp = res.template_aggregation_preview;
      renderBarChart(container.querySelector("#ql-sim-chart"), {
        title: "Multi-Shot Spherical Template Pooling Weights (Softmax ||z_raw||_2)",
        subtitle: `Cos(norm_pool, uniform) = ${tp.template_cosine_norm_vs_uniform}`,
        categories: items.slice(0, 8).map((it) => it.face_instance_id),
        values: tp.learned_norm_weights.slice(0, 8),
      });
    }

    container.querySelector("#ql-table-box").innerHTML = `
      <div class="wb-banner" style="margin-bottom:8px;font-size:11.5px;">
        <strong>Measurement Semantics:</strong> Laplacian, Tenengrad, Exposure, Blockiness, Inter-Ocular px, and Yaw Asymmetry are
        <em>deterministic raw pixel/geometric measurements</em>. <code>||z_raw||_2</code> is a <em>model-dependent uncalibrated feature norm</em>
        (explicitly regularized only in AdaFace; per P3.1 audit, never assume universal cross-model calibration).
        Occlusion Estimator: <code>NOT_IMPLEMENTED</code>.
      </div>
      <div class="wb-table-wrap">
        <table class="wb-table">
          <thead>
            <tr>
              <th>Crop</th>
              <th>Face ID</th>
              <th>Category / Filter</th>
              <th data-tip-key="learned_embedding_l2_norm">||z_raw||_2 ${helpBadgeHtml("learned_embedding_l2_norm")}</th>
              <th data-tip-key="sharpness_laplacian_var">Laplacian / Tenengrad ${helpBadgeHtml("sharpness_laplacian_var")} ${helpBadgeHtml("sharpness_tenengrad")}</th>
              <th>Lum / Contrast / Exp</th>
              <th>JPEG Block</th>
              <th>IOD / Yaw / Roll</th>
              <th data-tip-key="classical_composite_score">Classical Composite ${helpBadgeHtml("classical_composite_score")}</th>
              <th>P1 Warnings</th>
            </tr>
          </thead>
          <tbody>
            ${items
              .map((it, rIdx) => {
                const e = it.extended_p3;
                const c = it.classical_p1;
                return `<tr data-ql-row="${rIdx}" class="${rIdx === selectedQualityIdx ? "selected-row" : ""}" style="cursor:pointer;${
                  it.retained_after_filter ? "" : "opacity:0.52;"
                }">
                  <td><img src="${it.crop_data_url}" style="width:48px;height:48px;border-radius:4px;" /></td>
                  <td class="wb-mono">${esc(it.face_instance_id)}</td>
                  <td>
                    <span class="wb-badge">${esc(it.quality_category)}</span>
                    <div class="wb-muted-xs">${it.retained_after_filter ? "Retained" : "Rejected"}</div>
                  </td>
                  <td class="wb-mono"><strong>${e.learned_embedding_l2_norm.toFixed(3)}</strong></td>
                  <td class="wb-mono">${e.sharpness_laplacian_var.toFixed(1)} / ${e.sharpness_tenengrad.toFixed(0)}</td>
                  <td class="wb-mono">${e.mean_luminance.toFixed(0)} / ${e.contrast_std.toFixed(1)} / ${e.exposure_score.toFixed(2)}</td>
                  <td class="wb-mono">${e.jpeg_blockiness_ratio.toFixed(2)}</td>
                  <td class="wb-mono">${e.inter_ocular_px.toFixed(1)}px / ${e.yaw_asymmetry_ratio.toFixed(2)} / ${e.roll_angle_deg.toFixed(1)}°</td>
                  <td class="wb-mono">${e.classical_composite_score.toFixed(3)}</td>
                  <td class="wb-mono" style="font-size:10.5px;">${c.warnings.length ? esc(c.warnings.join(", ")) : "—"}</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>
    `;

    container.querySelectorAll("[data-ql-row]").forEach((tr) => {
      tr.addEventListener("click", () => {
        selectedQualityIdx = Number(tr.getAttribute("data-ql-row"));
        container.querySelectorAll("[data-ql-row]").forEach((r) => {
          r.classList.toggle("selected-row", Number(r.getAttribute("data-ql-row")) === selectedQualityIdx);
        });
        syncQualityViewerToRow(selectedQualityIdx);
      });
    });
  }

  const bindDeg = (id, labelId, key, fmtFn) => {
    const el = container.querySelector(id);
    const lbl = container.querySelector(labelId);
    el.addEventListener("input", () => {
      const v = Number(el.value);
      store.exp.qualityDegradation[key] = v;
      lbl.textContent = fmtFn(v);
    });
    el.addEventListener("change", () => executeQuality());
  };
  bindDeg("#ql-blur", "#ql-blur-val", "blur_sigma", (v) => v.toFixed(1));
  bindDeg("#ql-res", "#ql-res-val", "effective_res", (v) => `${v}×${v}`);
  bindDeg("#ql-gam", "#ql-gam-val", "gamma", (v) => v.toFixed(1));
  bindDeg("#ql-jpg", "#ql-jpg-val", "jpeg_quality", (v) => String(v));

  container.querySelector("#ql-reset-deg-btn").addEventListener("click", () => {
    store.exp.qualityDegradation = { blur_sigma: 0.0, effective_res: 112, gamma: 1.0, jpeg_quality: 100 };
    renderQualityLab(container);
  });
  const qlRunBtn = container.querySelector("#ql-run-btn");
  container.querySelector("#ql-rec-select").addEventListener("change", () => executeQuality());
  qlRunBtn.addEventListener("click", () => withBusyButton(qlRunBtn, () => executeQuality(), "Assessing…"));

  await executeQuality();
}
