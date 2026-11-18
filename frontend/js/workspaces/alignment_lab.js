/**
 * Workspace 4 — Landmark & Alignment Laboratory (`frontend/js/workspaces/alignment_lab.js`).
 *
 * Interactive inspection of YuNet vs SCRFD 5-point landmarks, 4-DoF Umeyama vs 6-DoF Affine warps,
 * canonical template correspondence lines, 2x3 transformation parameters, and pixel crop differences.
 */

import { api } from "../api.js";
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

export async function renderAlignmentLab(container) {
  const alImages = Array.from(new Set([store.exp.selectedImageRef, ...store.exp.explorerImages].filter(Boolean)));
  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>04 · Landmark &amp; Alignment Laboratory</h2>
        <p class="wb-workspace-desc">
          Compare YuNet vs. SCRFD 5-point landmarks, Umeyama 4-DoF similarity vs. 6-DoF affine warps,
          interpolation kernels, and controlled landmark perturbations. Overlays show geometric landmark correspondences (not embedding similarity).
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <button class="wb-btn wb-btn-sm" id="al-run-btn" type="button" title="Run 5-point landmark detection and geometric warp on the selected face">▶ Compute Alignment</button>
      </div>
    </div>

    <div class="wb-split-layout">
      <!-- Controls -->
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Source Image &amp; Landmark Configuration ${helpBadgeHtml("canonical_template")}</span></div>
          <div class="wb-panel-body">
            <div class="wb-field">
              <label class="wb-label" for="al-image-select">Source Image</label>
              <select id="al-image-select" class="wb-select" title="Select source photograph (click any detected face box in Viewer 1 to align that individual)">
                ${alImages
                  .map(
                    (r) =>
                      `<option value="${esc(r)}" ${r === store.exp.selectedImageRef ? "selected" : ""}>${esc(
                        r.split("/").pop()
                      )}</option>`
                  )
                  .join("")}
              </select>
            </div>
            <div class="wb-field">
              <label class="wb-label" for="al-lm-source">Primary Landmark Detector</label>
              <select id="al-lm-source" class="wb-select" title="Detector supplying the 5 anatomical landmark coordinates (eyes, nose, mouth corners)">
                <option value="yunet_2023mar" ${store.exp.landmarkSource === "yunet_2023mar" ? "selected" : ""}>YuNet 5-Point Landmarks</option>
                <option value="scrfd_500m_research" ${store.exp.landmarkSource === "scrfd_500m_research" ? "selected" : ""}>SCRFD-500M 5-Point Landmarks</option>
              </select>
            </div>
            <div class="wb-field" data-tip-key="transform_mode">
              <label class="wb-label" for="al-cmp-mode">Comparison Mode ${helpBadgeHtml("transform_mode")}</label>
              <select id="al-cmp-mode" class="wb-select" data-tip-key="transform_mode">
                <option value="both_detectors">Overlay &amp; Diff YuNet vs. SCRFD Landmarks</option>
                <option value="vs_affine6">Compare Current Warp vs. 6-DoF Affine</option>
                <option value="vs_zoom10">Compare Current Warp vs. +10% Zoom-Out</option>
              </select>
            </div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Geometric Alignment Parameters</span></div>
          <div class="wb-panel-body" id="al-param-editor"></div>
        </div>
      </div>

      <!-- Visualizers & Matrices -->
      <div>
        <div class="wb-grid-2" style="margin-bottom:12px;">
          <div id="al-orig-viewer"></div>
          <div id="al-crop-viewer"></div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Side-by-Side Aligned Crops &amp; Pixel Difference Heatmap (TURBO)</span></div>
          <div class="wb-panel-body" id="al-crops-strip"></div>
        </div>

        <div class="wb-grid-2">
          <div class="wb-panel">
            <div class="wb-panel-header"><span>2×3 Transformation Matrix &amp; Decomposition</span></div>
            <div class="wb-panel-body" id="al-matrix-box"></div>
          </div>
          <div class="wb-panel">
            <div class="wb-panel-header"><span>5-Point Landmark Coordinates &amp; Displacement</span></div>
            <div class="wb-panel-body" id="al-disp-box"></div>
          </div>
        </div>
      </div>
    </div>
  `;

  const origViewer = new ScientificImageViewer(container.querySelector("#al-orig-viewer"), {
    title: "1. Original Frame + Detected Landmarks + Crop Boundary",
    heightPx: 370,
    overlays: { landmarkLabels: true },
    onSelectFace: (idx) => {
      store.exp.selectedFaceIndex = idx;
      executeAlignment();
    },
  });

  const cropViewer = new ScientificImageViewer(container.querySelector("#al-crop-viewer"), {
    title: "2. Aligned Crop + Canonical Template Correspondences",
    heightPx: 370,
    overlays: { bboxes: false, gtBoxes: false, landmarkLabels: true, correspondenceLines: true },
  });

  const alignSpec = store.algorithmsById["five_point_canonical_aligner"];
  renderParameterEditor(container.querySelector("#al-param-editor"), alignSpec, store.exp.alignmentParams, () => {
    executeAlignment();
  });

  async function executeAlignment() {
    const cmpEl = container.querySelector("#al-cmp-mode");
    if (!cmpEl) return;
    try {
      const cmpVal = cmpEl.value;
      const compareBoth = cmpVal === "both_detectors";
      const compareTransform =
        cmpVal === "vs_affine6"
          ? "affine_least_squares_6dof"
          : cmpVal === "vs_zoom10"
          ? "similarity_zoom_out_10pct"
          : null;

      const res = await api.runAlignment({
        image_ref: store.exp.selectedImageRef,
        face_index: store.exp.selectedFaceIndex,
        landmark_source: store.exp.landmarkSource,
        compare_both_detectors: compareBoth,
        compare_transform_mode: compareTransform,
        params: store.exp.alignmentParams,
      });
      store.exp.lastAlignmentResult = res;
      if (!container.querySelector("#al-crops-strip")) return;

      const prim = res.primary_alignment;
      const sec = res.secondary_alignment;

      // Update Original Viewer
      await origViewer.loadImageUrl(api.rawImageUrl(store.exp.selectedImageRef), res.image_meta, true);
      const detList = res.all_detected_faces.map((f, i) => ({
        ...f,
        landmarks:
          i === prim.selected_face_index
            ? { points: prim.perturbed_source_landmarks_orig_px }
            : undefined,
      }));
      origViewer.setScene({
        detections: detList,
        selectedFaceIndex: prim.selected_face_index,
        gtBoxes: res.image_meta.ground_truth_boxes || [],
        secondaryLandmarks: sec ? sec.source_landmarks_orig_px : null,
        alignmentQuadOrig: prim.crop_boundary_quad_orig_px,
      });

      // Update Crop Viewer (Crop-relative coordinates!)
      await cropViewer.loadImageUrl(prim.crop_data_url, { format: "PNG" }, false);
      cropViewer.setScene({
        detections: [
          {
            bounding_box: { x: 0, y: 0, w: prim.crop_size_wh[0], h: prim.crop_size_wh[1] },
            landmarks: { points: prim.transformed_landmarks_crop_px },
          },
        ],
        selectedFaceIndex: 0,
        templateCorrespondences: prim.destination_template_crop_px,
      });

      // Side-by-side crops + difference heatmap
      container.querySelector("#al-crops-strip").innerHTML = `
        <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;">
          <div style="text-align:center;">
            <img src="${prim.crop_data_url}" style="width:128px;height:128px;border:2px solid var(--accent);border-radius:4px;image-rendering:auto;" />
            <div class="wb-mono" style="font-size:11px;margin-top:4px;">Primary: ${esc(prim.landmark_source)}<br/>(${esc(
        prim.transform_mode
      )})</div>
          </div>
          ${
            sec
              ? `
              <div style="text-align:center;">
                <img src="${sec.crop_data_url}" style="width:128px;height:128px;border:2px solid #f97316;border-radius:4px;" />
                <div class="wb-mono" style="font-size:11px;margin-top:4px;">Alt: ${esc(sec.landmark_source)}<br/>(${esc(
                  sec.transform_mode
                )})</div>
              </div>
              <div style="text-align:center;">
                <img src="${res.crop_difference_data_url}" style="width:128px;height:128px;border:1px solid var(--border-strong);border-radius:4px;" />
                <div class="wb-mono" style="font-size:11px;margin-top:4px;">Abs Diff Heatmap (×2.5)<br/>Mean |ΔI| = <strong>${res.crop_mean_abs_pixel_diff}</strong> / 255</div>
              </div>`
              : `<div class="wb-help">No secondary face/alignment matched for comparison.</div>`
          }
          <div style="flex:1;min-width:200px;" class="wb-help">
            <strong>Scientific Note:</strong> ${esc(res.geometry_note)}<br/>
            Alignment Warp Time: <strong class="wb-mono">${prim.execution_time_ms} ms</strong> (${esc(prim.interpolation)})
          </div>
        </div>
      `;

      const m = prim.affine_matrix_2x3;
      const dec = prim.decomposition;
      container.querySelector("#al-matrix-box").innerHTML = `
        <pre class="wb-mono" style="background:var(--bg-metric);padding:8px;border-radius:4px;margin:0 0 8px 0;font-size:11.5px;">[ ${m[0]
          .map((v) => v.toFixed(5).padStart(10))
          .join(", ")} ]
[ ${m[1].map((v) => v.toFixed(5).padStart(10)).join(", ")} ]</pre>
        <table class="wb-table">
          <tbody>
            <tr><th>Estimated Scale</th><td class="wb-mono">${dec.mean_scale}</td></tr>
            <tr><th>In-Plane Rotation</th><td class="wb-mono">${dec.rotation_deg}°</td></tr>
            <tr><th>Translation (tx, ty)</th><td class="wb-mono">(${dec.translation_xy_px[0]}, ${dec.translation_xy_px[1]}) px</td></tr>
            <tr><th>Anisotropic Shear Ratio</th><td class="wb-mono">${dec.anisotropy_ratio} ${
        dec.anisotropy_ratio > 0.001 ? "(Affine Shear Active)" : "(Isotropic Similarity)"
      }</td></tr>
            <tr><th>Source Inter-Ocular Dist</th><td class="wb-mono">${dec.inter_ocular_orig_px} px</td></tr>
            <tr><th>Mean Template Fit Residual</th><td class="wb-mono">${prim.mean_fit_residual_px} px</td></tr>
          </tbody>
        </table>
      `;

      const disp = res.landmark_displacement;
      container.querySelector("#al-disp-box").innerHTML = `
        <table class="wb-table">
          <thead>
            <tr>
              <th>Landmark</th>
              <th>Source (px)</th>
              <th>Warped → Dest (px)</th>
              <th>${disp ? "YuNet↔SCRFD Δ" : "Fit Residual"}</th>
            </tr>
          </thead>
          <tbody>
            ${res.canonical_landmark_names
              .map((nm, k) => {
                const sp = prim.source_landmarks_orig_px[k];
                const tp = prim.transformed_landmarks_crop_px[k];
                const dp = prim.destination_template_crop_px[k];
                return `<tr>
                  <td class="wb-mono" style="font-size:11px;">${esc(nm)}</td>
                  <td class="wb-mono">(${sp[0].toFixed(1)}, ${sp[1].toFixed(1)})</td>
                  <td class="wb-mono">(${tp[0].toFixed(1)}, ${tp[1].toFixed(1)}) → (${dp[0].toFixed(1)}, ${dp[1].toFixed(1)})</td>
                  <td class="wb-mono">${
                    disp ? `${disp.per_point_displacement_px[k].toFixed(2)} px` : `${prim.landmark_fit_residual_px[k].toFixed(2)} px`
                  }</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
        ${
          disp
            ? `<div class="wb-help" style="margin-top:6px;">Mean Cross-Detector Landmark Displacement: <strong class="wb-mono">${disp.mean_displacement_px} px</strong> (Max <strong class="wb-mono">${disp.max_displacement_px} px</strong>, Box IoU <strong class="wb-mono">${disp.bbox_iou}</strong>)</div>`
            : ""
        }
      `;
    } catch (err) {
      showToast(`Alignment lab error: ${err.message}`, true);
    }
  }

  container.querySelector("#al-image-select").addEventListener("change", (e) => {
    store.exp.selectedImageRef = e.target.value;
    store.exp.selectedFaceIndex = 0;
    executeAlignment();
  });
  container.querySelector("#al-lm-source").addEventListener("change", (e) => {
    store.exp.landmarkSource = e.target.value;
    executeAlignment();
  });
  container.querySelector("#al-cmp-mode").addEventListener("change", () => executeAlignment());
  const alRunBtn = container.querySelector("#al-run-btn");
  alRunBtn.addEventListener("click", () => withBusyButton(alRunBtn, () => executeAlignment(), "Aligning…"));

  await executeAlignment();
}
