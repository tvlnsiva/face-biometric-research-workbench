/**
 * Workspace 3 — Face Detection Laboratory (`frontend/js/workspaces/detection_lab.js`).
 *
 * Model-specific dynamic parameter controls, GT TP/FP/FN overlays, baseline comparison, and
 * async parameter sweeps across resolution, confidence, NMS, and canvas padding strategies.
 */

import { api } from "../api.js";
import { renderLineChart } from "../components/charts.js";
import { ScientificImageViewer } from "../components/image_viewer.js";
import { renderParameterEditor } from "../components/param_editor.js";
import { showToast, store } from "../state.js";
import { withBusyButton } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderDetectionLab(container, _navigateTo, waitJobFn) {
  const detectors = (store.registry?.algorithms || []).filter((a) => a.category === "detection");
  const detImages = Array.from(new Set([store.exp.selectedImageRef, ...store.exp.explorerImages].filter(Boolean)));

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>03 · Face Detection Laboratory</h2>
        <p class="wb-workspace-desc">
          Compare YuNet and SCRFD-500M with model-specific parameters, inspect pixel-accurate bounding boxes
          and 5-point landmarks, compare against frozen baselines, and execute parameter sweeps.
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="det-compare-base-btn" type="button" title="Run detection alongside frozen baseline defaults and display delta">Compare With Baseline</button>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="det-save-run-btn" type="button" title="Execute detection and persist a reproducible experiment record in Experiment History">Save Run</button>
        <button class="wb-btn wb-btn-sm" id="det-run-btn" type="button" title="Execute the selected face detector with current parameters on the target image">▶ Run Detection</button>
      </div>
    </div>

    <div class="wb-split-layout">
      <!-- Left: Model & Model-Specific Parameters + Sweep Launcher -->
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Image &amp; Detector Selection</span></div>
          <div class="wb-panel-body">
            <div class="wb-field">
              <label class="wb-label" for="det-image-select">Target Image</label>
              <select id="det-image-select" class="wb-select" title="Select target image from current working set (add more via Image Explorer)">
                ${detImages
                  .map(
                    (r) =>
                      `<option value="${esc(r)}" ${r === store.exp.selectedImageRef ? "selected" : ""}>${esc(
                        r.split("/").pop()
                      )} (${esc(r.split("/").slice(-2, -1)[0] || "")})</option>`
                  )
                  .join("")}
              </select>
            </div>
            <div class="wb-field" style="margin-bottom:0;">
              <label class="wb-label" for="det-algo-select">Detection Algorithm</label>
              <select id="det-algo-select" class="wb-select" title="Switch between YuNet (OpenCV CPU) and SCRFD-500M (ONNX Runtime CUDA/CPU)">
                ${detectors
                  .map(
                    (d) =>
                      `<option value="${esc(d.algorithm_id)}" ${
                        d.algorithm_id === store.exp.detectorId ? "selected" : ""
                      }>${esc(d.name)}</option>`
                  )
                  .join("")}
              </select>
            </div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Model-Specific Parameters</span></div>
          <div class="wb-panel-body" id="det-param-editor"></div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Controlled Parameter Sweep</span></div>
          <div class="wb-panel-body">
            <div class="wb-field">
              <label class="wb-label" for="det-sweep-param">Sweep Parameter</label>
              <select id="det-sweep-param" class="wb-select" title="Parameter varied across multiple operating points during asynchronous sweep">
                <option value="score_threshold">Confidence Threshold (0.30 .. 0.85)</option>
                <option value="resolution">Input Resolution (320 .. 1600)</option>
                <option value="nms_threshold">Hard-NMS IoU Threshold (0.20 .. 0.70)</option>
              </select>
            </div>
            <div class="wb-field">
              <label class="wb-label" for="det-sweep-scope">Evaluation Target</label>
              <select id="det-sweep-scope" class="wb-select" title="Dataset split or single image evaluated at each sweep step">
                <option value="dataset_p11">NASA P1.1 Multi-Scale Split (9 Imgs / 86 GT Faces)</option>
                <option value="dataset_p1">NASA P1 Human-Verified Split (6 Imgs / 15 GT Faces)</option>
                <option value="current_image">Current Active Image Only</option>
              </select>
            </div>
            <button class="wb-btn wb-btn-secondary" style="width:100%;" id="det-sweep-btn" type="button" title="Submit serialized background sweep job and plot Recall/F1/Latency response curve">
              Run Parameter Sweep (Async Queue)
            </button>
          </div>
        </div>
      </div>

      <!-- Right: Viewer + Quantitative Statistics + Sweep Charts -->
      <div>
        <div id="det-viewer-box" style="margin-bottom:12px;"></div>

        <div class="wb-grid-6" id="det-kpi-row" style="margin-bottom:12px;"></div>

        <div id="det-baseline-diff-box"></div>

        <div class="wb-grid-2">
          <div class="wb-panel">
            <div class="wb-panel-header"><span>Detected Face Instances (Original Pixel Space)</span></div>
            <div class="wb-panel-body" id="det-faces-table" style="max-height:280px;overflow-y:auto;"></div>
          </div>
          <div class="wb-panel">
            <div class="wb-panel-header"><span>Sweep Response Curve</span></div>
            <div class="wb-panel-body" id="det-sweep-chart">
              <div class="wb-help">Launch a parameter sweep on the left to plot Recall, F1, and Latency vs. parameter value.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  const viewer = new ScientificImageViewer(container.querySelector("#det-viewer-box"), {
    title: "Face Detection & Ground-Truth Overlay Viewer",
    heightPx: 440,
    onSelectFace: (idx) => {
      store.exp.selectedFaceIndex = idx;
    },
  });

  function mountParamEditor() {
    const algo = store.algorithmsById[store.exp.detectorId];
    const values = store.exp.detectorParamsByModel[store.exp.detectorId];
    renderParameterEditor(container.querySelector("#det-param-editor"), algo, values, () => {});
  }

  async function executeDetection({ compareBaseline = false, saveRun = false } = {}) {
    try {
      const detId = store.exp.detectorId;
      const res = await api.runDetection({
        image_ref: store.exp.selectedImageRef,
        detector_id: detId,
        params: store.exp.detectorParamsByModel[detId] || {},
        compare_baseline: compareBaseline,
        save_experiment: saveRun,
      });
      store.exp.lastDetectionResult = res;
      if (res.saved_experiment_id) {
        const expResp = await api.listExperiments(true);
        store.experiments = expResp.experiments;
        showToast(`Saved detection experiment ${res.saved_experiment_id}`);
      }

      await viewer.loadImageUrl(api.rawImageUrl(store.exp.selectedImageRef), res.image_meta, true);
      viewer.setScene({
        detections: res.detections,
        selectedFaceIndex: 0,
        gtBoxes: res.image_meta.ground_truth_boxes || [],
        unmatchedGtBoxes: res.gt_evaluation.unmatched_gt_boxes || [],
      });

      renderStats(res);
    } catch (err) {
      showToast(`Detection failed: ${err.message}`, true);
    }
  }

  function renderStats(res) {
    const kpiRow = container.querySelector("#det-kpi-row");
    if (!kpiRow) return;
    const gt = res.gt_evaluation;
    const lat = res.latency_and_throughput;
    const gpu = res.hardware_snapshot?.gpu || {};
    const na = `<span style="font-size:12px;color:var(--text-muted);">UNAVAILABLE (No GT)</span>`;

    kpiRow.innerHTML = `
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Detections</span></div>
        <div class="wb-metric-value">${res.detection_count}</div>
        <div class="wb-metric-prov">${gt.available ? `GT: ${gt.gt_count} (TP ${gt.tp} · FP ${gt.fp} · FN ${gt.fn})` : "No GT manifest"}</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Precision / Recall</span></div>
        <div class="wb-metric-value">${gt.available ? `${(gt.precision * 100).toFixed(1)}% / ${(gt.recall * 100).toFixed(1)}%` : na}</div>
        <div class="wb-metric-prov">${gt.available ? `IoU ≥ ${gt.iou_threshold}` : esc(gt.reason)}</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>F1 / Mean IoU</span></div>
        <div class="wb-metric-value">${gt.available ? `${gt.f1.toFixed(4)} / ${gt.mean_iou.toFixed(3)}` : na}</div>
        <div class="wb-metric-prov">${
          gt.available
            ? `S:${gt.small_lt32_recall ?? "—"} M:${gt.medium_32_64_recall ?? "—"} L:${gt.large_gt64_recall ?? "—"}`
            : "Size recall unavailable"
        }</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Total Latency</span></div>
        <div class="wb-metric-value">${lat.total_detect_ms.toFixed(2)} ms</div>
        <div class="wb-metric-prov">Input ${res.execution_meta.detector_input_size_wh.join("×")} (${esc(res.execution_meta.device)})</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Throughput</span></div>
        <div class="wb-metric-value">${lat.images_per_sec} img/s</div>
        <div class="wb-metric-prov">${lat.faces_per_sec} faces/s emitted</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Memory</span></div>
        <div class="wb-metric-value">${gpu.vram_device_used_mib ? `${gpu.vram_device_used_mib} MiB` : `${res.hardware_snapshot.process_rss_mib} MiB`}</div>
        <div class="wb-metric-prov">RSS ${res.hardware_snapshot.process_rss_mib} MiB</div>
      </div>
    `;

    const baseBox = container.querySelector("#det-baseline-diff-box");
    if (res.baseline_comparison) {
      const b = res.baseline_comparison;
      baseBox.innerHTML = `
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Baseline vs. Current Configuration Comparison</span></div>
          <div class="wb-panel-body wb-mono" style="font-size:11.5px;">
            Baseline Detections: <strong>${b.baseline_detection_count}</strong> (${b.baseline_latency_ms} ms${
        b.baseline_gt.available ? `, F1=${b.baseline_gt.f1}` : ""
      })
            &nbsp;→&nbsp;
            Current Detections: <strong>${b.current_detection_count}</strong> (${b.current_latency_ms} ms${
        gt.available ? `, F1=${gt.f1}` : ""
      })
          </div>
        </div>
      `;
    } else {
      baseBox.innerHTML = "";
    }

    container.querySelector("#det-faces-table").innerHTML =
      res.detections.length === 0
        ? `<div class="wb-help">0 detections emitted.</div>`
        : `<table class="wb-table">
            <thead><tr><th>ID</th><th>Box (x, y, w, h)</th><th>Category</th><th>Conf</th><th>GT</th></tr></thead>
            <tbody>
              ${res.detections
                .map(
                  (d) => `<tr>
                    <td class="wb-mono">face_${String(d.face_index).padStart(3, "0")}</td>
                    <td class="wb-mono">(${d.bounding_box.x.toFixed(0)}, ${d.bounding_box.y.toFixed(0)}, ${d.bounding_box.w.toFixed(0)}×${d.bounding_box.h.toFixed(0)})</td>
                    <td>${esc(d.size_category)}</td>
                    <td class="wb-mono">${(d.confidence * 100).toFixed(2)}%</td>
                    <td><span class="wb-badge">${esc(d.gt_status.toUpperCase())}${d.gt_iou !== null ? ` (${d.gt_iou})` : ""}</span></td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>`;
  }

  function renderSweepChart(sweepRes) {
    if (!sweepRes) return;
    const pts = sweepRes.points || [];
    const hasGt = pts.some((p) => p.f1 !== null && p.f1 !== undefined);
    if (hasGt) {
      renderLineChart(container.querySelector("#det-sweep-chart"), {
        title: `${sweepRes.detector_id} — F1 & Recall vs. ${sweepRes.sweep_param}`,
        subtitle: `Split: ${sweepRes.dataset_split}`,
        xLabel: sweepRes.sweep_param,
        yLabel: "Score [0..1]",
        series: [
          { name: "F1 Score", points: pts.map((p) => ({ x: Number(p.param_value), y: Number(p.f1 ?? 0) })) },
          { name: "Recall", points: pts.map((p) => ({ x: Number(p.param_value), y: Number(p.recall ?? 0) })) },
          {
            name: "Small (<32px) Recall",
            points: pts
              .filter((p) => p.small_lt32_recall !== null)
              .map((p) => ({ x: Number(p.param_value), y: Number(p.small_lt32_recall) })),
          },
        ],
      });
    } else {
      renderLineChart(container.querySelector("#det-sweep-chart"), {
        title: `${sweepRes.detector_id} — P50 Latency (ms) vs. ${sweepRes.sweep_param}`,
        subtitle: `Split: ${sweepRes.dataset_split} (No GT)`,
        xLabel: sweepRes.sweep_param,
        yLabel: "P50 Latency (ms)",
        series: [{ name: "P50 Latency (ms)", points: pts.map((p) => ({ x: Number(p.param_value), y: p.p50_latency_ms })) }],
      });
    }
  }

  container.querySelector("#det-image-select").addEventListener("change", (e) => {
    store.exp.selectedImageRef = e.target.value;
    executeDetection();
  });

  container.querySelector("#det-algo-select").addEventListener("change", (e) => {
    store.exp.detectorId = e.target.value;
    mountParamEditor();
    store.notify("config_change");
    executeDetection();
  });

  const runBtn = container.querySelector("#det-run-btn");
  const cmpBtn = container.querySelector("#det-compare-base-btn");
  const saveBtn = container.querySelector("#det-save-run-btn");
  const sweepBtn = container.querySelector("#det-sweep-btn");

  runBtn.addEventListener("click", () => withBusyButton(runBtn, () => executeDetection(), "Detecting…"));
  cmpBtn.addEventListener("click", () => withBusyButton(cmpBtn, () => executeDetection({ compareBaseline: true }), "Comparing…"));
  saveBtn.addEventListener("click", () => withBusyButton(saveBtn, () => executeDetection({ saveRun: true }), "Saving…"));

  sweepBtn.addEventListener("click", () =>
    withBusyButton(sweepBtn, async () => {
      const detId = store.exp.detectorId;
      const rawKind = container.querySelector("#det-sweep-param").value;
      const scope = container.querySelector("#det-sweep-scope").value;

      let sweepParam = rawKind;
      let sweepValues = [0.3, 0.45, 0.5, 0.6, 0.75, 0.85];
      if (rawKind === "resolution") {
        sweepParam = detId === "yunet_2023mar" ? "max_input_dim" : "max_dim";
        sweepValues = [320, 640, 960, 1280, 1600];
      } else if (rawKind === "nms_threshold") {
        sweepValues = [0.2, 0.3, 0.4, 0.55, 0.7];
      }

      const jobSub = await api.submitJob({
        job_type: "detection_sweep",
        name: `Detection Sweep: ${detId} (${sweepParam})`,
        config: {
          detector_id: detId,
          sweep_param: sweepParam,
          sweep_values: sweepValues,
          base_params: store.exp.detectorParamsByModel[detId] || {},
          dataset_id: scope === "dataset_p1" ? "nasa_p1_detection" : "nasa_p11_multiscale",
          image_ref: scope === "current_image" ? store.exp.selectedImageRef : null,
        },
        save_experiment: true,
      });
      showToast(`Submitted sweep job ${jobSub.job_id}`);
      const done = await waitJobFn(jobSub.job_id);
      if (done?.result && container.querySelector("#det-sweep-chart")) {
        store.exp.lastDetectionSweep = done.result;
        renderSweepChart(done.result);
      }
    }, "Sweeping…")
  );

  mountParamEditor();
  if (store.exp.lastDetectionSweep) renderSweepChart(store.exp.lastDetectionSweep);
  await executeDetection();
}
