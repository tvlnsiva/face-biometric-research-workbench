/**
 * Workspace 9 — Multi-Run Algorithm Comparison Workspace (`frontend/js/workspaces/comparison_workspace.js`).
 *
 * Side-by-side comparison table, automatic cross-dataset/protocol mismatch warnings, and all 12
 * requested scientific performance/trade-off charts backed by real P2/P3 and Workbench runs.
 */

import { api } from "../api.js";
import { renderBarChart, renderLineChart } from "../components/charts.js";
import { renderTwoRunConfigComparison } from "../components/config_delta.js";
import { renderScoreDistributionExplorer } from "../components/threshold_explorer.js";
import { showToast, store } from "../state.js";
import { helpBadgeHtml, sanitizeNumericInput } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderComparisonWorkspace(container, navigateTo) {
  const expResp = await api.listExperiments(true);
  if (!container.isConnected) return;
  store.experiments = expResp.experiments;
  const exps = store.experiments;

  let activeChartMode = "recall_vs_resolution";

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>09 · Algorithm &amp; Configuration Comparison Workspace</h2>
        <p class="wb-workspace-desc">
          Compare detection, alignment, recognition, batching, precision, and 1:N retrieval runs side by side.
          Cross-dataset comparisons automatically trigger protocol mismatch warnings.
        </p>
      </div>
      <div class="wb-toolbar-actions wb-muted-xs" style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
        <span>Metric Glossary:</span>
        <span>TAR ${helpBadgeHtml("tar")}</span>
        <span>FAR ${helpBadgeHtml("far")}</span>
        <span>FRR ${helpBadgeHtml("frr")}</span>
        <span>FNMR ${helpBadgeHtml("fnmr")}</span>
        <span>EER ${helpBadgeHtml("eer")}</span>
        <span>ROC AUC ${helpBadgeHtml("roc_auc")}</span>
        <span>Recall@K ${helpBadgeHtml("recall_at_k")}</span>
      </div>
    </div>

    <div id="cmp-protocol-warning"></div>

    <div class="wb-split-layout">
      <!-- Run Selector -->
      <div class="wb-panel">
        <div class="wb-panel-header"><span>Select Runs to Compare (${exps.length} Available)</span></div>
        <div class="wb-panel-body" style="max-height:540px;overflow-y:auto;" id="cmp-run-list">
          ${exps
            .map((e) => {
              const checked = store.exp.selectedExperimentIds.includes(e.experiment_id);
              return `
                <label style="display:flex;align-items:flex-start;gap:7px;padding:6px 4px;border-bottom:1px solid var(--border);cursor:pointer;"
                  data-tip-title="${esc(e.name)}"
                  data-tip-body="Dataset: ${esc(e.dataset_id)} (${esc(e.dataset_split)}) · Precision: ${esc(e.precision)} · Batch: ${esc(e.batch_size)}">
                  <input type="checkbox" data-exp-cb="${esc(e.experiment_id)}" aria-label="Include ${esc(e.name)} in comparison" ${checked ? "checked" : ""} style="margin-top:3px;" />
                  <div>
                    <div style="font-weight:600;font-size:11.5px;">${esc(e.name)}</div>
                    <div class="wb-muted-xs wb-mono">${esc(e.experiment_id)} · ${esc(e.dataset_id)}</div>
                  </div>
                </label>
              `;
            })
            .join("")}
        </div>
      </div>

      <!-- Comparison Table + 12 Chart Modes -->
      <div style="min-width:0;">
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Side-by-Side Comparison Matrix</span></div>
          <div class="wb-panel-body" id="cmp-matrix-box"></div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header" style="flex-wrap:wrap;gap:6px;">
            <span>Scientific Trade-Off &amp; Operating-Point Charts</span>
            <div style="display:flex;align-items:center;gap:6px;max-width:100%;">
              <label for="cmp-chart-mode" class="wb-muted-xs">Plot:</label>
              <select id="cmp-chart-mode" class="wb-select" style="width:310px;max-width:100%;"
                data-tip-title="Scientific Chart Mode Selector"
                data-tip-body="Select among 12 empirical detection, verification ROC/DET, latency/throughput, and 1:N retrieval scaling curves.">
                <option value="recall_vs_resolution">1. Detection Recall &amp; F1 vs. Input Resolution</option>
                <option value="recall_vs_confidence">2. Detection Recall &amp; Precision vs. Confidence</option>
                <option value="pr_curve">3. Detection Precision-Recall Curve</option>
                <option value="roc_curve">4. Verification ROC Curve (TAR vs. FAR)</option>
                <option value="det_curve">5. Verification DET Curve (FRR vs. FAR)</option>
                <option value="far_vs_threshold">6. Verification FAR &amp; FRR vs. Threshold</option>
                <option value="acc_vs_latency">7. Verification Accuracy vs. Latency Trade-Off</option>
                <option value="latency_vs_batch">8. Recognition Stream Latency vs. Batch Size</option>
                <option value="throughput_vs_batch">9. Recognition Throughput (faces/s) vs. Batch Size</option>
                <option value="recall_vs_nprobe">10. 1:N Recall@K vs. Index Parameter (nprobe / efSearch)</option>
                <option value="memory_vs_gallery">11. 1:N Index Memory (MiB) vs. Gallery Size</option>
                <option value="selected_bar">12. Selected Runs Headline Metric Bar Comparison</option>
              </select>
            </div>
          </div>
          <div class="wb-panel-body" id="cmp-chart-canvas"></div>
        </div>

        <!-- P-1: Two-Run Configuration Comparison (Stage & Model Context) -->
        <div class="wb-panel">
          <div class="wb-panel-header">
            <span>Two-Run Configuration Comparison (Stage, Model &amp; Parameter Context)</span>
          </div>
          <div class="wb-panel-body" id="cmp-two-run-config-box"></div>
        </div>

        <!-- P-2: Interactive Genuine / Impostor Score Distribution & Threshold Explorer -->
        <div class="wb-panel">
          <div class="wb-panel-header" style="flex-wrap:wrap;gap:8px;">
            <span>Genuine / Impostor Score Distribution &amp; Operating Point</span>
            <div style="display:flex;align-items:center;gap:6px;max-width:100%;">
              <label for="cmp-p2-run-select" class="wb-muted-xs">Evaluation Run:</label>
              <select id="cmp-p2-run-select" class="wb-select" style="width:360px;max-width:100%;"
                aria-label="Select evaluation run for Genuine / Impostor Score Distribution Explorer">
                ${exps
                  .map((e) => {
                    const hasScores = Boolean(e.curves?.score_distributions?.partitions);
                    const tag = hasScores ? "★ [Score Arrays]" : "[Summary Only]";
                    return `<option value="${esc(e.experiment_id)}">${tag} ${esc(e.experiment_id)} — ${esc(
                      e.name.slice(0, 42)
                    )}</option>`;
                  })
                  .join("")}
              </select>
            </div>
          </div>
          <div class="wb-panel-body" id="cmp-p2-explorer-box"></div>
        </div>
      </div>
    </div>
  `;

  function getSelectedRuns() {
    return exps.filter((e) => store.exp.selectedExperimentIds.includes(e.experiment_id));
  }

  function renderMatrixAndWarnings() {
    const sel = getSelectedRuns();
    const dsSet = new Set(sel.map((s) => `${s.dataset_id}::${s.dataset_split}`));
    const warnEl = container.querySelector("#cmp-protocol-warning");
    if (sel.length > 1 && dsSet.size > 1) {
      warnEl.innerHTML = `
        <div class="wb-banner wb-banner-warn">
          ⚠ <strong>Cross-Dataset / Protocol Warning:</strong> Selected experiments span <strong>${dsSet.size}</strong>
          different dataset/split protocols (${esc(Array.from(dsSet).join("  |  "))}).
          Do NOT rank algorithms across different datasets or calibration splits.
        </div>
      `;
    } else {
      warnEl.innerHTML = "";
    }

    const box = container.querySelector("#cmp-matrix-box");
    if (!sel.length) {
      box.innerHTML = `<div class="wb-help">Select at least one experiment on the left to populate the comparison table.</div>`;
      return;
    }

    box.innerHTML = `
      <div class="wb-table-wrap">
        <table class="wb-table">
          <thead>
            <tr>
              <th>Experiment ID</th>
              <th>Algorithm / Model</th>
              <th>Dataset &amp; Split</th>
              <th>Config / Provider</th>
              <th>Prec / Batch</th>
              <th>Accuracy / Recall / EER</th>
              <th>Latency (P50)</th>
              <th>Throughput</th>
              <th>Memory</th>
            </tr>
          </thead>
          <tbody>
            ${sel
              .map((r) => {
                const m = r.metrics || {};
                const accParts = [];
                if (m.f1 !== undefined && m.f1 !== null) accParts.push(`F1=${m.f1}`);
                if (m.recall !== undefined && m.recall !== null) accParts.push(`Rec=${m.recall}`);
                if (m.test_accuracy_pct !== undefined) accParts.push(`Acc=${m.test_accuracy_pct}%`);
                if (m.test_eer_pct !== undefined) accParts.push(`EER=${m.test_eer_pct}%`);
                if (m.test_roc_auc !== undefined) accParts.push(`AUC=${m.test_roc_auc}`);
                if (m.recall_at_1 !== undefined) accParts.push(`R@1=${m.recall_at_1}`);
                if (m.ivf_pq_100k_recall_at_5 !== undefined) accParts.push(`PQ R@5=${m.ivf_pq_100k_recall_at_5}`);

                const latVal =
                  m.p50_latency_ms ??
                  m.detection_p50_ms ??
                  m.fp16_b8_fixed_p50_ms ??
                  m.cuda_fp16_100k_q1_p50_ms ??
                  m.single_query_p50_ms ??
                  m.total_pipeline_ms;
                const tpVal =
                  m.faces_per_sec ??
                  m.peak_throughput_faces_per_sec ??
                  m.images_per_sec ??
                  m.batched_qps ??
                  m.cuda_fp16_100k_q64_qps;
                const memVal = m.gpu_memory_mib ?? m.index_memory_mib ?? m.ivf_pq_100k_memory_mib;

                return `<tr>
                  <td class="wb-mono" style="font-size:11px;">${esc(r.experiment_id)}</td>
                  <td><strong>${esc(r.name)}</strong><div class="wb-muted-xs wb-mono">${esc(JSON.stringify(r.algorithms))}</div></td>
                  <td style="font-size:11px;">${esc(r.dataset_id)}<div class="wb-muted-xs">${esc(r.dataset_split)}</div></td>
                  <td class="wb-mono" style="font-size:10.5px;">${esc(r.execution_provider)}</td>
                  <td class="wb-mono">${esc(r.precision)} / B=${r.batch_size}</td>
                  <td class="wb-mono">${accParts.length ? esc(accParts.join(" · ")) : '<span class="wb-muted-xs">N/A</span>'}</td>
                  <td class="wb-mono">${latVal !== undefined && latVal !== null ? `${latVal} ms` : '<span class="wb-muted-xs">N/A</span>'}</td>
                  <td class="wb-mono">${tpVal !== undefined && tpVal !== null ? tpVal : '<span class="wb-muted-xs">N/A</span>'}</td>
                  <td class="wb-mono">${memVal !== undefined && memVal !== null ? `${memVal} MiB` : '<span class="wb-muted-xs">N/A</span>'}</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function renderSelectedChart() {
    const slot = container.querySelector("#cmp-chart-canvas");
    const p3Det = exps.find((e) => e.experiment_id === "HIST-P3-EXP01-DET-SWEEP");
    const p3Batch = exps.find((e) => e.experiment_id === "HIST-P3-EXP07-BATCH-SWEEP");
    const p3Ret = exps.find((e) => e.experiment_id === "HIST-P3-EXP08-RETRIEVAL-SCALING");
    const wbWithRoc = exps.filter((e) => Array.isArray(e.curves?.roc_det_sweep) && e.curves.roc_det_sweep.length > 0);

    if (activeChartMode === "recall_vs_resolution" && p3Det) {
      const y = p3Det.curves.yunet_resolution_sweep || [];
      const s = p3Det.curves.scrfd_resolution_sweep || [];
      renderLineChart(slot, {
        title: "Detection Recall & Small-Face (<32px) Recall vs. Input Resolution",
        subtitle: "Dataset: NASA P1.1 9-Image / 86-Face Split (HIST-P3-EXP01)",
        xLabel: "max_dim (px)",
        yLabel: "Recall [0..1]",
        series: [
          { name: "YuNet Overall Recall", points: y.map((r) => ({ x: r.max_dim, y: r.recall })) },
          { name: "SCRFD-500M Overall Recall", points: s.map((r) => ({ x: r.max_dim, y: r.recall })) },
          { name: "SCRFD Small (<32px) Recall", points: s.map((r) => ({ x: r.max_dim, y: r.small_lt32_recall })) },
        ],
      });
      return;
    }

    if (activeChartMode === "recall_vs_confidence" && p3Det) {
      const y = p3Det.curves.yunet_confidence_sweep || [];
      const s = p3Det.curves.scrfd_confidence_sweep || [];
      renderLineChart(slot, {
        title: "Detection F1 & Recall vs. Confidence Threshold (@1280)",
        subtitle: "Dataset: NASA P1.1 9-Image / 86-Face Split (HIST-P3-EXP01)",
        xLabel: "score_threshold",
        yLabel: "Metric [0..1]",
        series: [
          { name: "YuNet F1", points: y.map((r) => ({ x: r.score_threshold, y: r.f1 })) },
          { name: "SCRFD-500M F1", points: s.map((r) => ({ x: r.score_threshold, y: r.f1 })) },
          { name: "SCRFD-500M Recall", points: s.map((r) => ({ x: r.score_threshold, y: r.recall })) },
        ],
      });
      return;
    }

    if (activeChartMode === "pr_curve" && p3Det) {
      const y = p3Det.curves.yunet_confidence_sweep || [];
      const s = p3Det.curves.scrfd_confidence_sweep || [];
      renderLineChart(slot, {
        title: "Precision-Recall Operating Curve (Confidence Sweep @1280)",
        subtitle: "Dataset: NASA P1.1 9-Image / 86-Face Split",
        xLabel: "Recall",
        yLabel: "Precision",
        series: [
          { name: "YuNet 2023mar", points: y.map((r) => ({ x: r.recall, y: r.precision, label: `th=${r.score_threshold}` })) },
          { name: "SCRFD-500M", points: s.map((r) => ({ x: r.recall, y: r.precision, label: `th=${r.score_threshold}` })) },
        ],
      });
      return;
    }

    if (activeChartMode === "roc_curve" || activeChartMode === "det_curve" || activeChartMode === "far_vs_threshold") {
      if (wbWithRoc.length > 0) {
        if (activeChartMode === "roc_curve") {
          renderLineChart(slot, {
            title: "Verification ROC Curve (TAR vs. FAR)",
            subtitle: `Workbench Pipeline Verification Runs (${wbWithRoc.length})`,
            xLabel: "False Accept Rate (FAR)",
            yLabel: "True Accept Rate (TAR)",
            series: wbWithRoc.slice(0, 4).map((e) => ({
              name: e.name,
              points: e.curves.roc_det_sweep.map((p) => ({ x: p.far, y: p.tar })),
            })),
          });
        } else if (activeChartMode === "det_curve") {
          renderLineChart(slot, {
            title: "Verification DET Curve (FRR vs. FAR)",
            subtitle: `Workbench Pipeline Verification Runs (${wbWithRoc.length})`,
            xLabel: "False Accept Rate (FAR)",
            yLabel: "False Reject Rate (FRR = 1 − TAR)",
            series: wbWithRoc.slice(0, 4).map((e) => ({
              name: e.name,
              points: e.curves.roc_det_sweep.map((p) => ({ x: p.far, y: p.frr })),
            })),
          });
        } else {
          const e0 = wbWithRoc[0];
          renderLineChart(slot, {
            title: `FAR & FRR vs. Decision Threshold (${e0.name})`,
            subtitle: `Dataset: ${e0.dataset_id}`,
            xLabel: "Threshold τ",
            yLabel: "Error Rate [0..1]",
            series: [
              { name: "FAR(τ)", points: e0.curves.roc_det_sweep.map((p) => ({ x: p.threshold, y: p.far })) },
              { name: "FRR(τ)", points: e0.curves.roc_det_sweep.map((p) => ({ x: p.threshold, y: p.frr })) },
            ],
          });
        }
        return;
      }
      // Fallback to P3 calibrated operating points across recognizers
      const verRuns = exps.filter((e) => e.metrics?.test_tar_at_calib_far_1e3 !== undefined);
      renderLineChart(slot, {
        title: "Realized Held-Out TAR vs. Realized FAR at Calibrated FAR≤10⁻³ (LFW Folds 6–10)",
        subtitle: "Tip: Run a Verification Pipeline in Workspace 08 to add dense 45-point ROC/DET curves!",
        xLabel: "Realized Held-Out FAR",
        yLabel: "Realized Held-Out TAR",
        series: verRuns.map((r) => ({
          name: r.name,
          points: [{ x: r.metrics.test_realized_far_at_calib_1e3, y: r.metrics.test_tar_at_calib_far_1e3 }],
        })),
      });
      return;
    }

    if (activeChartMode === "acc_vs_latency") {
      const ttaRuns = exps.filter((e) => e.experiment_id.startsWith("HIST-P3-EXP06-"));
      renderLineChart(slot, {
        title: "LFW Held-Out Accuracy (%) vs. B=8 Stream P50 Latency (ms)",
        subtitle: "Dataset: LFW Folds 6–10 (EXP-P3-06 Single-Pass vs. Flip-TTA)",
        xLabel: "B=8 Latency P50 (ms)",
        yLabel: "Held-Out Test Accuracy (%)",
        series: ttaRuns.map((r) => ({
          name: r.name.replace("P3 EXP-P3-06: ", ""),
          points: [{ x: r.metrics.p50_latency_ms, y: r.metrics.test_accuracy_pct }],
        })),
      });
      return;
    }

    if ((activeChartMode === "latency_vs_batch" || activeChartMode === "throughput_vs_batch") && p3Batch) {
      const rows = p3Batch.curves.batch_sweep || [];
      const yKey = activeChartMode === "latency_vs_batch" ? "p50" : "tp";
      const groups = [
        { label: "FP16 Guarded Fixed-Padded", prec: "fp16", pol: "guarded_fixed_padded" },
        { label: "FP32 Guarded Fixed-Padded", prec: "fp32", pol: "guarded_fixed_padded" },
        { label: "FP16 Dynamic Unpadded", prec: "fp16", pol: "dynamic_unpadded" },
        { label: "FP32 Dynamic Unpadded", prec: "fp32", pol: "dynamic_unpadded" },
      ];
      renderLineChart(slot, {
        title:
          yKey === "p50"
            ? "86-Face Stream P50 Latency (ms) vs. Batch Size (AdaFace IR-50)"
            : "86-Face Stream Throughput (faces/s) vs. Batch Size (AdaFace IR-50)",
        subtitle: "Dataset: NASA P1.1 8-Scene Variable-Count Stream (HIST-P3-EXP07)",
        xLabel: "Batch Size (B)",
        yLabel: yKey === "p50" ? "Stream P50 Latency (ms)" : "Throughput (faces/s)",
        series: groups.map((g) => ({
          name: g.label,
          points: rows
            .filter((r) => r.precision === g.prec && r.batch_policy === g.pol)
            .map((r) => ({
              x: r.batch_size,
              y: yKey === "p50" ? r.stream_86faces_pass_ms.p50_ms : r.throughput_faces_per_sec,
            })),
        })),
      });
      return;
    }

    if (activeChartMode === "recall_vs_nprobe" && p3Ret) {
      const rows = (p3Ret.curves.flat_ivf_pq_scaling || []).filter((r) => r.gallery_size === 100000);
      renderLineChart(slot, {
        title: "1:N Recall@1 & Recall@5 vs. nprobe (N = 100,000)",
        subtitle: "HIST-P3-EXP08 (13,233 Real LFW + Unit-Hypersphere Distractors)",
        xLabel: "nprobe",
        yLabel: "Recall",
        series: [
          {
            name: "IVF-Flat Recall@1",
            points: rows.filter((r) => r.index_type === "IVF_Flat_CPU").map((r) => ({ x: r.params.nprobe, y: r.recall_at_1 })),
          },
          {
            name: "IVF-PQ (M=32) Recall@1",
            points: rows.filter((r) => r.index_type === "IVF_PQ_ADC_CPU").map((r) => ({ x: r.params.nprobe, y: r.recall_at_1 })),
          },
          {
            name: "IVF-PQ (M=32) Recall@5",
            points: rows.filter((r) => r.index_type === "IVF_PQ_ADC_CPU").map((r) => ({ x: r.params.nprobe, y: r.recall_at_5 })),
          },
        ],
      });
      return;
    }

    if (activeChartMode === "memory_vs_gallery" && p3Ret) {
      const rows = p3Ret.curves.flat_ivf_pq_scaling || [];
      renderLineChart(slot, {
        title: "Index Memory Footprint (MiB) vs. Gallery Size (N)",
        subtitle: "ExactFlat FP32 vs. CUDA FP16 vs. IVF-PQ M=32 (64× vector compression)",
        xLabel: "Gallery Size (N)",
        yLabel: "Index Memory (MiB)",
        series: [
          {
            name: "ExactFlat CPU FP32",
            points: rows.filter((r) => r.index_type === "ExactFlat_CPU_FP32").map((r) => ({ x: r.gallery_size, y: r.memory_mib })),
          },
          {
            name: "ExactFlat CUDA FP16",
            points: rows.filter((r) => r.index_type === "ExactFlat_TORCH_CUDA_FP16").map((r) => ({ x: r.gallery_size, y: r.memory_mib })),
          },
          {
            name: "IVF-PQ (M=32, nprobe=16)",
            points: rows
              .filter((r) => r.index_type === "IVF_PQ_ADC_CPU" && r.params.nprobe === 16)
              .map((r) => ({ x: r.gallery_size, y: r.memory_mib })),
          },
        ],
      });
      return;
    }

    // Mode 12: Bar chart over selected runs
    const sel = getSelectedRuns();
    renderBarChart(slot, {
      title: "Selected Experiments — Primary Accuracy / Recall Metric",
      subtitle: "Context-specific values (observe protocol warning banner above when mixing datasets)",
      categories: sel.map((s) => s.name),
      values: sel.map(
        (s) =>
          s.metrics.test_accuracy_pct ??
          (s.metrics.f1 !== undefined ? s.metrics.f1 * 100 : null) ??
          (s.metrics.recall_at_1 !== undefined ? s.metrics.recall_at_1 * 100 : 0)
      ),
      unit: "%",
    });
  }

  function mountTwoRunComparison() {
    const slot = container.querySelector("#cmp-two-run-config-box");
    if (!slot) return;
    const wbRuns = exps.filter((e) => e.source_origin !== "historical_readonly");
    const defaultPair =
      store.exp.selectedExperimentIds.length >= 2
        ? store.exp.selectedExperimentIds.slice(0, 2)
        : wbRuns.length >= 2
        ? [wbRuns[0].experiment_id, wbRuns[1].experiment_id]
        : exps.slice(0, 2).map((x) => x.experiment_id);
    renderTwoRunConfigComparison(slot, {
      experiments: exps,
      algorithmsById: store.algorithmsById,
      selectedIds: defaultPair,
    });
  }

  // Default P-2 selector to the newest run that actually contains empirical score_distributions (if any)
  const p2Select = container.querySelector("#cmp-p2-run-select");
  const firstWithScores = exps.find((e) => Boolean(e.curves?.score_distributions?.partitions));
  if (p2Select && firstWithScores) {
    p2Select.value = firstWithScores.experiment_id;
  }

  function mountScoreExplorer() {
    const slot = container.querySelector("#cmp-p2-explorer-box");
    const selId = container.querySelector("#cmp-p2-run-select")?.value;
    const rec = exps.find((e) => e.experiment_id === selId) || exps[0];
    if (!slot || !rec) return;

    renderScoreDistributionExplorer(slot, {
      experimentId: rec.experiment_id,
      experimentName: rec.name,
      scoreDistributions: rec.curves?.score_distributions || null,
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
        if (navigateTo) navigateTo("recognition_lab");
      },
    });
  }

  p2Select?.addEventListener("change", () => mountScoreExplorer());

  container.querySelectorAll("[data-exp-cb]").forEach((cb) => {
    cb.addEventListener("change", () => {
      const id = cb.getAttribute("data-exp-cb");
      if (cb.checked) {
        if (!store.exp.selectedExperimentIds.includes(id)) store.exp.selectedExperimentIds.push(id);
      } else {
        store.exp.selectedExperimentIds = store.exp.selectedExperimentIds.filter((x) => x !== id);
      }
      renderMatrixAndWarnings();
      mountTwoRunComparison();
      if (activeChartMode === "selected_bar") renderSelectedChart();
    });
  });

  container.querySelector("#cmp-chart-mode").addEventListener("change", (e) => {
    activeChartMode = e.target.value;
    renderSelectedChart();
  });

  renderMatrixAndWarnings();
  renderSelectedChart();
  mountTwoRunComparison();
  mountScoreExplorer();
}
