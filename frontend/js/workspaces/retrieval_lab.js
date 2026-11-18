/**
 * Workspace 7 — 1:N Search & Retrieval Laboratory (`frontend/js/workspaces/retrieval_lab.js`).
 *
 * Exposes ExactFlat, IVF-Flat, IVF-PQ, and HNSW with index-specific controls, Exact-vs-ANN
 * rank comparison, and controlled gallery-size experiments (10K, 13.2K Real LFW, 50K, 100K, 1M).
 */

import { api } from "../api.js";
import { renderBarChart, renderLineChart } from "../components/charts.js";
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

export async function renderRetrievalLab(container, _navigateTo, waitJobFn) {
  const indexes = (store.registry?.algorithms || []).filter((a) => a.category === "retrieval");

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>07 · 1:N Vector Search &amp; Approximate Nearest Neighbor Laboratory</h2>
        <p class="wb-workspace-desc">
          Compare ExactFlat (CPU/CUDA FP32/FP16), IVF-Flat, IVF-PQ (ADC), and HNSW against Exact-NN ground truth.
          Synthetic distractors are used strictly for latency/memory/ANN recall scaling — never biometric accuracy claims.
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <button class="wb-btn wb-btn-sm" id="ret-run-btn" type="button" title="Build selected index on N gallery vectors and measure Recall@K, latency, and memory vs. ExactFlat">▶ Run Controlled Retrieval Experiment</button>
      </div>
    </div>

    <div class="wb-split-layout">
      <!-- Left: Index Selection, Dynamic Index-Specific Controls & Gallery Size -->
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Index &amp; Controlled Gallery Scale ${helpBadgeHtml("product_quantization")}</span></div>
          <div class="wb-panel-body">
            <div class="wb-field" data-tip-key="product_quantization">
              <label class="wb-label" for="ret-index-select">Retrieval Index Algorithm</label>
              <select id="ret-index-select" class="wb-select" data-tip-key="product_quantization">
                ${indexes
                  .map(
                    (ix) =>
                      `<option value="${esc(ix.algorithm_id)}" ${
                        ix.algorithm_id === store.exp.retrievalIndexId ? "selected" : ""
                      }>${esc(ix.name)}</option>`
                  )
                  .join("")}
              </select>
            </div>
            <div class="wb-field">
              <label class="wb-label" for="ret-gallery-size">Gallery Size (N)</label>
              <select id="ret-gallery-size" class="wb-select" title="Total enrolled gallery vectors (real LFW embeddings + unit-hypersphere distractors; HNSW capped at N <= 15,000)">
                <option value="2000" ${store.exp.retrievalGallerySize === 2000 ? "selected" : ""}>2,000 Vectors (Fast Interactive)</option>
                <option value="10000" ${store.exp.retrievalGallerySize === 10000 ? "selected" : ""}>10,000 Vectors (Supports HNSW)</option>
                <option value="50000" ${store.exp.retrievalGallerySize === 50000 ? "selected" : ""}>50,000 Vectors</option>
                <option value="100000" ${store.exp.retrievalGallerySize === 100000 ? "selected" : ""}>100,000 Vectors</option>
                <option value="1000000" ${store.exp.retrievalGallerySize === 1000000 ? "selected" : ""}>1,000,000 Vectors (Flat / IVF / IVF-PQ)</option>
              </select>
            </div>
            <div class="wb-field" style="margin-bottom:0;" data-tip-key="recall_at_k">
              <label class="wb-label" for="ret-query-count">Query Count (Q) ${helpBadgeHtml("recall_at_k")}</label>
              <input type="number" id="ret-query-count" class="wb-input wb-mono" min="20" max="500" value="${store.exp.retrievalQueryCount}" data-tip-key="recall_at_k" />
            </div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Index-Specific Parameters</span></div>
          <div class="wb-panel-body" id="ret-param-editor"></div>
        </div>
      </div>

      <!-- Right: KPIs, Historical P3 Scaling Charts & Exact-vs-ANN Rank Table -->
      <div>
        <div id="ret-disclaimer-slot"></div>
        <div class="wb-grid-6" id="ret-kpi-row" style="margin-bottom:12px;"></div>

        <div class="wb-grid-2">
          <div class="wb-panel">
            <div class="wb-panel-header"><span>Recall@K &amp; Rank Distribution</span></div>
            <div class="wb-panel-body" id="ret-recall-chart"></div>
          </div>
          <div class="wb-panel">
            <div class="wb-panel-header"><span>P3 Reference: Memory &amp; Recall@1 vs. nprobe (N=100K)</span></div>
            <div class="wb-panel-body" id="ret-p3-chart"></div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Exact-Nearest-Neighbor vs. ANN Rank Comparison (Per-Query Audit)</span></div>
          <div class="wb-panel-body" id="ret-samples-table">
            <div class="wb-help">Click "Run Controlled Retrieval Experiment" to compare ANN results against ExactFlat ground truth.</div>
          </div>
        </div>
      </div>
    </div>
  `;

  function mountIndexParams() {
    const algo = store.algorithmsById[store.exp.retrievalIndexId];
    const vals = store.exp.retrievalParamsByIndex[store.exp.retrievalIndexId] || {};
    renderParameterEditor(container.querySelector("#ret-param-editor"), algo, vals, () => {});
  }

  function renderReferenceP3Chart() {
    const p3 = (store.experiments || []).find((e) => e.experiment_id === "HIST-P3-EXP08-RETRIEVAL-SCALING");
    const rows = (p3?.curves?.flat_ivf_pq_scaling || []).filter((r) => r.gallery_size === 100000);
    const ivfRows = rows.filter((r) => r.index_type === "IVF_Flat_CPU");
    const pqRows = rows.filter((r) => r.index_type === "IVF_PQ_ADC_CPU");
    renderLineChart(container.querySelector("#ret-p3-chart"), {
      title: "Recall@1 vs. nprobe at N = 100,000 (EXP-P3-08)",
      subtitle: "IVF-Flat (196.6 MiB) vs. IVF-PQ M=32 (4.82 MiB = 64× code compression)",
      xLabel: "nprobe (Probed Voronoi Cells)",
      yLabel: "Recall@1",
      series: [
        { name: "IVF-Flat (nlist=256)", points: ivfRows.map((r) => ({ x: r.params.nprobe, y: r.recall_at_1 })) },
        { name: "IVF-PQ (M=32, 8-bit)", points: pqRows.map((r) => ({ x: r.params.nprobe, y: r.recall_at_1 })) },
      ],
    });
  }

  function renderBenchmarkResult(res) {
    if (!res) return;
    const m = res.metrics;
    container.querySelector("#ret-disclaimer-slot").innerHTML = `
      <div class="wb-banner wb-banner-warn" style="margin-bottom:10px;">
        <strong>Provenance &amp; Scope:</strong> ${esc(res.synthetic_disclaimer)}
        (${res.real_embeddings_count.toLocaleString()} real LFW embeddings + ${res.synthetic_distractors_count.toLocaleString()} unit-hypersphere vectors).
      </div>
    `;

    container.querySelector("#ret-kpi-row").innerHTML = `
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Gallery / Queries</span></div>
        <div class="wb-metric-value">${res.gallery_size.toLocaleString()}</div>
        <div class="wb-metric-prov">${res.query_count} noisy queries · ${res.embedding_dim}-D</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Index Size</span></div>
        <div class="wb-metric-value">${m.index_memory_mib} MiB</div>
        <div class="wb-metric-prov">${m.code_compression_vs_fp32}× code compression vs FP32</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Build Time</span></div>
        <div class="wb-metric-value">${m.build_time_ms} ms</div>
        <div class="wb-metric-prov">ExactFlat ref: ${m.exact_reference_build_ms} ms</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Search P50 / P95</span></div>
        <div class="wb-metric-value">${m.single_query_p50_ms.toFixed(2)} ms</div>
        <div class="wb-metric-prov">P95: ${m.single_query_p95_ms.toFixed(2)} ms (Q=1)</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Throughput</span></div>
        <div class="wb-metric-value">${m.batched_qps.toLocaleString()} QPS</div>
        <div class="wb-metric-prov">Batched (${res.query_count} queries)</div>
      </div>
      <div class="wb-metric-card">
        <div class="wb-metric-label"><span>Recall @1 / @5 / @10</span></div>
        <div class="wb-metric-value">${(m.recall_at_1 * 100).toFixed(1)}%</div>
        <div class="wb-metric-prov">R@5 ${(m.recall_at_5 * 100).toFixed(1)}% · R@10 ${(m.recall_at_10 * 100).toFixed(1)}%</div>
      </div>
    `;

    renderBarChart(container.querySelector("#ret-recall-chart"), {
      title: `${res.index_id} — Exact-NN Recall@K (N=${res.gallery_size.toLocaleString()})`,
      subtitle: `Found in Top-10: ${m.rank_distribution.found_within_returned_pct}% (Missed: ${m.rank_distribution.missed_count})`,
      categories: ["Recall@1", "Recall@5", "Recall@10"],
      values: [m.recall_at_1 * 100, m.recall_at_5 * 100, m.recall_at_10 * 100],
      unit: "%",
    });

    container.querySelector("#ret-samples-table").innerHTML = `
      <div class="wb-table-wrap">
        <table class="wb-table">
          <thead>
            <tr>
              <th>Query #</th>
              <th>Exact-NN True Top-1 ID</th>
              <th>Exact Top-1 Cosine</th>
              <th>ANN Returned Top-1 ID</th>
              <th>ANN Top-1 Score</th>
              <th>True NN Rank in ANN</th>
              <th>Rank Shift / Status</th>
            </tr>
          </thead>
          <tbody>
            ${(res.exact_vs_ann_query_samples || [])
              .map(
                (s) => `<tr>
                  <td class="wb-mono">Q_${String(s.query_index).padStart(3, "0")}</td>
                  <td class="wb-mono">#${s.exact_top1_id}</td>
                  <td class="wb-mono">${s.exact_top1_cosine}</td>
                  <td class="wb-mono">#${s.ann_top1_id}</td>
                  <td class="wb-mono">${s.ann_top1_score}</td>
                  <td class="wb-mono">${s.exact_top1_rank_in_ann ?? "Missed (>10)"}</td>
                  <td><span class="wb-badge ${s.missed_in_top10 ? "" : "wb-badge-perf"}">${
                  s.missed_in_top10
                    ? "MISSED IN TOP-10"
                    : s.rank_change === 0
                    ? "EXACT RANK #1 (+0)"
                    : `SHIFTED +${s.rank_change}`
                }</span></td>
                </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  container.querySelector("#ret-index-select").addEventListener("change", (e) => {
    store.exp.retrievalIndexId = e.target.value;
    mountIndexParams();
    store.notify("config_change");
  });
  container.querySelector("#ret-gallery-size").addEventListener("change", (e) => {
    store.exp.retrievalGallerySize = parseInt(e.target.value, 10);
  });
  container.querySelector("#ret-query-count").addEventListener("change", (e) => {
    const check = sanitizeNumericInput(e.target.value, { min: 20, max: 500, isInt: true, fallback: store.exp.retrievalQueryCount });
    if (check.invalid || check.clamped) {
      showToast(`Query Count (Q): ${check.reason}`, true);
    }
    store.exp.retrievalQueryCount = check.value;
    e.target.value = String(check.value);
  });

  const retRunBtn = container.querySelector("#ret-run-btn");
  retRunBtn.addEventListener("click", () =>
    withBusyButton(retRunBtn, async () => {
      try {
        const idxId = store.exp.retrievalIndexId;
        const sub = await api.submitJob({
          job_type: "retrieval_benchmark",
          name: `1:N Retrieval (${idxId}, N=${store.exp.retrievalGallerySize.toLocaleString()})`,
          config: {
            index_id: idxId,
            index_params: store.exp.retrievalParamsByIndex[idxId] || {},
            gallery_size: store.exp.retrievalGallerySize,
            num_queries: store.exp.retrievalQueryCount,
            embedding_dim: 512,
          },
          save_experiment: true,
        });
        showToast(`Queued retrieval benchmark ${sub.job_id}`);
        const done = await waitJobFn(sub.job_id);
        if (done?.result && container.querySelector("#ret-kpi-row")) {
          store.exp.lastRetrievalBenchmark = done.result;
          renderBenchmarkResult(done.result);
        }
      } catch (err) {
        showToast(err.message, true);
      }
    }, "Benchmarking…")
  );

  mountIndexParams();
  renderReferenceP3Chart();
  if (store.exp.lastRetrievalBenchmark) renderBenchmarkResult(store.exp.lastRetrievalBenchmark);
}
