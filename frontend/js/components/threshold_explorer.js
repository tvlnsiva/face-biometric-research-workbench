/**
 * P-2 — Interactive Genuine / Impostor Score Distribution & Threshold Explorer
 * (`frontend/js/components/threshold_explorer.js`).
 *
 * Pure mathematical functions (`computeEmpiricalOperatingPoint`, `computeScoreHistograms`,
 * `computeEmpiricalRocCurve`, `classifyPairsAtThreshold`) are exported standalone with zero DOM
 * dependency for deterministic Node.js unit testing and exact parity checks against Python.
 */

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Computes exact empirical TP, FN, FP, TN and TAR, FRR, FAR, TNR directly from actual genuine
 * and impostor score arrays.
 *
 * Zero-denominator populations return `null` (NEVER `NaN` or `Infinity`).
 */
export function computeEmpiricalOperatingPoint(
  genuineScores = [],
  impostorScores = [],
  threshold = 0.2169,
  higherIsMoreSimilar = true
) {
  const tau = Number(threshold);
  const nGen = Array.isArray(genuineScores) ? genuineScores.length : 0;
  const nImp = Array.isArray(impostorScores) ? impostorScores.length : 0;

  let tp = 0;
  for (let i = 0; i < nGen; i++) {
    const s = Number(genuineScores[i]);
    if (higherIsMoreSimilar ? s >= tau : s <= tau) tp += 1;
  }
  const fn = nGen - tp;

  let fp = 0;
  for (let i = 0; i < nImp; i++) {
    const s = Number(impostorScores[i]);
    if (higherIsMoreSimilar ? s >= tau : s <= tau) fp += 1;
  }
  const tn = nImp - fp;

  const tar = nGen > 0 ? tp / nGen : null;
  const frr = nGen > 0 ? fn / nGen : null;
  const far = nImp > 0 ? fp / nImp : null;
  const tnr = nImp > 0 ? tn / nImp : null;

  return {
    decision_threshold: Number(tau.toFixed(6)),
    higher_is_more_similar: Boolean(higherIsMoreSimilar),
    decision_rule: higherIsMoreSimilar
      ? "score >= threshold → MATCH; score < threshold → NON-MATCH"
      : "score <= threshold → MATCH; score > threshold → NON-MATCH",
    genuine_count: nGen,
    impostor_count: nImp,
    total_pairs: nGen + nImp,
    tp,
    fn,
    fp,
    tn,
    tar: tar !== null ? Number(tar.toFixed(6)) : null,
    frr: frr !== null ? Number(frr.toFixed(6)) : null,
    far: far !== null ? Number(far.toFixed(6)) : null,
    tnr: tnr !== null ? Number(tnr.toFixed(6)) : null,
  };
}

/**
 * Computes exact genuine and impostor histogram bins over `[minEdge, maxEdge]` without
 * fabricating synthetic scores. Guarantees exact population sum conservation.
 */
export function computeScoreHistograms(
  genuineScores = [],
  impostorScores = [],
  { binCount = 30, minScore = null, maxScore = null } = {}
) {
  const nBins = Math.max(4, Math.min(80, Math.round(binCount)));
  const all = [...(genuineScores || []), ...(impostorScores || [])].map(Number).filter(Number.isFinite);

  let lo = minScore !== null && Number.isFinite(minScore) ? Number(minScore) : all.length ? Math.min(...all) : -0.2;
  let hi = maxScore !== null && Number.isFinite(maxScore) ? Number(maxScore) : all.length ? Math.max(...all) : 1.0;

  if (Math.abs(hi - lo) < 1e-6) {
    lo -= 0.1;
    hi += 0.1;
  } else {
    const pad = (hi - lo) * 0.02;
    lo -= pad;
    hi += pad;
  }

  const width = (hi - lo) / nBins;
  const genuineBins = new Array(nBins).fill(0);
  const impostorBins = new Array(nBins).fill(0);

  const assignBin = (v) => {
    const idx = Math.floor((Number(v) - lo) / width);
    return Math.max(0, Math.min(nBins - 1, idx));
  };

  for (const s of genuineScores || []) {
    if (Number.isFinite(Number(s))) genuineBins[assignBin(s)] += 1;
  }
  for (const s of impostorScores || []) {
    if (Number.isFinite(Number(s))) impostorBins[assignBin(s)] += 1;
  }

  const edges = [];
  for (let b = 0; b <= nBins; b++) {
    edges.push(Number((lo + b * width).toFixed(6)));
  }

  return {
    binCount: nBins,
    minEdge: Number(lo.toFixed(6)),
    maxEdge: Number(hi.toFixed(6)),
    binWidth: Number(width.toFixed(6)),
    edges,
    genuineBins,
    impostorBins,
    genuineSum: genuineBins.reduce((a, b) => a + b, 0),
    impostorSum: impostorBins.reduce((a, b) => a + b, 0),
    maxBinHeight: Math.max(1, ...genuineBins, ...impostorBins),
  };
}

/**
 * Computes exact empirical ROC/DET operating points directly from unique score cutoffs
 * (plus extreme endpoints so `FAR=0` and `FAR=1` are represented without smoothing).
 */
export function computeEmpiricalRocCurve(genuineScores = [], impostorScores = [], higherIsMoreSimilar = true) {
  const all = [...(genuineScores || []), ...(impostorScores || [])].map(Number).filter(Number.isFinite);
  if (all.length === 0 || genuineScores.length === 0 || impostorScores.length === 0) {
    return { points: [], eerPoint: null };
  }

  const uniqueSorted = Array.from(new Set(all)).sort((a, b) => a - b);
  // Downsample evenly to at most 80 empirical thresholds while keeping exact min/max & impostor tail
  let candidateThresholds = [];
  const loBound = uniqueSorted[0] - 1e-5;
  const hiBound = uniqueSorted[uniqueSorted.length - 1] + 1e-5;
  if (uniqueSorted.length <= 75) {
    candidateThresholds = [hiBound, ...uniqueSorted.slice().reverse(), loBound];
  } else {
    candidateThresholds.push(hiBound);
    // Keep top 20 highest scores (low-FAR security tail!) + 55 evenly spaced scores
    const tail = uniqueSorted.slice(-20).reverse();
    const rest = uniqueSorted.slice(0, -20);
    const step = Math.max(1, Math.floor(rest.length / 55));
    const sampledRest = [];
    for (let i = rest.length - 1; i >= 0; i -= step) {
      sampledRest.push(rest[i]);
    }
    candidateThresholds.push(...tail, ...sampledRest, loBound);
  }

  const points = candidateThresholds.map((tau) => {
    const op = computeEmpiricalOperatingPoint(genuineScores, impostorScores, tau, higherIsMoreSimilar);
    return {
      threshold: op.decision_threshold,
      far: op.far ?? 0,
      tar: op.tar ?? 0,
      frr: op.frr ?? 0,
      tp: op.tp,
      fp: op.fp,
      fn: op.fn,
      tn: op.tn,
    };
  });

  // Sort monotonically by FAR ascending, then TAR ascending
  points.sort((a, b) => (a.far !== b.far ? a.far - b.far : a.tar - b.tar));

  let eerPoint = points[0];
  let bestGap = Infinity;
  for (const pt of points) {
    const gap = Math.abs(pt.far - pt.frr);
    if (gap < bestGap) {
      bestGap = gap;
      eerPoint = pt;
    }
  }

  return { points, eerPoint };
}

/**
 * Partitions anonymized pair records into False Accepts (FP), False Rejects (FN),
 * True Accepts (TP), and True Rejects (TN) at `threshold`.
 */
export function classifyPairsAtThreshold(pairRefs = [], pairScores = [], threshold = 0.2169, higherIsMoreSimilar = true) {
  const tau = Number(threshold);
  const falseAccepts = [];
  const falseRejects = [];
  const trueAccepts = [];
  const trueRejects = [];

  const n = Math.min(pairRefs.length, pairScores.length);
  for (let i = 0; i < n; i++) {
    const ref = pairRefs[i];
    const sc = Number(pairScores[i]);
    const predMatch = higherIsMoreSimilar ? sc >= tau : sc <= tau;
    const isGen = Boolean(ref.is_same_subject);
    const entry = {
      ...ref,
      score: sc,
      margin: Number((higherIsMoreSimilar ? sc - tau : tau - sc).toFixed(6)),
    };

    if (isGen && predMatch) trueAccepts.push(entry);
    else if (isGen && !predMatch) falseRejects.push(entry);
    else if (!isGen && predMatch) falseAccepts.push(entry);
    else trueRejects.push(entry);
  }

  // Sort errors so worst / closest-to-boundary errors appear at top
  falseAccepts.sort((a, b) => b.score - a.score);
  falseRejects.sort((a, b) => a.score - b.score);
  trueAccepts.sort((a, b) => a.score - b.score);
  trueRejects.sort((a, b) => b.score - a.score);

  return { falseAccepts, falseRejects, trueAccepts, trueRejects };
}

function fmtRateAndCount(rate, num, den, labelDen) {
  if (rate === null || den === 0) {
    return `N/A (0 / 0 ${labelDen})`;
  }
  return `${(rate * 100).toFixed(2)}% (${num} / ${den})`;
}

/**
 * Interactive Genuine / Impostor Score Distribution, Draggable Threshold & Synchronized ROC/DET Explorer.
 */
export function renderScoreDistributionExplorer(
  container,
  {
    experimentId = "LIVE-PIPELINE",
    experimentName = "Verification Evaluation",
    scoreDistributions = null,
    onInspectPair = null,
    sanitizeNumericFn = null,
    onToast = null,
  }
) {
  if (!container) return;

  // Requirement P-2 Acceptance #10 & #11: Clear explanatory state when score arrays are unavailable
  if (!scoreDistributions || !scoreDistributions.partitions || Object.keys(scoreDistributions.partitions).length === 0) {
    container.innerHTML = `
      <div class="wb-banner" data-role="p2-unavailable-state">
        <strong>Score Distribution Arrays Unavailable for <code>${esc(experimentId)}</code>:</strong>
        This legacy or non-verification experiment record does not contain raw genuine/impostor pair-score arrays
        (only summary metrics or pre-WB3 curve points are stored; <em>no synthetic scores are ever fabricated</em>).
        Select or execute a WB3 Verification Pipeline run (e.g., on <code>lfw_research_split</code> or <code>nasa_p11_verification</code>
        in Workspace <code>08 · Pipeline Builder</code>) to explore empirical score distributions, operating points, and error pairs.
      </div>
    `;
    return;
  }

  const partitions = scoreDistributions.partitions;
  const partKeys = Object.keys(partitions);
  let activePartKey = partitions[scoreDistributions.default_partition] ? scoreDistributions.default_partition : partKeys[0];
  let curveMode = "roc"; // "roc" | "det"
  let activeOutcomeTab = "fp"; // "fp" | "fn" | "tp" | "tn"
  let binCount = 30;

  let curPart = partitions[activePartKey];
  let activeTau = Number(curPart.default_threshold ?? scoreDistributions.configured_threshold ?? 0.2169);

  // Precompute static per-partition structures once when partition changes (fast drag!)
  let hist = null;
  let rocData = null;

  function recomputePartitionStructures() {
    curPart = partitions[activePartKey];
    hist = computeScoreHistograms(curPart.genuine_scores, curPart.impostor_scores, { binCount });
    rocData = computeEmpiricalRocCurve(
      curPart.genuine_scores,
      curPart.impostor_scores,
      curPart.higher_is_more_similar !== false
    );
  }
  recomputePartitionStructures();

  const W_HIST = 520;
  const H_HIST = 248;
  const PAD_H = { top: 24, right: 18, bottom: 38, left: 46 };
  const PLOT_W = W_HIST - PAD_H.left - PAD_H.right;
  const PLOT_H = H_HIST - PAD_H.top - PAD_H.bottom;

  const scoreToX = (s) => {
    const clamped = Math.max(hist.minEdge, Math.min(hist.maxEdge, Number(s)));
    return PAD_H.left + ((clamped - hist.minEdge) / Math.max(1e-9, hist.maxEdge - hist.minEdge)) * PLOT_W;
  };
  const xToScore = (svgX) => {
    const frac = Math.max(0, Math.min(1, (svgX - PAD_H.left) / PLOT_W));
    return Number((hist.minEdge + frac * (hist.maxEdge - hist.minEdge)).toFixed(5));
  };

  function buildFullMountHtml() {
    const preTh = scoreDistributions.pre_established_thresholds || {};
    const recShaShort = scoreDistributions.recognizer_sha256 ? `${scoreDistributions.recognizer_sha256.slice(0, 16)}…` : "N/A";

    const barsSvg = [];
    const bwPx = PLOT_W / hist.binCount;
    for (let b = 0; b < hist.binCount; b++) {
      const bx = PAD_H.left + b * bwPx;
      const ic = hist.impostorBins[b];
      const gc = hist.genuineBins[b];
      const edgeLo = hist.edges[b].toFixed(3);
      const edgeHi = hist.edges[b + 1].toFixed(3);
      if (ic > 0) {
        const ih = (ic / hist.maxBinHeight) * PLOT_H;
        barsSvg.push(
          `<rect x="${(bx + 0.5).toFixed(1)}" y="${(PAD_H.top + PLOT_H - ih).toFixed(1)}" width="${Math.max(
            1,
            bwPx - 1
          ).toFixed(1)}" height="${ih.toFixed(1)}" fill="#dc2626" fill-opacity="0.40" stroke="#dc2626" stroke-width="0.7">
            <title>Impostor bin [${edgeLo}, ${edgeHi}): ${ic} pairs</title>
          </rect>`
        );
      }
      if (gc > 0) {
        const gh = (gc / hist.maxBinHeight) * PLOT_H;
        barsSvg.push(
          `<rect x="${(bx + 0.5).toFixed(1)}" y="${(PAD_H.top + PLOT_H - gh).toFixed(1)}" width="${Math.max(
            1,
            bwPx - 1
          ).toFixed(1)}" height="${gh.toFixed(1)}" fill="#16a34a" fill-opacity="0.42" stroke="#16a34a" stroke-width="0.7">
            <title>Genuine bin [${edgeLo}, ${edgeHi}): ${gc} pairs</title>
          </rect>`
        );
      }
    }

    let xTicksSvg = "";
    for (let t = 0; t <= 4; t++) {
      const xv = hist.minEdge + (t / 4) * (hist.maxEdge - hist.minEdge);
      const xp = PAD_H.left + (t / 4) * PLOT_W;
      xTicksSvg += `<line x1="${xp}" y1="${PAD_H.top}" x2="${xp}" y2="${PAD_H.top + PLOT_H}" stroke="var(--border)" stroke-dasharray="3,3" />`;
      xTicksSvg += `<text x="${xp}" y="${H_HIST - 18}" text-anchor="middle" font-size="10" fill="var(--text-muted)">${xv.toFixed(3)}</text>`;
    }

    let yTicksSvg = "";
    for (let t = 0; t <= 3; t++) {
      const yv = Math.round(hist.maxBinHeight * (1 - t / 3));
      const yp = PAD_H.top + (t / 3) * PLOT_H;
      yTicksSvg += `<line x1="${PAD_H.left}" y1="${yp}" x2="${W_HIST - PAD_H.right}" y2="${yp}" stroke="var(--border)" stroke-dasharray="3,3" />`;
      yTicksSvg += `<text x="${PAD_H.left - 6}" y="${yp + 3.5}" text-anchor="end" font-size="10" fill="var(--text-muted)">${yv}</text>`;
    }

    return `
      <div data-role="p2-explorer-root">
        <!-- Provenance & Partition Safeguard Header -->
        <div style="padding:8px 10px;background:var(--bg-metric);border:1px solid var(--border);border-radius:5px;margin-bottom:8px;font-size:11.5px;" data-role="p2-provenance-bar">
          <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;margin-bottom:4px;">
            <span><strong>Run ID:</strong> <code class="wb-mono">${esc(
              scoreDistributions.experiment_id || experimentId
            )}</code> (${esc(experimentName)})</span>
            <span><strong>Recognizer:</strong> <code class="wb-mono">${esc(scoreDistributions.recognizer_id)}</code> · SHA-256: <code class="wb-mono">${esc(
      recShaShort
    )}</code></span>
          </div>
          <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;">
            <span><strong>Dataset &amp; Protocol:</strong> ${esc(scoreDistributions.dataset_name)} · ${esc(
      scoreDistributions.evaluation_protocol
    )}</span>
            <span><strong>Metric &amp; Rule:</strong> <code class="wb-mono">${esc(scoreDistributions.metric)}</code> (${esc(
      curPart.decision_rule
    )})</span>
          </div>
          <div class="wb-muted-xs" style="margin-top:3px;"><strong>Identity Partition Safeguard:</strong> ${esc(
            curPart.identity_partition_info
          )}</div>
        </div>

        <!-- Partition Policy Warning / Status Slot -->
        <div data-role="p2-protocol-banner" style="margin-bottom:8px;"></div>

        <!-- Partition, Threshold Controls & Preset Bar -->
        <div class="wb-panel" style="margin-bottom:10px;background:var(--bg-surface);">
          <div class="wb-panel-body" style="display:grid;grid-template-columns:repeat(auto-fit, minmax(210px, 1fr));gap:10px;align-items:end;">
            <div>
              <label class="wb-label" for="p2-part-select">Evaluated Partition (Never Mixed)</label>
              <select id="p2-part-select" class="wb-select">
                ${partKeys
                  .map(
                    (pk) =>
                      `<option value="${esc(pk)}" ${pk === activePartKey ? "selected" : ""}>${esc(
                        partitions[pk].split_name
                      )}</option>`
                  )
                  .join("")}
              </select>
            </div>
            <div>
              <div class="wb-field-top">
                <label class="wb-label" for="p2-tau-slider">Interactive Decision Threshold (τ)</label>
                <span class="wb-mono" data-role="p2-tau-readout">${activeTau.toFixed(5)}</span>
              </div>
              <div class="wb-slider-row">
                <input type="range" id="p2-tau-slider" min="${hist.minEdge}" max="${hist.maxEdge}" step="0.001" value="${activeTau}" aria-label="Score distribution decision threshold slider" />
                <input type="number" id="p2-tau-num" class="wb-input wb-mono" min="-2.5" max="1.5" step="0.001" value="${activeTau.toFixed(
                  5
                )}" aria-label="Decision threshold numeric input" />
              </div>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:5px;align-items:center;">
              ${
                preTh.calibrated_best_acc !== undefined
                  ? `<button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-preset-tau="${preTh.calibrated_best_acc}" title="Apply fixed threshold calibrated on Calibration Folds 1–5">τ_calib Acc (${Number(
                      preTh.calibrated_best_acc
                    ).toFixed(4)})</button>`
                  : ""
              }
              ${
                preTh.calibrated_far_1e3 !== undefined
                  ? `<button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-preset-tau="${preTh.calibrated_far_1e3}" title="Apply Folds 1–5 FAR<=1e-3 calibrated threshold">τ_calib FAR≤10⁻³ (${Number(
                      preTh.calibrated_far_1e3
                    ).toFixed(4)})</button>`
                  : ""
              }
              ${
                rocData.eerPoint
                  ? `<button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-preset-tau="${rocData.eerPoint.threshold}" title="Jump to empirical FAR≈FRR crossover on active partition">Empirical EER (${rocData.eerPoint.threshold.toFixed(
                      4
                    )})</button>`
                  : ""
              }
              <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="export-op-json" title="Download current threshold operating-point metrics and provenance as JSON">Export OP JSON</button>
            </div>
          </div>
        </div>

        <!-- Empirical Operating-Point Metrics Grid (Counts + Rates) -->
        <div class="wb-grid-6" data-role="p2-metrics-grid" style="margin-bottom:10px;"></div>

        <!-- Dual Synchronized Charts: Left = Draggable Score Histogram, Right = Clickable Empirical ROC / DET -->
        <div class="wb-grid-2" style="margin-bottom:10px;">
          <div class="wb-panel" style="margin-bottom:0;">
            <div class="wb-panel-header">
              <span>Genuine vs. Impostor Score Distribution (${hist.binCount} Bins: [${hist.minEdge.toFixed(
      3
    )}, ${hist.maxEdge.toFixed(3)}])</span>
              <span class="wb-muted-xs">Drag vertical τ line</span>
            </div>
            <div class="wb-panel-body">
              <svg viewBox="0 0 ${W_HIST} ${H_HIST}" data-role="p2-hist-svg" style="width:100%;height:auto;display:block;user-select:none;touch-action:none;cursor:ew-resize;">
                ${yTicksSvg}
                ${xTicksSvg}
                <rect x="${PAD_H.left}" y="${PAD_H.top}" width="${PLOT_W}" height="${PLOT_H}" fill="none" stroke="var(--border-strong)" />
                ${barsSvg.join("")}
                <!-- Draggable Threshold Cursor Group -->
                <g data-role="p2-cursor-group">
                  <line data-role="p2-cursor-line" x1="${scoreToX(activeTau).toFixed(1)}" y1="${PAD_H.top - 4}" x2="${scoreToX(
      activeTau
    ).toFixed(1)}" y2="${PAD_H.top + PLOT_H + 4}" stroke="#0f172a" stroke-width="2.4" />
                  <rect data-role="p2-cursor-pill" x="${(scoreToX(activeTau) - 34).toFixed(1)}" y="4" width="68" height="16" rx="3" fill="#0f172a" />
                  <text data-role="p2-cursor-text" x="${scoreToX(activeTau).toFixed(1)}" y="15" text-anchor="middle" font-size="9.5" font-weight="700" fill="#ffffff">τ=${activeTau.toFixed(
      3
    )}</text>
                </g>
                <text x="${PAD_H.left + PLOT_W / 2}" y="${H_HIST - 3}" text-anchor="middle" font-size="10.5" fill="var(--text-secondary)">${esc(
      scoreDistributions.metric
    )} Similarity Score (NON-MATCH ← τ → MATCH)</text>
                <text x="12" y="${PAD_H.top + PLOT_H / 2}" transform="rotate(-90 12 ${
      PAD_H.top + PLOT_H / 2
    })" text-anchor="middle" font-size="10.5" fill="var(--text-secondary)">Pair Count</text>
              </svg>
              <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font-size:11px;margin-top:4px;">
                <div>
                  <span class="wb-legend-item"><span class="wb-swatch" style="background:#16a34a"></span>Genuine Pairs (N=${
                    hist.genuineSum
                  })</span>
                  <span class="wb-legend-item" style="margin-left:10px;"><span class="wb-swatch" style="background:#dc2626"></span>Impostor Pairs (N=${
                    hist.impostorSum
                  })</span>
                </div>
                <span class="wb-muted-xs wb-mono">Raw geometric similarity — NOT probability</span>
              </div>
            </div>
          </div>

          <div class="wb-panel" style="margin-bottom:0;">
            <div class="wb-panel-header">
              <span>Synchronized Empirical Operating Curve (Click Point to Set τ)</span>
              <select id="p2-curve-mode" class="wb-select" style="width:185px;" aria-label="Select ROC or DET operating curve">
                <option value="roc" ${curveMode === "roc" ? "selected" : ""}>ROC (TAR vs. FAR)</option>
                <option value="det" ${curveMode === "det" ? "selected" : ""}>DET (FRR vs. FAR)</option>
              </select>
            </div>
            <div class="wb-panel-body" data-role="p2-roc-box"></div>
          </div>
        </div>

        <!-- Threshold Error & Outcome Pair Inspector -->
        <div class="wb-panel" style="margin-bottom:0;">
          <div class="wb-panel-header" style="flex-wrap:wrap;gap:6px;">
            <span>Threshold Pair Outcome &amp; Error Inspector (Anonymized Manifest Refs)</span>
            <div style="display:flex;gap:4px;flex-wrap:wrap;" data-role="p2-outcome-tabs"></div>
          </div>
          <div class="wb-panel-body" data-role="p2-pairs-box"></div>
        </div>
      </div>
    `;
  }

  function renderDynamicSubviews() {
    const op = computeEmpiricalOperatingPoint(
      curPart.genuine_scores,
      curPart.impostor_scores,
      activeTau,
      curPart.higher_is_more_similar !== false
    );

    // 1. Update histogram cursor SVG coordinates without rebuilding histogram bars!
    const cx = scoreToX(activeTau);
    const cLine = container.querySelector('[data-role="p2-cursor-line"]');
    const cPill = container.querySelector('[data-role="p2-cursor-pill"]');
    const cText = container.querySelector('[data-role="p2-cursor-text"]');
    if (cLine) {
      cLine.setAttribute("x1", cx.toFixed(1));
      cLine.setAttribute("x2", cx.toFixed(1));
    }
    if (cPill) cPill.setAttribute("x", (cx - 34).toFixed(1));
    if (cText) {
      cText.setAttribute("x", cx.toFixed(1));
      cText.textContent = `τ=${activeTau.toFixed(3)}`;
    }

    const readout = container.querySelector('[data-role="p2-tau-readout"]');
    if (readout) readout.textContent = activeTau.toFixed(5);
    const slider = container.querySelector("#p2-tau-slider");
    if (slider && document.activeElement !== slider) slider.value = String(activeTau);
    const numInp = container.querySelector("#p2-tau-num");
    if (numInp && document.activeElement !== numInp) numInp.value = activeTau.toFixed(5);

    // 2. Protocol & Split Safeguard Banner (Requirement P-2.F)
    const bannerSlot = container.querySelector('[data-role="p2-protocol-banner"]');
    if (bannerSlot) {
      const preCalibAcc = scoreDistributions.pre_established_thresholds?.calibrated_best_acc;
      const preCalib1e3 = scoreDistributions.pre_established_thresholds?.calibrated_far_1e3;
      const matchesPreCalib =
        (preCalibAcc !== undefined && Math.abs(activeTau - preCalibAcc) < 1e-4) ||
        (preCalib1e3 !== undefined && Math.abs(activeTau - preCalib1e3) < 1e-4);

      if (curPart.is_smoke_test) {
        bannerSlot.innerHTML = `<div class="wb-banner wb-banner-warn" style="margin-bottom:0;">
          ⚠ <strong>Integration Smoke-Test Split Only:</strong> Scores belong to the 120-pair NASA public-domain smoke test. Never mix with LFW or cite as general recognition accuracy.
        </div>`;
      } else if (curPart.is_calibration) {
        bannerSlot.innerHTML = `<div class="wb-banner" style="margin-bottom:0;">
          ℹ <strong>Calibration Partition Measurements (Folds 1–5):</strong> Metrics below reflect calibration-split pairs ONLY (suitable for selecting operating threshold τ prior to single-pass Held-Out Test evaluation).
        </div>`;
      } else if (curPart.is_held_out_test) {
        if (matchesPreCalib) {
          bannerSlot.innerHTML = `<div class="wb-banner wb-banner-match" style="margin-bottom:0;">
            ✓ <strong>Fixed Pre-Established Calibration Threshold Active:</strong> Threshold τ=${activeTau.toFixed(
              5
            )} was determined exclusively on Calibration Folds 1–5 before Held-Out Test evaluation (unbiased held-out estimate).
          </div>`;
        } else {
          bannerSlot.innerHTML = `<div class="wb-banner wb-banner-warn" style="margin-bottom:0;">
            ⚠ <strong>Exploratory Post-Hoc Threshold on Held-Out Test Split (Folds 6–10):</strong> Threshold τ=${activeTau.toFixed(
              5
            )} was adjusted interactively away from pre-established Folds 1–5 calibration threshold (${
              preCalibAcc !== undefined ? Number(preCalibAcc).toFixed(5) : "N/A"
            }). Do NOT report post-hoc tuned test operating points as unbiased benchmark estimates.
          </div>`;
        }
      }
    }

    // 3. Empirical Operating-Point Metrics Grid (Counts + Percentages)
    const mGrid = container.querySelector('[data-role="p2-metrics-grid"]');
    if (mGrid) {
      mGrid.innerHTML = `
        <div class="wb-metric-card">
          <div class="wb-metric-label"><span>Evaluated Pairs</span></div>
          <div class="wb-metric-value" style="font-size:15px;">${op.total_pairs}</div>
          <div class="wb-metric-prov">${op.genuine_count} Genuine · ${op.impostor_count} Impostor</div>
        </div>
        <div class="wb-metric-card">
          <div class="wb-metric-label"><span>TAR (True Accept)</span></div>
          <div class="wb-metric-value" style="font-size:15px;color:#16a34a;">${op.tar !== null ? (op.tar * 100).toFixed(2) + "%" : "N/A"}</div>
          <div class="wb-metric-prov">TP = ${op.tp} / ${op.genuine_count} genuine</div>
        </div>
        <div class="wb-metric-card">
          <div class="wb-metric-label"><span>FRR (False Reject)</span></div>
          <div class="wb-metric-value" style="font-size:15px;color:#ea580c;">${op.frr !== null ? (op.frr * 100).toFixed(2) + "%" : "N/A"}</div>
          <div class="wb-metric-prov">FN = ${op.fn} / ${op.genuine_count} genuine</div>
        </div>
        <div class="wb-metric-card">
          <div class="wb-metric-label"><span>FAR (False Accept)</span></div>
          <div class="wb-metric-value" style="font-size:15px;color:#dc2626;">${op.far !== null ? (op.far * 100).toFixed(2) + "%" : "N/A"}</div>
          <div class="wb-metric-prov">FP = ${op.fp} / ${op.impostor_count} impostor</div>
        </div>
        <div class="wb-metric-card">
          <div class="wb-metric-label"><span>TNR (True Reject)</span></div>
          <div class="wb-metric-value" style="font-size:15px;">${op.tnr !== null ? (op.tnr * 100).toFixed(2) + "%" : "N/A"}</div>
          <div class="wb-metric-prov">TN = ${op.tn} / ${op.impostor_count} impostor</div>
        </div>
        <div class="wb-metric-card">
          <div class="wb-metric-label"><span>Active Cutoff τ</span></div>
          <div class="wb-metric-value wb-mono" style="font-size:15px;">${op.decision_threshold.toFixed(4)}</div>
          <div class="wb-metric-prov">${esc(scoreDistributions.metric)} (${op.higher_is_more_similar ? "≥ τ Match" : "≤ τ Match"})</div>
        </div>
      `;
    }

    // 4. Synchronized Empirical ROC / DET SVG with Clickable Operating Points
    const rocBox = container.querySelector('[data-role="p2-roc-box"]');
    if (rocBox) {
      const pts = rocData.points || [];
      if (!pts.length) {
        rocBox.innerHTML = `<div class="wb-help">Both genuine and impostor pairs are required to plot ROC/DET curves.</div>`;
      } else {
        const RW = 520;
        const RH = 248;
        const RP = { top: 24, right: 20, bottom: 38, left: 48 };
        const pw = RW - RP.left - RP.right;
        const ph = RH - RP.top - RP.bottom;
        const rx = (farV) => RP.left + Math.max(0, Math.min(1, farV)) * pw;
        const ry = (yv) => RP.top + ph - Math.max(0, Math.min(1, yv)) * ph;

        let gridR = "";
        for (let i = 0; i <= 4; i++) {
          const v = i / 4;
          const gx = rx(v);
          const gy = ry(v);
          gridR += `<line x1="${gx}" y1="${RP.top}" x2="${gx}" y2="${RP.top + ph}" stroke="var(--border)" stroke-dasharray="3,3" />`;
          gridR += `<line x1="${RP.left}" y1="${gy}" x2="${RW - RP.right}" y2="${gy}" stroke="var(--border)" stroke-dasharray="3,3" />`;
          gridR += `<text x="${gx}" y="${RH - 18}" text-anchor="middle" font-size="10" fill="var(--text-muted)">${v.toFixed(2)}</text>`;
          gridR += `<text x="${RP.left - 6}" y="${gy + 3.5}" text-anchor="end" font-size="10" fill="var(--text-muted)">${v.toFixed(2)}</text>`;
        }

        const stepPath = pts
          .map((p, idx) => {
            const yVal = curveMode === "roc" ? p.tar : p.frr;
            return `${idx === 0 ? "M" : "L"}${rx(p.far).toFixed(1)},${ry(yVal).toFixed(1)}`;
          })
          .join(" ");

        const circlesSvg = pts
          .map((p, idx) => {
            const yVal = curveMode === "roc" ? p.tar : p.frr;
            return `<circle cx="${rx(p.far).toFixed(1)}" cy="${ry(yVal).toFixed(1)}" r="4.2" fill="#1d4ed8" style="cursor:pointer;" data-roc-idx="${idx}">
              <title>Empirical Point: τ=${p.threshold.toFixed(4)} · FAR=${(p.far * 100).toFixed(2)}% (FP=${p.fp}) · ${
              curveMode === "roc" ? `TAR=${(p.tar * 100).toFixed(2)}%` : `FRR=${(p.frr * 100).toFixed(2)}%`
            } — Click to apply τ</title>
            </circle>`;
          })
          .join("");

        const curX = rx(op.far ?? 0);
        const curY = ry(curveMode === "roc" ? op.tar ?? 0 : op.frr ?? 0);

        rocBox.innerHTML = `
          <svg viewBox="0 0 ${RW} ${RH}" style="width:100%;height:auto;display:block;">
            ${gridR}
            <rect x="${RP.left}" y="${RP.top}" width="${pw}" height="${ph}" fill="none" stroke="var(--border-strong)" />
            <line x1="${rx(0)}" y1="${ry(curveMode === "roc" ? 0 : 1)}" x2="${rx(1)}" y2="${ry(
          curveMode === "roc" ? 1 : 0
        )}" stroke="var(--text-muted)" stroke-dasharray="4,4" stroke-width="1" />
            <path d="${stepPath}" fill="none" stroke="#1d4ed8" stroke-width="2" />
            ${circlesSvg}
            <!-- Synchronized Active Operating-Point Crosshair -->
            <line x1="${curX.toFixed(1)}" y1="${RP.top}" x2="${curX.toFixed(1)}" y2="${RP.top + ph}" stroke="#dc2626" stroke-dasharray="2,2" />
            <line x1="${RP.left}" y1="${curY.toFixed(1)}" x2="${RW - RP.right}" y2="${curY.toFixed(1)}" stroke="#dc2626" stroke-dasharray="2,2" />
            <circle cx="${curX.toFixed(1)}" cy="${curY.toFixed(1)}" r="6.5" fill="#dc2626" stroke="#ffffff" stroke-width="2" data-role="p2-active-roc-marker">
              <title>Active τ=${op.decision_threshold.toFixed(4)} · FAR=${fmtRateAndCount(
          op.far,
          op.fp,
          op.impostor_count,
          "imp"
        )} · ${
          curveMode === "roc"
            ? `TAR=${fmtRateAndCount(op.tar, op.tp, op.genuine_count, "gen")}`
            : `FRR=${fmtRateAndCount(op.frr, op.fn, op.genuine_count, "gen")}`
        }</title>
            </circle>
            <text x="${RP.left + pw / 2}" y="${RH - 3}" text-anchor="middle" font-size="10.5" fill="var(--text-secondary)">Empirical False Accept Rate (FAR)</text>
            <text x="12" y="${RP.top + ph / 2}" transform="rotate(-90 12 ${
          RP.top + ph / 2
        })" text-anchor="middle" font-size="10.5" fill="var(--text-secondary)">${
          curveMode === "roc" ? "Empirical True Accept Rate (TAR)" : "Empirical False Reject Rate (FRR)"
        }</text>
          </svg>
          <div style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px;">
            <span><span class="wb-swatch" style="background:#1d4ed8;display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;"></span>Exact Empirical Step Points (No Smoothing)</span>
            <span class="wb-mono" style="color:#dc2626;font-weight:600;">Active: FAR=${
              op.far !== null ? (op.far * 100).toFixed(2) + "%" : "N/A"
            }, ${curveMode === "roc" ? "TAR" : "FRR"}=${
          (curveMode === "roc" ? op.tar : op.frr) !== null
            ? ((curveMode === "roc" ? op.tar : op.frr) * 100).toFixed(2) + "%"
            : "N/A"
        }</span>
          </div>
        `;

        rocBox.querySelectorAll("[data-roc-idx]").forEach((cEl) => {
          cEl.addEventListener("click", () => {
            const pt = pts[Number(cEl.getAttribute("data-roc-idx"))];
            if (pt) {
              activeTau = pt.threshold;
              renderDynamicSubviews();
            }
          });
        });
      }
    }

    // 5. Pair Error & Outcome Inspector Table
    const classified = classifyPairsAtThreshold(
      curPart.pair_refs || [],
      curPart.pair_scores || [],
      activeTau,
      curPart.higher_is_more_similar !== false
    );
    const tabsMap = {
      fp: { label: `False Accepts · FP (${classified.falseAccepts.length})`, rows: classified.falseAccepts, badge: "wb-badge-acc" },
      fn: { label: `False Rejects · FN (${classified.falseRejects.length})`, rows: classified.falseRejects, badge: "wb-badge-acc" },
      tp: { label: `True Accepts · TP (${classified.trueAccepts.length})`, rows: classified.trueAccepts, badge: "wb-badge-perf" },
      tn: { label: `True Rejects · TN (${classified.trueRejects.length})`, rows: classified.trueRejects, badge: "" },
    };

    const tabsSlot = container.querySelector('[data-role="p2-outcome-tabs"]');
    if (tabsSlot) {
      tabsSlot.innerHTML = Object.entries(tabsMap)
        .map(
          ([k, info]) =>
            `<button type="button" class="wb-btn ${k === activeOutcomeTab ? "" : "wb-btn-secondary"} wb-btn-sm" data-outcome-tab="${k}">${esc(
              info.label
            )}</button>`
        )
        .join("");
      tabsSlot.querySelectorAll("[data-outcome-tab]").forEach((b) => {
        b.addEventListener("click", () => {
          activeOutcomeTab = b.getAttribute("data-outcome-tab");
          renderDynamicSubviews();
        });
      });
    }

    const pairsSlot = container.querySelector('[data-role="p2-pairs-box"]');
    if (pairsSlot) {
      const activeList = tabsMap[activeOutcomeTab]?.rows || [];
      if (activeList.length === 0) {
        pairsSlot.innerHTML = `<div class="wb-help" data-role="p2-pairs-empty">0 pairs in <strong>${esc(
          tabsMap[activeOutcomeTab].label
        )}</strong> at threshold τ = ${activeTau.toFixed(5)}. Adjust τ or select another outcome tab above.</div>`;
      } else {
        pairsSlot.innerHTML = `
          <div class="wb-table-wrap" style="max-height:240px;overflow-y:auto;">
            <table class="wb-table" data-role="p2-pairs-table">
              <thead>
                <tr>
                  <th>Pair ID</th>
                  <th>Ground Truth</th>
                  <th>Image A (Permitted Ref)</th>
                  <th>Image B (Permitted Ref)</th>
                  <th>Score</th>
                  <th>Margin vs. τ</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                ${activeList
                  .slice(0, 50)
                  .map(
                    (pr, rIdx) => `<tr>
                      <td class="wb-mono">${esc(pr.pair_id)}</td>
                      <td><span class="wb-badge ${pr.is_same_subject ? "wb-badge-perf" : ""}">${
                      pr.is_same_subject ? "GENUINE" : "IMPOSTOR"
                    }</span></td>
                      <td class="wb-mono" style="font-size:11px;">${esc(pr.image_a.split("/").pop())}</td>
                      <td class="wb-mono" style="font-size:11px;">${esc(pr.image_b.split("/").pop())}</td>
                      <td class="wb-mono"><strong>${pr.score.toFixed(5)}</strong></td>
                      <td class="wb-mono">${pr.margin >= 0 ? "+" : ""}${pr.margin.toFixed(5)}</td>
                      <td>
                        <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-inspect-pair-idx="${rIdx}">
                          Inspect in 06 · Lab →
                        </button>
                      </td>
                    </tr>`
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `;
        pairsSlot.querySelectorAll("[data-inspect-pair-idx]").forEach((btn) => {
          btn.addEventListener("click", () => {
            const pairObj = activeList[Number(btn.getAttribute("data-inspect-pair-idx"))];
            if (pairObj && onInspectPair) {
              onInspectPair({
                pair: pairObj,
                threshold: activeTau,
                metric: scoreDistributions.metric || "cosine",
                recognizerId: scoreDistributions.recognizer_id,
                detectorId: scoreDistributions.detector_id,
              });
            }
          });
        });
      }
    }
  }

  function mountFull() {
    container.innerHTML = buildFullMountHtml();
    renderDynamicSubviews();

    // Drag interaction on Histogram SVG (zero network calls, lightweight cursor update!)
    const histSvg = container.querySelector('[data-role="p2-hist-svg"]');
    let isDragging = false;

    const pointerToTau = (clientX) => {
      const r = histSvg.getBoundingClientRect();
      if (!r.width) return activeTau;
      const svgX = ((clientX - r.left) / r.width) * W_HIST;
      return xToScore(svgX);
    };

    histSvg?.addEventListener("pointerdown", (e) => {
      isDragging = true;
      histSvg.setPointerCapture?.(e.pointerId);
      activeTau = pointerToTau(e.clientX);
      renderDynamicSubviews();
    });
    histSvg?.addEventListener("pointermove", (e) => {
      if (!isDragging) return;
      activeTau = pointerToTau(e.clientX);
      renderDynamicSubviews();
    });
    const stopDrag = () => {
      isDragging = false;
    };
    histSvg?.addEventListener("pointerup", stopDrag);
    histSvg?.addEventListener("pointercancel", stopDrag);

    container.querySelector("#p2-part-select")?.addEventListener("change", (e) => {
      activePartKey = e.target.value;
      recomputePartitionStructures();
      activeTau = Number(curPart.default_threshold ?? activeTau);
      mountFull();
    });

    container.querySelector("#p2-tau-slider")?.addEventListener("input", (e) => {
      activeTau = Number(e.target.value);
      renderDynamicSubviews();
    });

    container.querySelector("#p2-tau-num")?.addEventListener("change", (e) => {
      if (sanitizeNumericFn) {
        const chk = sanitizeNumericFn(e.target.value, { min: -2.5, max: 1.5, isInt: false, fallback: activeTau });
        if ((chk.invalid || chk.clamped) && onToast) {
          onToast(`Threshold Explorer (τ): ${chk.reason}`, true);
        }
        activeTau = chk.value;
      } else {
        const p = parseFloat(e.target.value);
        activeTau = Number.isFinite(p) ? Math.max(-2.5, Math.min(1.5, p)) : activeTau;
      }
      e.target.value = activeTau.toFixed(5);
      renderDynamicSubviews();
    });

    container.querySelectorAll("[data-preset-tau]").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeTau = Number(btn.getAttribute("data-preset-tau"));
        renderDynamicSubviews();
      });
    });

    container.querySelector("#p2-curve-mode")?.addEventListener("change", (e) => {
      curveMode = e.target.value;
      renderDynamicSubviews();
    });

    container.querySelector('[data-act="export-op-json"]')?.addEventListener("click", () => {
      const op = computeEmpiricalOperatingPoint(
        curPart.genuine_scores,
        curPart.impostor_scores,
        activeTau,
        curPart.higher_is_more_similar !== false
      );
      const exportDoc = {
        exported_at_utc: new Date().toISOString(),
        experiment_id: scoreDistributions.experiment_id || experimentId,
        dataset_id: scoreDistributions.dataset_id,
        partition_id: curPart.partition_id,
        split_name: curPart.split_name,
        evaluation_protocol: scoreDistributions.evaluation_protocol,
        recognizer_id: scoreDistributions.recognizer_id,
        recognizer_sha256: scoreDistributions.recognizer_sha256,
        metric: scoreDistributions.metric,
        empirical_operating_point: op,
      };
      const blob = new Blob([JSON.stringify(exportDoc, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${scoreDistributions.experiment_id || "WB3"}_${curPart.partition_id}_op_${activeTau.toFixed(4)}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      if (onToast) onToast(`Exported empirical operating point at τ=${activeTau.toFixed(4)}.`);
    });
  }

  mountFull();
}
