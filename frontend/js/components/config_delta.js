/**
 * P-1 — Global Cross-Stage Configuration Delta & Two-Run Configuration Comparison Engine
 * (`frontend/js/components/config_delta.js`).
 *
 * Pure mathematical/schema diffing functions are exported standalone (zero DOM requirement)
 * so they can be verified deterministically in Node.js and cross-checked against Python.
 */

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export const FIVE_PIPELINE_STAGES = [
  { key: "detection", label: "Detection", category: "detection" },
  { key: "alignment", label: "Alignment", category: "landmark_alignment" },
  { key: "quality", label: "Quality", category: "quality" },
  { key: "recognition", label: "Recognition", category: "recognition" },
  { key: "retrieval", label: "Retrieval", category: "retrieval" },
];

export function isParamVisibleForValues(paramSpec, values = {}, allParameters = []) {
  if (!paramSpec) return false;
  if (!paramSpec.visible_when) return true;
  const defaultsMap = {};
  for (const p of allParameters || []) {
    defaultsMap[p.name] = p.default;
  }
  for (const [depKey, expectedVal] of Object.entries(paramSpec.visible_when)) {
    const actualVal = values[depKey] !== undefined ? values[depKey] : defaultsMap[depKey];
    if (actualVal !== expectedVal) return false;
  }
  return true;
}

export function areParamValuesEqual(a, b, valueType = "string") {
  if (a === b) return true;
  if ((a === null || a === undefined) && (b === null || b === undefined)) return true;
  if (a === null || a === undefined || b === null || b === undefined) return false;

  if (valueType === "float") {
    const na = Number(a);
    const nb = Number(b);
    return Number.isFinite(na) && Number.isFinite(nb) && Math.abs(na - nb) <= 1e-9;
  }
  if (valueType === "int") {
    const na = Number(a);
    const nb = Number(b);
    return Number.isFinite(na) && Number.isFinite(nb) && Math.round(na) === Math.round(nb);
  }
  if (valueType === "bool") {
    return Boolean(a) === Boolean(b);
  }
  if (typeof a === "object" || typeof b === "object") {
    try {
      return JSON.stringify(a) === JSON.stringify(b);
    } catch (_e) {
      return false;
    }
  }
  return String(a) === String(b);
}

export function formatParamValue(val) {
  if (val === null || val === undefined) return "null";
  if (typeof val === "boolean") return val ? "true" : "false";
  if (typeof val === "object") return JSON.stringify(val);
  return String(val);
}

export function computeSingleStageDelta({
  stageKey,
  stageLabel,
  algoSpec,
  currentValues = {},
  stageEnabled = true,
}) {
  if (!algoSpec) {
    return {
      stageKey,
      stageLabel,
      algorithmId: "unassigned",
      algorithmName: "Unassigned",
      stageEnabled: false,
      modifiedCount: 0,
      totalActiveParams: 0,
      rows: [],
    };
  }

  const params = algoSpec.parameters || [];
  const rows = [];
  let modifiedCount = 0;

  if (!stageEnabled) {
    rows.push({
      stageKey,
      stageLabel,
      algorithmId: algoSpec.algorithm_id,
      algorithmName: algoSpec.name,
      paramName: "(stage disabled)",
      paramLabel: "Stage Execution",
      valueType: "string",
      defaultValue: "enabled",
      currentValue: "disabled",
      status: "Not Applicable",
    });
    return {
      stageKey,
      stageLabel,
      algorithmId: algoSpec.algorithm_id,
      algorithmName: algoSpec.name,
      stageEnabled: false,
      modifiedCount: 0,
      totalActiveParams: 0,
      rows,
    };
  }

  for (const p of params) {
    if (!isParamVisibleForValues(p, currentValues, params)) continue;
    const defVal = p.default;
    const hasCur =
      currentValues &&
      Object.prototype.hasOwnProperty.call(currentValues, p.name) &&
      currentValues[p.name] !== undefined;
    const curVal = hasCur ? currentValues[p.name] : defVal;

    let status = "Default";
    if (p.read_only) {
      status = "Locked";
    } else if (!areParamValuesEqual(curVal, defVal, p.value_type)) {
      status = "Modified";
      modifiedCount += 1;
    }

    rows.push({
      stageKey,
      stageLabel,
      algorithmId: algoSpec.algorithm_id,
      algorithmName: algoSpec.name,
      paramName: p.name,
      paramLabel: p.label,
      valueType: p.value_type,
      defaultValue: defVal,
      currentValue: curVal,
      status,
    });
  }

  return {
    stageKey,
    stageLabel,
    algorithmId: algoSpec.algorithm_id,
    algorithmName: algoSpec.name,
    stageEnabled: true,
    modifiedCount,
    totalActiveParams: rows.length,
    rows,
  };
}

export function computeCrossStageDelta(algorithmsById = {}, expState = {}, { scope = "global" } = {}) {
  let stageInputs = [];

  if (scope === "pipeline" && expState.pipelineA) {
    const p = expState.pipelineA;
    const recParamsMerged = {
      ...(p.recognition?.params || {}),
      ...(p.normalization && p.normalization.enabled !== undefined
        ? { l2_normalize: Boolean(p.normalization.enabled) }
        : {}),
    };
    stageInputs = [
      {
        stageKey: "detection",
        stageLabel: "Detection",
        algoId: p.detection?.algorithm_id || "yunet_2023mar",
        values: p.detection?.params || {},
        enabled: true,
      },
      {
        stageKey: "alignment",
        stageLabel: "Alignment",
        algoId: p.alignment?.algorithm_id || "five_point_canonical_aligner",
        values: p.alignment?.params || {},
        enabled: true,
      },
      {
        stageKey: "quality",
        stageLabel: "Quality",
        algoId: "hybrid_classical_and_norm_quality",
        values: p.quality?.params || {},
        enabled: p.quality?.enabled !== false,
      },
      {
        stageKey: "recognition",
        stageLabel: "Recognition",
        algoId: p.recognition?.algorithm_id || "adaface_ir50_webface4m",
        values: recParamsMerged,
        enabled: p.recognition?.enabled !== false,
      },
      {
        stageKey: "retrieval",
        stageLabel: "Retrieval",
        algoId: p.retrieval?.algorithm_id || "exact_flat_index",
        values: p.retrieval?.params || {},
        enabled: Boolean(p.retrieval?.enabled),
      },
    ];
  } else {
    const detId = expState.detectorId || "yunet_2023mar";
    const recId = expState.recognizerId || "adaface_ir50_webface4m";
    const retId = expState.retrievalIndexId || "ivf_pq_index";
    stageInputs = [
      {
        stageKey: "detection",
        stageLabel: "Detection",
        algoId: detId,
        values: (expState.detectorParamsByModel && expState.detectorParamsByModel[detId]) || {},
        enabled: true,
      },
      {
        stageKey: "alignment",
        stageLabel: "Alignment",
        algoId: "five_point_canonical_aligner",
        values: expState.alignmentParams || {},
        enabled: true,
      },
      {
        stageKey: "quality",
        stageLabel: "Quality",
        algoId: "hybrid_classical_and_norm_quality",
        values: expState.qualityParams || {},
        enabled: true,
      },
      {
        stageKey: "recognition",
        stageLabel: "Recognition",
        algoId: recId,
        values: (expState.recognizerParamsByModel && expState.recognizerParamsByModel[recId]) || {},
        enabled: true,
      },
      {
        stageKey: "retrieval",
        stageLabel: "Retrieval",
        algoId: retId,
        values: (expState.retrievalParamsByIndex && expState.retrievalParamsByIndex[retId]) || {},
        enabled: true,
      },
    ];
  }

  const stages = stageInputs.map((st) =>
    computeSingleStageDelta({
      stageKey: st.stageKey,
      stageLabel: st.stageLabel,
      algoSpec: algorithmsById[st.algoId],
      currentValues: st.values,
      stageEnabled: st.enabled,
    })
  );

  const stageCounts = {};
  let totalModified = 0;
  const allRows = [];
  const modifiedRows = [];

  for (const s of stages) {
    stageCounts[s.stageKey] = s.modifiedCount;
    totalModified += s.modifiedCount;
    for (const r of s.rows) {
      allRows.push(r);
      if (r.status === "Modified") modifiedRows.push(r);
    }
  }

  return { scope, totalModified, stageCounts, stages, modifiedRows, allRows };
}

export function extractNormalizedRunStages(rec, algorithmsById = {}) {
  const out = {};
  if (!rec) return out;
  const p = rec.parameters || {};
  const algos = rec.algorithms || {};

  const fillWithRegistryDefaults = (algoId, rawParams = {}) => {
    const spec = algorithmsById[algoId];
    const merged = {};
    if (spec && Array.isArray(spec.parameters)) {
      for (const param of spec.parameters) {
        if (!isParamVisibleForValues(param, rawParams, spec.parameters)) continue;
        merged[param.name] =
          rawParams && rawParams[param.name] !== undefined ? rawParams[param.name] : param.default;
      }
    }
    for (const [k, v] of Object.entries(rawParams || {})) {
      if (typeof v !== "object" || v === null || Array.isArray(v)) {
        const specParam = spec?.parameters?.find((sp) => sp.name === k);
        if (specParam && !isParamVisibleForValues(specParam, rawParams, spec.parameters)) continue;
        merged[k] = v;
      }
    }
    return merged;
  };

  if (p.detection || p.recognition || rec.category === "pipeline" || rec.category === "pipeline_builder") {
    if (p.detection || algos.detector) {
      const detId = p.detection?.algorithm_id || algos.detector || "yunet_2023mar";
      out.detection = {
        stageLabel: "Detection",
        algorithmId: detId,
        params: fillWithRegistryDefaults(detId, p.detection?.params || {}),
      };
    }
    if (p.alignment) {
      const alId = p.alignment?.algorithm_id || "five_point_canonical_aligner";
      out.alignment = {
        stageLabel: "Alignment",
        algorithmId: alId,
        params: fillWithRegistryDefaults(alId, p.alignment?.params || {}),
      };
    }
    if (p.quality) {
      const qEnabled = p.quality.enabled !== false;
      out.quality = {
        stageLabel: "Quality",
        algorithmId: qEnabled ? "hybrid_classical_and_norm_quality" : "disabled",
        params: qEnabled
          ? fillWithRegistryDefaults("hybrid_classical_and_norm_quality", p.quality.params || {})
          : { enabled: false },
      };
    }
    if (p.recognition || algos.recognizer) {
      const rEnabled = p.recognition ? p.recognition.enabled !== false : true;
      const recId = p.recognition?.algorithm_id || algos.recognizer || "adaface_ir50_webface4m";
      const recRaw = {
        ...(p.recognition?.params || {}),
        ...(p.normalization && p.normalization.enabled !== undefined
          ? { l2_normalize: Boolean(p.normalization.enabled) }
          : {}),
      };
      out.recognition = {
        stageLabel: "Recognition",
        algorithmId: rEnabled ? recId : "disabled",
        params: rEnabled ? fillWithRegistryDefaults(recId, recRaw) : { enabled: false },
      };
    }
    if (p.matching) {
      out.matching = {
        stageLabel: "Matching",
        algorithmId: "pairwise_vector_matcher",
        params: { metric: p.matching.metric ?? "cosine", threshold: p.matching.threshold ?? 0.2169 },
      };
    }
    if (p.retrieval && p.retrieval.enabled) {
      const retId = p.retrieval.algorithm_id || "exact_flat_index";
      out.retrieval = {
        stageLabel: "Retrieval",
        algorithmId: retId,
        params: fillWithRegistryDefaults(retId, p.retrieval.params || {}),
      };
    }
    if (Object.keys(out).length > 0) return out;
  }

  if (rec.category === "detection") {
    const detId =
      algos.detector || p.detector_id || (Array.isArray(algos.detectors) ? algos.detectors.join("+") : "detector");
    const rawDetParams = p.base_params ? { ...p.base_params, sweep_param: p.sweep_param } : p;
    out.detection = {
      stageLabel: "Detection",
      algorithmId: detId,
      params: fillWithRegistryDefaults(detId, rawDetParams),
    };
  }

  if (rec.category === "recognition") {
    if (p.detector_id || algos.landmark_source) {
      out.detection = {
        stageLabel: "Detection / Landmarks",
        algorithmId: p.detector_id || algos.landmark_source,
        params: {},
      };
    }
    const recId = p.recognizer_id || algos.recognizer || "recognizer";
    out.recognition = {
      stageLabel: "Recognition",
      algorithmId: recId,
      params: fillWithRegistryDefaults(recId, p.recognizer_params || p),
    };
    if (p.metric !== undefined || p.threshold !== undefined) {
      out.matching = {
        stageLabel: "Matching",
        algorithmId: "pairwise_vector_matcher",
        params: {
          ...(p.metric !== undefined ? { metric: p.metric } : {}),
          ...(p.threshold !== undefined ? { threshold: p.threshold } : {}),
        },
      };
    }
  }

  if (rec.category === "retrieval") {
    const idxId =
      p.index_id || algos.retrieval_index || (Array.isArray(algos.indexes) ? algos.indexes.join("+") : "retrieval_index");
    out.retrieval = {
      stageLabel: "Retrieval",
      algorithmId: idxId,
      params: {
        ...(p.gallery_size !== undefined ? { gallery_size: p.gallery_size } : {}),
        ...(p.num_queries !== undefined ? { num_queries: p.num_queries } : {}),
        ...fillWithRegistryDefaults(idxId, p.index_params || p),
      },
    };
  }

  return out;
}

export function compareTwoExperimentConfigs(recA, recB, algorithmsById = {}) {
  if (!recA || !recB) {
    return { valid: false, warnings: [], counts: { changed: 0, identical: 0, added: 0, absent: 0, total: 0 }, rows: [] };
  }

  const warnings = [];
  const datasetMismatch = recA.dataset_id !== recB.dataset_id;
  const splitMismatch = recA.dataset_split !== recB.dataset_split;
  const categoryMismatch = recA.category !== recB.category;

  if (datasetMismatch) {
    warnings.push(
      `Dataset Mismatch: Run A uses '${recA.dataset_id}' whereas Run B uses '${recB.dataset_id}'. Metrics across distinct datasets are not directly comparable.`
    );
  } else if (splitMismatch) {
    warnings.push(
      `Split / Protocol Mismatch: Run A uses split '${recA.dataset_split}' whereas Run B uses '${recB.dataset_split}'.`
    );
  }
  if (categoryMismatch) {
    warnings.push(`Category Mismatch: Run A is '${recA.category}' whereas Run B is '${recB.category}'.`);
  }

  const stagesA = extractNormalizedRunStages(recA, algorithmsById);
  const stagesB = extractNormalizedRunStages(recB, algorithmsById);
  const orderedStageKeys = ["detection", "alignment", "quality", "recognition", "matching", "retrieval"];
  const allStageKeys = Array.from(new Set([...orderedStageKeys, ...Object.keys(stagesA), ...Object.keys(stagesB)]));

  const rows = [];
  let changed = 0;
  let identical = 0;
  let added = 0;
  let absent = 0;
  let modelFamilyMismatch = false;

  for (const sk of allStageKeys) {
    const sa = stagesA[sk];
    const sb = stagesB[sk];
    if (!sa && !sb) continue;

    const stageLabel = sa?.stageLabel || sb?.stageLabel || sk;
    const modelA = sa?.algorithmId || "— (Stage Absent)";
    const modelB = sb?.algorithmId || "— (Stage Absent)";

    if (sa && sb && sa.algorithmId !== sb.algorithmId) {
      modelFamilyMismatch = true;
      rows.push({
        stageKey: sk,
        stageLabel,
        modelA,
        modelB,
        paramName: "algorithm_id",
        valA: modelA,
        valB: modelB,
        status: "Changed",
      });
      changed += 1;
    }

    const keysUnion = Array.from(new Set([...Object.keys(sa?.params || {}), ...Object.keys(sb?.params || {})]));
    for (const pk of keysUnion) {
      const inA = Boolean(sa && Object.prototype.hasOwnProperty.call(sa.params, pk));
      const inB = Boolean(sb && Object.prototype.hasOwnProperty.call(sb.params, pk));
      const va = inA ? sa.params[pk] : undefined;
      const vb = inB ? sb.params[pk] : undefined;

      if (inA && !inB) {
        rows.push({ stageKey: sk, stageLabel, modelA, modelB, paramName: pk, valA: va, valB: undefined, status: "Absent in Run B" });
        absent += 1;
      } else if (!inA && inB) {
        rows.push({ stageKey: sk, stageLabel, modelA, modelB, paramName: pk, valA: undefined, valB: vb, status: "Added in Run B" });
        added += 1;
      } else {
        const sameContextAndVal =
          sa.algorithmId === sb.algorithmId && areParamValuesEqual(va, vb, typeof va === "number" ? "float" : "string");
        if (sameContextAndVal) {
          rows.push({ stageKey: sk, stageLabel, modelA, modelB, paramName: pk, valA: va, valB: vb, status: "Identical" });
          identical += 1;
        } else {
          rows.push({ stageKey: sk, stageLabel, modelA, modelB, paramName: pk, valA: va, valB: vb, status: "Changed" });
          changed += 1;
        }
      }
    }
  }

  if (modelFamilyMismatch) {
    warnings.push(
      "Model / Algorithm Family Difference: At least one pipeline stage uses a different algorithm/backbone between Run A and Run B (inspect stage context alongside parameter values)."
    );
  }

  return {
    valid: true,
    runAId: recA.experiment_id,
    runBId: recB.experiment_id,
    datasetMismatch,
    splitMismatch,
    categoryMismatch,
    modelFamilyMismatch,
    warnings,
    counts: { changed, identical, added, absent, total: rows.length },
    rows,
  };
}

export function renderConfigDeltaPanel(
  container,
  {
    algorithmsById,
    expState,
    scope = "global",
    title = "Active Configuration Delta vs. Frozen Registry Defaults",
    onResetConfirmed = null,
    onClose = null,
  }
) {
  if (!container) return;
  let showAll = false;
  let confirmingReset = false;

  function draw() {
    const delta = computeCrossStageDelta(algorithmsById, expState, { scope });
    const rowsToShow = showAll ? delta.allRows : delta.modifiedRows;

    const stagePillsHtml = delta.stages
      .map((st) => {
        const activeCls = st.modifiedCount > 0 ? "wb-badge-acc" : "";
        const suffix = !st.stageEnabled ? "Off" : `${st.modifiedCount} mod`;
        return `<span class="wb-badge ${activeCls}" title="${esc(st.algorithmName)}">${esc(st.stageLabel)} (${esc(
          st.algorithmId
        )}): ${suffix}</span>`;
      })
      .join(" ");

    container.innerHTML = `
      <div class="wb-panel" style="margin-bottom:12px;border-color:var(--border-strong);">
        <div class="wb-panel-header" style="flex-wrap:wrap;gap:8px;">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <strong>${esc(title)}</strong>
            <span class="wb-badge ${delta.totalModified > 0 ? "wb-badge-acc" : "wb-badge-perf"}" data-role="delta-total-badge">
              ${delta.totalModified} Active Modification${delta.totalModified === 1 ? "" : "s"}
            </span>
          </div>
          <div class="wb-toolbar-actions" style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <label style="display:inline-flex;align-items:center;gap:5px;font-size:11.5px;cursor:pointer;">
              <input type="checkbox" data-act="toggle-show-all" ${showAll ? "checked" : ""} />
              Show Default &amp; Locked Rows (${delta.allRows.length})
            </label>
            ${
              onResetConfirmed
                ? `<button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="request-reset" ${
                    delta.totalModified === 0
                      ? 'disabled title="All stages are already at frozen registry defaults."'
                      : 'title="Reset all 5 active stages to frozen registry defaults (requires confirmation; never alters saved history)."'
                  }>Reset All Stages to Defaults</button>`
                : ""
            }
            ${
              onClose
                ? `<button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="close-drawer" aria-label="Close Active Configuration Delta drawer">✕ Close</button>`
                : ""
            }
          </div>
        </div>
        <div class="wb-panel-body">
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;" data-role="stage-summary-pills">${stagePillsHtml}</div>

          ${
            confirmingReset
              ? `
            <div class="wb-banner wb-banner-warn" style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;" data-role="reset-confirm-box">
              <span><strong>Confirm Multi-Stage Reset:</strong> Restore all active stage parameters to frozen registry defaults? Immutable historical records and saved experiments on disk will NOT be modified.</span>
              <div style="display:flex;gap:6px;">
                <button type="button" class="wb-btn wb-btn-sm" data-act="confirm-reset-yes">Confirm Reset</button>
                <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="confirm-reset-cancel">Cancel</button>
              </div>
            </div>`
              : ""
          }

          ${
            delta.totalModified === 0 && !showAll
              ? `<div class="wb-banner wb-banner-match" style="margin-bottom:0;" data-role="delta-empty-state">
                   ✓ <strong>Zero Active Modifications:</strong> All 5 pipeline stages (Detection, Alignment, Quality, Recognition, Retrieval) match their frozen registry defaults. Toggle "Show Default &amp; Locked Rows" to inspect every active parameter contract.
                 </div>`
              : `
            <div class="wb-table-wrap" style="max-height:300px;overflow-y:auto;">
              <table class="wb-table" data-role="delta-table">
                <thead>
                  <tr>
                    <th>Stage</th>
                    <th>Algorithm / Model</th>
                    <th>Parameter</th>
                    <th>Frozen Baseline</th>
                    <th>Current Value</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  ${rowsToShow
                    .map((r) => {
                      const badgeCls =
                        r.status === "Modified"
                          ? "wb-badge-acc"
                          : r.status === "Locked"
                          ? "wb-badge-readonly"
                          : r.status === "Default"
                          ? "wb-badge-perf"
                          : "";
                      return `<tr data-delta-stage="${esc(r.stageKey)}" data-delta-param="${esc(r.paramName)}">
                        <td><strong>${esc(r.stageLabel)}</strong></td>
                        <td class="wb-mono" style="font-size:11px;">${esc(r.algorithmId)}</td>
                        <td class="wb-mono"><strong>${esc(r.paramName)}</strong><div class="wb-muted-xs">${esc(r.paramLabel)}</div></td>
                        <td class="wb-mono">${esc(formatParamValue(r.defaultValue))}</td>
                        <td class="wb-mono"><strong>${esc(formatParamValue(r.currentValue))}</strong></td>
                        <td><span class="wb-badge ${badgeCls}">${esc(r.status)}</span></td>
                      </tr>`;
                    })
                    .join("")}
                </tbody>
              </table>
            </div>`
          }
        </div>
      </div>
    `;

    container.querySelector('[data-act="toggle-show-all"]')?.addEventListener("change", (e) => {
      showAll = e.target.checked;
      draw();
    });
    container.querySelector('[data-act="request-reset"]')?.addEventListener("click", () => {
      confirmingReset = true;
      draw();
    });
    container.querySelector('[data-act="confirm-reset-cancel"]')?.addEventListener("click", () => {
      confirmingReset = false;
      draw();
    });
    container.querySelector('[data-act="confirm-reset-yes"]')?.addEventListener("click", () => {
      confirmingReset = false;
      if (onResetConfirmed) onResetConfirmed();
      draw();
    });
    container.querySelector('[data-act="close-drawer"]')?.addEventListener("click", () => {
      if (onClose) onClose();
    });
  }

  draw();
  return { refresh: draw };
}

export function renderTwoRunConfigComparison(container, { experiments = [], algorithmsById = {}, selectedIds = [] }) {
  if (!container) return;
  if (experiments.length < 2) {
    container.innerHTML = `<div class="wb-help">At least two experiment records are required to compare configurations.</div>`;
    return;
  }

  let runAId = selectedIds[0] || experiments[0]?.experiment_id;
  let runBId =
    selectedIds[1] || experiments.find((e) => e.experiment_id !== runAId)?.experiment_id || experiments[0]?.experiment_id;
  let diffOnly = false;

  function draw() {
    const recA = experiments.find((e) => e.experiment_id === runAId) || experiments[0];
    const recB = experiments.find((e) => e.experiment_id === runBId) || experiments[1] || experiments[0];
    const cmp = compareTwoExperimentConfigs(recA, recB, algorithmsById);
    const displayRows = diffOnly ? cmp.rows.filter((r) => r.status !== "Identical") : cmp.rows;

    container.innerHTML = `
      <div class="wb-grid-3" style="margin-bottom:10px;align-items:end;">
        <div>
          <label class="wb-label" for="cmp-cfg-run-a">Run A (Reference Configuration)</label>
          <select id="cmp-cfg-run-a" class="wb-select">
            ${experiments
              .map(
                (e) =>
                  `<option value="${esc(e.experiment_id)}" ${e.experiment_id === recA.experiment_id ? "selected" : ""}>${esc(
                    e.experiment_id
                  )} — ${esc(e.name.slice(0, 48))}</option>`
              )
              .join("")}
          </select>
        </div>
        <div>
          <label class="wb-label" for="cmp-cfg-run-b">Run B (Comparison Configuration)</label>
          <select id="cmp-cfg-run-b" class="wb-select">
            ${experiments
              .map(
                (e) =>
                  `<option value="${esc(e.experiment_id)}" ${e.experiment_id === recB.experiment_id ? "selected" : ""}>${esc(
                    e.experiment_id
                  )} — ${esc(e.name.slice(0, 48))}</option>`
              )
              .join("")}
          </select>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;">
          <label style="display:inline-flex;align-items:center;gap:5px;font-size:12px;cursor:pointer;">
            <input type="checkbox" id="cmp-cfg-diff-only" ${diffOnly ? "checked" : ""} />
            Show Differences Only
          </label>
          <div style="display:flex;gap:4px;flex-wrap:wrap;" data-role="two-run-summary">
            <span class="wb-badge wb-badge-acc">Changed: ${cmp.counts.changed}</span>
            <span class="wb-badge wb-badge-perf">Identical: ${cmp.counts.identical}</span>
            <span class="wb-badge">Added: ${cmp.counts.added}</span>
            <span class="wb-badge">Absent: ${cmp.counts.absent}</span>
          </div>
        </div>
      </div>

      ${
        cmp.warnings.length > 0
          ? `<div class="wb-banner wb-banner-warn" style="margin-bottom:8px;" data-role="two-run-warning">
               ⚠ <strong>Configuration &amp; Protocol Compatibility Warning:</strong>
               <ul style="margin:4px 0 0 18px;padding:0;">
                 ${cmp.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}
               </ul>
             </div>`
          : ""
      }

      <div class="wb-muted-xs" style="margin-bottom:8px;">
        <strong>Scientific Interpretation Note:</strong> Configuration diffs document parameter/stage setup differences between Run A (${esc(
          recA.source_origin
        )}) and Run B (${esc(recB.source_origin)}) and do <em>not</em> by themselves prove causal metric attribution when datasets, splits, or model families differ.
      </div>

      ${
        displayRows.length === 0
          ? `<div class="wb-banner wb-banner-match">✓ No differing parameters under current filter (${cmp.counts.identical} identical parameters).</div>`
          : `<div class="wb-table-wrap" style="max-height:320px;overflow-y:auto;">
               <table class="wb-table" id="cmp-two-run-table">
                 <thead>
                   <tr>
                     <th>Stage</th>
                     <th>Run A Model</th>
                     <th>Run B Model</th>
                     <th>Parameter</th>
                     <th>Run A Value</th>
                     <th>Run B Value</th>
                     <th>Status</th>
                   </tr>
                 </thead>
                 <tbody>
                   ${displayRows
                     .map((r) => {
                       const bCls =
                         r.status === "Changed" ? "wb-badge-acc" : r.status === "Identical" ? "wb-badge-perf" : "";
                       return `<tr>
                         <td><strong>${esc(r.stageLabel)}</strong></td>
                         <td class="wb-mono" style="font-size:11px;">${esc(r.modelA)}</td>
                         <td class="wb-mono" style="font-size:11px;">${esc(r.modelB)}</td>
                         <td class="wb-mono"><strong>${esc(r.paramName)}</strong></td>
                         <td class="wb-mono">${
                           r.valA === undefined ? '<span class="wb-muted-xs">— (Absent)</span>' : esc(formatParamValue(r.valA))
                         }</td>
                         <td class="wb-mono">${
                           r.valB === undefined ? '<span class="wb-muted-xs">— (Absent)</span>' : esc(formatParamValue(r.valB))
                         }</td>
                         <td><span class="wb-badge ${bCls}">${esc(r.status)}</span></td>
                       </tr>`;
                     })
                     .join("")}
                 </tbody>
               </table>
             </div>`
      }
    `;

    container.querySelector("#cmp-cfg-run-a")?.addEventListener("change", (e) => {
      runAId = e.target.value;
      draw();
    });
    container.querySelector("#cmp-cfg-run-b")?.addEventListener("change", (e) => {
      runBId = e.target.value;
      draw();
    });
    container.querySelector("#cmp-cfg-diff-only")?.addEventListener("change", (e) => {
      diffOnly = e.target.checked;
      draw();
    });
  }

  draw();
}
