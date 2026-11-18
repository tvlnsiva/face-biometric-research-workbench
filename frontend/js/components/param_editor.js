/**
 * Dynamic Registry-Driven Parameter Editor (`frontend/js/components/param_editor.js`).
 *
 * Renders controls exclusively from `AlgorithmSpec.parameters` metadata — zero hard-coded
 * per-algorithm forms. Supports conditional `visible_when`, read-only locked contracts,
 * experimental override warnings, reset-to-default, configuration diffing, accessible
 * `<label for>` bindings, viewport-clamped contextual tooltips (`?`), and strict numeric clamping.
 */

import { showToast, store } from "../state.js";
import { TECHNICAL_GLOSSARY, sanitizeNumericInput } from "../utils/tooltips.js";
import { areParamValuesEqual, isParamVisibleForValues } from "./config_delta.js";

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function isParamVisible(param, values, allParameters = []) {
  return isParamVisibleForValues(param, values, allParameters);
}

function impactBadgeHtml(impact, readOnly) {
  if (readOnly) {
    return `<span class="wb-badge wb-badge-readonly" title="Locked read-only model specification — required by model contract">LOCKED</span>`;
  }
  if (impact === "accuracy") {
    return `<span class="wb-badge wb-badge-acc" title="Affects accuracy / recognition operating point">ACCURACY</span>`;
  }
  if (impact === "performance") {
    return `<span class="wb-badge wb-badge-perf" title="Affects latency, throughput, or memory">PERF</span>`;
  }
  return `<span class="wb-badge wb-badge-both" title="Affects both accuracy and runtime performance">ACC+PERF</span>`;
}

function buildTipAttrs(p, rangeHint) {
  if (TECHNICAL_GLOSSARY[p.name]) {
    return `data-tip-key="${escapeHtml(p.name)}"`;
  }
  const lim = p.read_only
    ? `Locked contract (${rangeHint}) — enable Experimental Override where supported to unlock.`
    : rangeHint;
  return `data-tip-title="${escapeHtml(`${p.label} (${p.name})`)}" data-tip-body="${escapeHtml(
    p.description
  )}" data-tip-note="${escapeHtml(lim)}"`;
}

export function computeConfigDiff(parameters, values) {
  const diffs = [];
  const list = parameters || [];
  for (const p of list) {
    if (p.read_only) continue;
    if (!isParamVisible(p, values || {}, list)) continue;
    const cur = values ? values[p.name] : undefined;
    const def = p.default;
    if (cur !== undefined && !areParamValuesEqual(cur, def, p.value_type)) {
      diffs.push({ name: p.name, label: p.label, default: def, current: cur });
    }
  }
  return diffs;
}

let instanceSeq = 0;

export function renderParameterEditor(container, algoSpec, values, onChange, { showHeader = true } = {}) {
  if (!container) return;
  if (!algoSpec) {
    container.innerHTML = `<div class="wb-help">No algorithm selected.</div>`;
    return;
  }

  if (!container.dataset.pePrefix) {
    instanceSeq += 1;
    container.dataset.pePrefix = `pe-${instanceSeq}`;
  }
  const domPrefix = `${container.dataset.pePrefix}-${algoSpec.algorithm_id}`;

  const params = algoSpec.parameters || [];
  const paramsByName = Object.fromEntries(params.map((p) => [p.name, p]));
  const diffs = computeConfigDiff(params, values);

  const rerender = () => renderParameterEditor(container, algoSpec, values, onChange, { showHeader });

  let html = "";
  if (showHeader) {
    html += `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:6px;">
        <div>
          <div style="font-weight:700;font-size:12.5px;">${escapeHtml(algoSpec.name)}</div>
          <div class="wb-muted-xs wb-mono">${escapeHtml(algoSpec.algorithm_id)} · ${diffs.length} modified</div>
        </div>
        <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-action="reset-defaults"
          title="Restore all editable parameters of ${escapeHtml(algoSpec.name)} to canonical defaults"
          data-tip-title="Reset Parameter Defaults"
          data-tip-body="Restores every editable parameter of ${escapeHtml(algoSpec.name)} back to its validated baseline default.">
          Reset Defaults
        </button>
      </div>
    `;
  }

  if (params.length === 0) {
    html += `<div class="wb-help">No user-editable parameters for this entry (${escapeHtml(algoSpec.availability_status)}).</div>`;
  }

  for (const p of params) {
    if (!isParamVisible(p, values, params)) continue;
    const val = values[p.name] !== undefined ? values[p.name] : p.default;
    const rangeHint =
      p.min_value !== null && p.min_value !== undefined && p.max_value !== null && p.max_value !== undefined
        ? `Range: [${p.min_value} .. ${p.max_value}] · Default: ${p.default}`
        : `Default: ${p.default}`;
    const tipAttrs = buildTipAttrs(p, rangeHint);
    const ctrlId = `${domPrefix}-${p.name}`;
    const disabledReason = p.read_only ? " title=\"Locked read-only model specification\"" : "";

    html += `<div class="wb-field" ${tipAttrs}>`;
    html += `
      <div class="wb-field-top">
        <span style="display:inline-flex;align-items:center;">
          <label class="wb-label" for="${escapeHtml(ctrlId)}">${escapeHtml(p.label)}</label>
          <button type="button" class="wb-help-trigger" ${tipAttrs} aria-label="Explain ${escapeHtml(p.label)}">?</button>
        </span>
        ${impactBadgeHtml(p.impact, p.read_only)}
      </div>
    `;

    if (p.control_type === "slider") {
      html += `
        <div class="wb-slider-row">
          <input type="range" id="${escapeHtml(ctrlId)}" data-param="${escapeHtml(p.name)}" data-vtype="${escapeHtml(p.value_type)}"
            min="${p.min_value}" max="${p.max_value}" step="${p.step ?? 1}" value="${escapeHtml(val)}"
            ${tipAttrs} ${p.read_only ? "disabled" : ""}${disabledReason} />
          <input type="number" class="wb-input wb-mono" aria-label="${escapeHtml(p.label)} numeric value"
            data-param-num="${escapeHtml(p.name)}" data-vtype="${escapeHtml(p.value_type)}"
            min="${p.min_value}" max="${p.max_value}" step="${p.step ?? 1}" value="${escapeHtml(val)}"
            ${tipAttrs} ${p.read_only ? "readonly" : ""}${disabledReason} />
        </div>
      `;
    } else if (p.control_type === "number") {
      html += `
        <input type="number" id="${escapeHtml(ctrlId)}" class="wb-input wb-mono" data-param-num="${escapeHtml(p.name)}" data-vtype="${escapeHtml(p.value_type)}"
          ${p.min_value !== null && p.min_value !== undefined ? `min="${p.min_value}"` : ""}
          ${p.max_value !== null && p.max_value !== undefined ? `max="${p.max_value}"` : ""}
          step="${p.step ?? 1}" value="${escapeHtml(val)}" ${tipAttrs} ${p.read_only ? "readonly" : ""}${disabledReason} />
      `;
    } else if (p.control_type === "select") {
      const opts = (p.options || [])
        .map(
          (o) =>
            `<option value="${escapeHtml(o.value)}" ${String(o.value) === String(val) ? "selected" : ""}>${escapeHtml(
              o.label
            )}</option>`
        )
        .join("");
      html += `
        <select id="${escapeHtml(ctrlId)}" class="wb-select" data-param="${escapeHtml(p.name)}" data-vtype="${escapeHtml(
        p.value_type
      )}" ${tipAttrs} ${p.read_only ? "disabled" : ""}${disabledReason}>${opts}</select>
      `;
    } else if (p.control_type === "boolean") {
      html += `
        <label for="${escapeHtml(ctrlId)}" style="display:inline-flex;align-items:center;gap:7px;cursor:pointer;font-size:12px;">
          <input type="checkbox" id="${escapeHtml(ctrlId)}" data-param="${escapeHtml(p.name)}" data-vtype="bool" ${
        val ? "checked" : ""
      } ${tipAttrs} ${p.read_only ? "disabled" : ""}${disabledReason} />
          <span>${val ? "Enabled (True)" : "Disabled (False)"}</span>
        </label>
      `;
      if (val && p.experimental_warning) {
        html += `<div class="wb-banner wb-banner-warn" style="margin-top:6px;margin-bottom:2px;">⚠ ${escapeHtml(
          p.experimental_warning
        )}</div>`;
      }
    }

    html += `<div class="wb-help">${escapeHtml(p.description)} <span class="wb-mono">(${escapeHtml(rangeHint)})</span></div>`;
    html += `</div>`;
  }

  if (diffs.length > 0) {
    html += `
      <details style="margin-top:8px;font-size:11.5px;">
        <summary style="cursor:pointer;font-weight:600;color:var(--accent);" title="Show parameters modified away from canonical defaults">Configuration Diff vs. Default (${diffs.length})</summary>
        <div class="wb-mono" style="margin-top:5px;padding:6px;background:var(--bg-metric);border-radius:4px;">
          ${diffs
            .map(
              (d) =>
                `<div>${escapeHtml(d.name)}: <span style="opacity:0.7">${escapeHtml(d.default)}</span> → <strong>${escapeHtml(
                  d.current
                )}</strong></div>`
            )
            .join("")}
        </div>
      </details>
    `;
  }

  container.innerHTML = html;

  const parseChoiceOrBool = (raw, vtype, checked) => {
    if (vtype === "bool") return Boolean(checked);
    if (vtype === "int") return parseInt(raw, 10);
    if (vtype === "float") return parseFloat(raw);
    return raw;
  };

  container.querySelectorAll("[data-param]").forEach((el) => {
    const pName = el.getAttribute("data-param");
    const vType = el.getAttribute("data-vtype");
    const evtName = el.type === "range" ? "input" : "change";
    el.addEventListener(evtName, () => {
      const parsed = parseChoiceOrBool(el.value, vType, el.checked);
      values[pName] = parsed;
      const partner = container.querySelector(`[data-param-num="${pName}"]`);
      if (partner) partner.value = parsed;
      if (el.type !== "range") {
        rerender();
      }
      store.notify("config_change");
      if (onChange) onChange(pName, parsed, values);
    });
    if (el.type === "range") {
      el.addEventListener("change", () => {
        rerender();
        store.notify("config_change");
      });
    }
  });

  container.querySelectorAll("[data-param-num]").forEach((el) => {
    const pName = el.getAttribute("data-param-num");
    const vType = el.getAttribute("data-vtype");
    const spec = paramsByName[pName];
    el.addEventListener("change", () => {
      const minV = spec?.min_value !== null && spec?.min_value !== undefined ? Number(spec.min_value) : -Infinity;
      const maxV = spec?.max_value !== null && spec?.max_value !== undefined ? Number(spec.max_value) : Infinity;
      const fallback = values[pName] !== undefined ? values[pName] : spec?.default ?? 0;
      const check = sanitizeNumericInput(el.value, {
        min: minV,
        max: maxV,
        isInt: vType === "int",
        fallback,
      });
      if (check.invalid || check.clamped) {
        showToast(`${spec?.label || pName}: ${check.reason}`, true);
      }
      values[pName] = check.value;
      el.value = String(check.value);
      rerender();
      store.notify("config_change");
      if (onChange) onChange(pName, check.value, values);
    });
  });

  const resetBtn = container.querySelector('[data-action="reset-defaults"]');
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      for (const p of params) {
        if (!p.read_only) values[p.name] = p.default;
      }
      rerender();
      store.notify("config_change");
      if (onChange) onChange("__reset__", null, values);
    });
  }
}
