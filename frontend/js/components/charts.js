/**
 * Reusable Scientific SVG Chart Suite (`frontend/js/components/charts.js`).
 *
 * Supports multi-series line sweeps, ROC / DET / PR curves, scatter plots, bar charts, and
 * non-probabilistic similarity-vs-threshold gauges with explicit dataset/protocol provenance.
 */

const SERIES_PALETTE = ["#1d4ed8", "#0d9488", "#d97706", "#7c3aed", "#dc2626", "#0284c7"];

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function renderLineChart(container, { title, subtitle = "", xLabel, yLabel, series = [], height = 250, logX = false }) {
  if (!container) return;
  const validSeries = series.filter((s) => Array.isArray(s.points) && s.points.length > 0);
  if (validSeries.length === 0) {
    container.innerHTML = `<div class="wb-help" style="padding:16px;">No data points available for "${esc(title)}".</div>`;
    return;
  }

  const W = 540;
  const H = height;
  const pad = { top: 28, right: 22, bottom: 38, left: 54 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const tx = (v) => (logX ? Math.log10(Math.max(v, 1e-6)) : v);
  const allPts = validSeries.flatMap((s) => s.points);
  let xMin = Math.min(...allPts.map((p) => tx(p.x)));
  let xMax = Math.max(...allPts.map((p) => tx(p.x)));
  let yMin = Math.min(...allPts.map((p) => p.y));
  let yMax = Math.max(...allPts.map((p) => p.y));

  if (Math.abs(xMax - xMin) < 1e-9) {
    xMin -= 1;
    xMax += 1;
  }
  if (Math.abs(yMax - yMin) < 1e-9) {
    yMin -= 0.05;
    yMax += 0.05;
  } else {
    const padY = (yMax - yMin) * 0.08;
    yMin -= padY;
    yMax += padY;
  }

  const sx = (v) => pad.left + ((tx(v) - xMin) / (xMax - xMin)) * plotW;
  const sy = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  let gridSvg = "";
  for (let i = 0; i <= 4; i++) {
    const gy = pad.top + (i / 4) * plotH;
    const val = yMax - (i / 4) * (yMax - yMin);
    gridSvg += `<line x1="${pad.left}" y1="${gy}" x2="${W - pad.right}" y2="${gy}" stroke="var(--border)" stroke-dasharray="3,3" />`;
    gridSvg += `<text x="${pad.left - 6}" y="${gy + 3.5}" text-anchor="end" font-size="10" fill="var(--text-muted)">${val.toFixed(
      Math.abs(yMax - yMin) < 0.05 ? 4 : 2
    )}</text>`;
  }

  for (let i = 0; i <= 4; i++) {
    const gx = pad.left + (i / 4) * plotW;
    const rawT = xMin + (i / 4) * (xMax - xMin);
    const val = logX ? Math.pow(10, rawT) : rawT;
    gridSvg += `<text x="${gx}" y="${H - 18}" text-anchor="middle" font-size="10" fill="var(--text-muted)">${
      logX ? val.toExponential(0) : Number(val.toPrecision(4))
    }</text>`;
  }

  let pathsSvg = "";
  validSeries.forEach((s, idx) => {
    const col = s.color || SERIES_PALETTE[idx % SERIES_PALETTE.length];
    const sorted = [...s.points].sort((a, b) => a.x - b.x);
    const d = sorted.map((p, k) => `${k === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    pathsSvg += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2" />`;
    sorted.forEach((p) => {
      pathsSvg += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="3.4" fill="${col}">
        <title>${esc(s.name)}: (${p.x}, ${p.y})${p.label ? ` — ${esc(p.label)}` : ""}</title>
      </circle>`;
    });
  });

  const legendHtml = validSeries
    .map((s, idx) => {
      const col = s.color || SERIES_PALETTE[idx % SERIES_PALETTE.length];
      return `<span class="wb-legend-item"><span class="wb-swatch" style="background:${col}"></span>${esc(s.name)}</span>`;
    })
    .join("");

  container.innerHTML = `
    <div>
      <div style="font-weight:600;font-size:12px;">${esc(title)}</div>
      ${subtitle ? `<div class="wb-muted-xs" style="margin-bottom:4px;">${esc(subtitle)}</div>` : ""}
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;">
        ${gridSvg}
        <rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" fill="none" stroke="var(--border-strong)" />
        ${pathsSvg}
        <text x="${pad.left + plotW / 2}" y="${H - 3}" text-anchor="middle" font-size="10.5" fill="var(--text-secondary)">${esc(
    xLabel
  )}</text>
        <text x="12" y="${pad.top + plotH / 2}" transform="rotate(-90 12 ${pad.top + plotH / 2})" text-anchor="middle" font-size="10.5" fill="var(--text-secondary)">${esc(
    yLabel
  )}</text>
      </svg>
      <div style="display:flex;flex-wrap:wrap;gap:10px;font-size:11px;margin-top:4px;">${legendHtml}</div>
    </div>
  `;
}

export function renderBarChart(container, { title, subtitle = "", categories = [], values = [], unit = "" }) {
  if (!container) return;
  if (!categories.length) {
    container.innerHTML = `<div class="wb-help">No bar chart data.</div>`;
    return;
  }
  const maxVal = Math.max(...values.map((v) => Math.abs(Number(v) || 0)), 1e-6);
  const rows = categories
    .map((cat, i) => {
      const val = Number(values[i]) || 0;
      const pct = Math.min(100, Math.max(2, (Math.abs(val) / maxVal) * 100));
      const col = SERIES_PALETTE[i % SERIES_PALETTE.length];
      return `
        <div style="margin-bottom:7px;">
          <div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:2px;">
            <span>${esc(cat)}</span>
            <strong class="wb-mono">${val.toFixed(val > 100 ? 1 : 4)}${esc(unit)}</strong>
          </div>
          <div style="height:8px;background:var(--bg-metric);border-radius:4px;overflow:hidden;">
            <div style="width:${pct.toFixed(1)}%;height:100%;background:${col};"></div>
          </div>
        </div>
      `;
    })
    .join("");

  container.innerHTML = `
    <div>
      <div style="font-weight:600;font-size:12px;margin-bottom:2px;">${esc(title)}</div>
      ${subtitle ? `<div class="wb-muted-xs" style="margin-bottom:8px;">${esc(subtitle)}</div>` : ""}
      ${rows}
    </div>
  `;
}

export function renderSimilarityGauge(container, { metric, score, threshold, minRange = -0.2, maxRange = 1.0 }) {
  if (!container) return;
  const lo = metric === "cosine" ? -0.2 : Math.min(score, threshold) - 1.0;
  const hi = metric === "cosine" ? 1.0 : Math.max(score, threshold, 0.0) + 0.2;
  const clampPct = (v) => Math.max(0, Math.min(100, ((v - lo) / Math.max(hi - lo, 1e-6)) * 100));
  const scorePct = clampPct(score);
  const thPct = clampPct(threshold);
  const isMatch = score >= threshold;

  container.innerHTML = `
    <div style="padding:8px 10px;background:var(--bg-metric);border:1px solid var(--border);border-radius:5px;">
      <div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:6px;">
        <span>Metric: <strong class="wb-mono">${esc(metric)}</strong> (Raw Geometric Score — NOT Probability)</span>
        <span class="wb-mono">Score: <strong>${Number(score).toFixed(5)}</strong> vs τ = <strong>${Number(threshold).toFixed(5)}</strong></span>
      </div>
      <div style="position:relative;height:18px;background:linear-gradient(90deg, #fee2e2 0%, #fef9c3 ${thPct.toFixed(
        1
      )}%, #dcfce7 100%);border:1px solid var(--border-strong);border-radius:4px;">
        <div title="Decision Threshold τ = ${threshold}" style="position:absolute;left:${thPct.toFixed(
          2
        )}%;top:-4px;bottom:-4px;width:2px;background:#0f172a;"></div>
        <div title="Measured Score = ${score}" style="position:absolute;left:calc(${scorePct.toFixed(
          2
        )}% - 6px);top:2px;width:12px;height:12px;border-radius:50%;background:${
    isMatch ? "#16a34a" : "#dc2626"
  };border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,0.3);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10.5px;color:var(--text-muted);margin-top:4px;" class="wb-mono">
        <span>${lo.toFixed(2)} (Non-Match Region)</span>
        <span>τ = ${Number(threshold).toFixed(4)}</span>
        <span>${hi.toFixed(2)} (Match Region)</span>
      </div>
    </div>
  `;
}
