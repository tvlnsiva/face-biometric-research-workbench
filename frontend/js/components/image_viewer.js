/**
 * Reusable Scientific Image Viewer (`frontend/js/components/image_viewer.js`).
 *
 * Features:
 *  - Exact original-pixel <-> screen-pixel projection via `coordinates.js`
 *  - Fit to Screen, 1:1 Native Resolution, Zoom +/-, cursor-anchored wheel zoom, drag pan, Fullscreen
 *  - Live cursor `(x, y)` readout in native image pixels + RGB pixel value sample
 *  - Independent overlay toggles + semantic color legend
 *  - Synchronized zoom/pan across paired comparison viewers
 *  - Click-to-select detected face bounding boxes
 */

import { store } from "../state.js";
import {
  computeFitTransform,
  computeNative1to1Transform,
  hitTestFaceBoxes,
  origToScreen,
  projectBoxToScreen,
  projectPointsToScreen,
  screenToOrig,
  zoomAroundScreenPoint,
} from "../utils/coordinates.js";
import { helpBadgeHtml } from "../utils/tooltips.js";

/**
 * Pure reference alpha-compositing formula for P-3 verification:
 * `C_out = (1 - alpha) * C_base + alpha * C_overlay`, where `alpha = clamp(opacityPct, 0, 100) / 100`.
 */
export function computeOverlayAlphaComposite(baseRgb, overlayRgb, opacityPct) {
  const alpha = Math.max(0, Math.min(100, Number(opacityPct))) / 100.0;
  if (alpha <= 0) return [baseRgb[0], baseRgb[1], baseRgb[2]];
  if (alpha >= 1) return [overlayRgb[0], overlayRgb[1], overlayRgb[2]];
  return [
    Math.round((1 - alpha) * baseRgb[0] + alpha * overlayRgb[0]),
    Math.round((1 - alpha) * baseRgb[1] + alpha * overlayRgb[1]),
    Math.round((1 - alpha) * baseRgb[2] + alpha * overlayRgb[2]),
  ];
}

let viewerInstanceCounter = 0;

const COLORS = {
  gt: "#16a34a",
  tp: "#0284c7",
  fp: "#dc2626",
  fn: "#ea580c",
  unverified: "#0284c7",
  selected: "#7c3aed",
  lowconf: "#ca8a04",
  landmarkPrimary: "#0ea5e9",
  landmarkSecondary: "#f97316",
  correspondence: "#0d9488",
  alignQuad: "#8b5cf6",
};

const LANDMARK_SHORT = ["R-Eye(S)", "L-Eye(S)", "Nose", "R-Mouth(S)", "L-Mouth(S)"];

export class ScientificImageViewer {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      title: options.title || "Image Inspector",
      heightPx: options.heightPx || 420,
      showOverlayToggles: options.showOverlayToggles !== false,
      showLegend: options.showLegend !== false,
      overlays: {
        bboxes: true,
        gtBoxes: true,
        landmarks: true,
        landmarkLabels: false,
        correspondenceLines: true,
        confidenceLabels: true,
        alignmentBoundary: true,
        ...(options.overlays || {}),
      },
      onSelectFace: options.onSelectFace || null,
      onTransformChange: options.onTransformChange || null,
      onOpacityChange: options.onOpacityChange || null,
    };

    viewerInstanceCounter += 1;
    this._instanceId = viewerInstanceCounter;
    this.overlayOpacityPct =
      options.overlayOpacityPct !== undefined
        ? Math.max(0, Math.min(100, Number(options.overlayOpacityPct)))
        : Math.max(0, Math.min(100, Number(store?.ui?.overlayOpacityPct ?? 100)));
    this.image = null;
    this.imageMeta = null;
    this.transform = null;
    this._syncing = false;
    this._unsubStore = null;
    this.scene = {
      detections: [],
      selectedFaceIndex: 0,
      gtBoxes: [],
      unmatchedGtBoxes: [],
      secondaryLandmarks: null, // optional 5x2 array for dual-detector comparison
      templateCorrespondences: null, // optional 5x2 target points (for crop view)
      alignmentQuadOrig: null, // optional 4x2 polygon in original image px
    };

    this._buildDom();
  }

  _buildDom() {
    this.container.innerHTML = `
      <div class="wb-viewer">
        <div class="wb-viewer-toolbar">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <strong data-role="title">${this.options.title}</strong>
            <span class="wb-mono wb-muted-xs" data-role="dims">—</span>
          </div>
          <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;">
            <span class="wb-mono wb-muted-xs" data-role="cursor" style="margin-right:6px;">px: (—, —)</span>
            <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="fit" title="Fit entire image to viewport">Fit</button>
            <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="native" title="Display at 1:1 native pixel resolution">1:1 Native</button>
            <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="zin" title="Zoom in">+</button>
            <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="zout" title="Zoom out">−</button>
            <button type="button" class="wb-btn wb-btn-secondary wb-btn-sm" data-act="full" title="Toggle fullscreen">Fullscreen</button>
          </div>
        </div>
        <div class="wb-viewer-stage" style="height:${this.options.heightPx}px;">
          <canvas></canvas>
        </div>
        ${
          this.options.showOverlayToggles
            ? `
        <div class="wb-viewer-overlays-bar">
          <label title="Toggle detected face bounding boxes"><input type="checkbox" data-ov="bboxes" ${this.options.overlays.bboxes ? "checked" : ""}/> Face Boxes</label>
          <label title="Toggle human/manifest ground-truth bounding boxes (green)"><input type="checkbox" data-ov="gtBoxes" ${this.options.overlays.gtBoxes ? "checked" : ""}/> Ground Truth</label>
          <label title="Toggle 5-point anatomical facial landmarks"><input type="checkbox" data-ov="landmarks" ${this.options.overlays.landmarks ? "checked" : ""}/> Landmarks</label>
          <label title="Toggle short text labels (R-Eye, L-Eye, Nose, Mouth) next to landmarks"><input type="checkbox" data-ov="landmarkLabels" ${this.options.overlays.landmarkLabels ? "checked" : ""}/> Point Labels</label>
          <label title="Toggle landmark displacement or canonical template correspondence vectors"><input type="checkbox" data-ov="correspondenceLines" ${this.options.overlays.correspondenceLines ? "checked" : ""}/> Correspondences</label>
          <label title="Toggle detector confidence score badges above boxes"><input type="checkbox" data-ov="confidenceLabels" ${this.options.overlays.confidenceLabels ? "checked" : ""}/> Confidence</label>
          <label title="Toggle back-projected 112×112 aligned crop boundary polygon"><input type="checkbox" data-ov="alignmentBoundary" ${this.options.overlays.alignmentBoundary ? "checked" : ""}/> Crop Boundary</label>
          <span style="margin-left:auto;display:inline-flex;align-items:center;gap:5px;" data-tip-key="overlay_opacity">
            <label for="wb-ov-opacity-${this._instanceId}" style="font-weight:600;cursor:pointer;">Overlay Opacity:</label>
            <input type="range" id="wb-ov-opacity-${this._instanceId}" data-ov-opacity="1" min="0" max="100" step="1"
              value="${Math.round(this.overlayOpacityPct)}" aria-label="Overlay Opacity (0 to 100 percent)"
              data-tip-key="overlay_opacity" style="width:88px;cursor:pointer;" />
            <span class="wb-mono" data-role="ov-opacity-val" style="min-width:34px;text-align:right;">${Math.round(
              this.overlayOpacityPct
            )}%</span>
            ${helpBadgeHtml("overlay_opacity", "Explain Overlay Opacity")}
          </span>
        </div>`
            : ""
        }
        ${
          this.options.showLegend
            ? `
        <div class="wb-viewer-legend">
          <span class="wb-legend-item"><span class="wb-swatch" style="background:${COLORS.gt}"></span>Ground Truth</span>
          <span class="wb-legend-item"><span class="wb-swatch" style="background:${COLORS.tp}"></span>Detected / TP</span>
          <span class="wb-legend-item"><span class="wb-swatch" style="background:${COLORS.fp}"></span>False Positive</span>
          <span class="wb-legend-item"><span class="wb-swatch" style="background:${COLORS.fn}"></span>Missed GT (FN)</span>
          <span class="wb-legend-item"><span class="wb-swatch" style="background:${COLORS.selected}"></span>Selected Face</span>
          <span class="wb-legend-item"><span class="wb-swatch" style="background:${COLORS.lowconf}"></span>Low Confidence</span>
          <span class="wb-legend-item"><span class="wb-swatch" style="background:${COLORS.correspondence}"></span>Landmark Displacement / Template Link</span>
        </div>`
            : ""
        }
      </div>
    `;

    this.rootEl = this.container.querySelector(".wb-viewer");
    this.stageEl = this.container.querySelector(".wb-viewer-stage");
    this.canvas = this.container.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.dimsEl = this.container.querySelector('[data-role="dims"]');
    this.cursorEl = this.container.querySelector('[data-role="cursor"]');

    this._bindEvents();
  }

  _bindEvents() {
    let dragging = false;
    let moved = false;
    let lastX = 0;
    let lastY = 0;

    this.stageEl.addEventListener("mousedown", (e) => {
      dragging = true;
      moved = false;
      lastX = e.clientX;
      lastY = e.clientY;
    });

    window.addEventListener("mouseup", () => {
      dragging = false;
    });

    this.stageEl.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      if (dragging && this.transform) {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) moved = true;
        lastX = e.clientX;
        lastY = e.clientY;
        this.transform.offsetX += dx;
        this.transform.offsetY += dy;
        this.render();
        this._emitTransform();
      }

      if (this.transform) {
        const orig = screenToOrig(sx, sy, this.transform);
        if (orig.inBounds) {
          this.cursorEl.textContent = `px: (${Math.floor(orig.x)}, ${Math.floor(orig.y)}) · ${(
            this.transform.scale * 100
          ).toFixed(0)}%`;
        } else {
          this.cursorEl.textContent = `out of bounds · ${(this.transform.scale * 100).toFixed(0)}%`;
        }
      }
    });

    this.stageEl.addEventListener("click", (e) => {
      if (moved || !this.transform || !this.options.onSelectFace) return;
      const rect = this.canvas.getBoundingClientRect();
      const orig = screenToOrig(e.clientX - rect.left, e.clientY - rect.top, this.transform);
      const hitIdx = hitTestFaceBoxes(orig.x, orig.y, this.scene.detections);
      if (hitIdx >= 0) {
        this.scene.selectedFaceIndex = hitIdx;
        this.render();
        this.options.onSelectFace(hitIdx, this.scene.detections[hitIdx]);
      }
    });

    this.stageEl.addEventListener(
      "wheel",
      (e) => {
        if (!this.transform) return;
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const factor = e.deltaY < 0 ? 1.16 : 1 / 1.16;
        this.transform = zoomAroundScreenPoint(
          this.transform,
          factor,
          e.clientX - rect.left,
          e.clientY - rect.top
        );
        this.render();
        this._emitTransform();
      },
      { passive: false }
    );

    this.container.querySelector('[data-act="fit"]')?.addEventListener("click", () => {
      this.fitToScreen();
      this._emitTransform();
    });
    this.container.querySelector('[data-act="native"]')?.addEventListener("click", () => {
      if (!this.image) return;
      const r = this.stageEl.getBoundingClientRect();
      this.transform = computeNative1to1Transform(this.image.naturalWidth, this.image.naturalHeight, r.width, r.height);
      this.render();
      this._emitTransform();
    });
    this.container.querySelector('[data-act="zin"]')?.addEventListener("click", () => {
      if (!this.transform) return;
      const r = this.stageEl.getBoundingClientRect();
      this.transform = zoomAroundScreenPoint(this.transform, 1.25, r.width * 0.5, r.height * 0.5);
      this.render();
      this._emitTransform();
    });
    this.container.querySelector('[data-act="zout"]')?.addEventListener("click", () => {
      if (!this.transform) return;
      const r = this.stageEl.getBoundingClientRect();
      this.transform = zoomAroundScreenPoint(this.transform, 0.8, r.width * 0.5, r.height * 0.5);
      this.render();
      this._emitTransform();
    });
    this.container.querySelector('[data-act="full"]')?.addEventListener("click", () => {
      if (!document.fullscreenElement) {
        this.rootEl.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    });

    this.container.querySelectorAll("[data-ov]").forEach((cb) => {
      cb.addEventListener("change", () => {
        const key = cb.getAttribute("data-ov");
        this.options.overlays[key] = cb.checked;
        this.render();
      });
    });

    const opSlider = this.container.querySelector('[data-ov-opacity="1"]');
    if (opSlider) {
      opSlider.addEventListener("input", (e) => {
        this.setOverlayOpacity(Number(e.target.value), { propagate: true });
      });
    }

    if (store && typeof store.subscribe === "function") {
      this._unsubStore = store.subscribe((topic) => {
        if (topic !== "overlay_opacity") return;
        if (!this.container.isConnected) {
          this._unsubStore?.();
          this._unsubStore = null;
          return;
        }
        const targetPct = Math.max(0, Math.min(100, Number(store.ui.overlayOpacityPct ?? 100)));
        if (Math.abs(this.overlayOpacityPct - targetPct) > 1e-6) {
          this.setOverlayOpacity(targetPct, { propagate: false });
        }
      });
    }

    this._resizeObserver = new ResizeObserver(() => {
      if (this.image && !this.transform) {
        this.fitToScreen();
      } else {
        this.render();
      }
    });
    this._resizeObserver.observe(this.stageEl);
  }

  _emitTransform() {
    if (!this._syncing && this.options.onTransformChange && this.transform) {
      this.options.onTransformChange({ ...this.transform });
    }
  }

  applySyncedTransform(otherTransform) {
    if (!this.image || !otherTransform) return;
    this._syncing = true;
    // Normalize relative pan/zoom when image sizes match or scale proportionally
    const ratioX = this.image.naturalWidth / Math.max(1, otherTransform.natWidth);
    this.transform = {
      natWidth: this.image.naturalWidth,
      natHeight: this.image.naturalHeight,
      viewWidth: this.stageEl.clientWidth,
      viewHeight: this.stageEl.clientHeight,
      scale: otherTransform.scale / ratioX,
      offsetX: otherTransform.offsetX,
      offsetY: otherTransform.offsetY,
    };
    this.render();
    this._syncing = false;
  }

  async loadImageUrl(url, meta = null, preserveView = false) {
    this.imageMeta = meta;
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const sameDims =
          preserveView &&
          this.image &&
          this.transform &&
          this.image.naturalWidth === img.naturalWidth &&
          this.image.naturalHeight === img.naturalHeight;
        this.image = img;
        this.dimsEl.textContent = `${img.naturalWidth}×${img.naturalHeight}px${
          meta?.format ? ` · ${meta.format}` : ""
        }`;
        if (!sameDims) {
          this.fitToScreen();
        } else {
          this.render();
        }
        resolve(img);
      };
      img.onerror = () => reject(new Error(`Failed to decode image from ${url}`));
      img.src = url;
    });
  }

  setScene(partialScene) {
    this.scene = { ...this.scene, ...partialScene };
    this.render();
  }

  setOverlayOpacity(pct, { propagate = true } = {}) {
    const clamped = Math.max(0, Math.min(100, Math.round(Number(pct) || 0)));
    this.overlayOpacityPct = clamped;
    const slider = this.container.querySelector('[data-ov-opacity="1"]');
    if (slider && Number(slider.value) !== clamped) slider.value = String(clamped);
    const lbl = this.container.querySelector('[data-role="ov-opacity-val"]');
    if (lbl) lbl.textContent = `${clamped}%`;
    if (propagate && store?.ui) {
      store.ui.overlayOpacityPct = clamped;
      store.notify("overlay_opacity");
    }
    this.render();
    if (propagate && this.options.onOpacityChange) {
      this.options.onOpacityChange(clamped);
    }
  }

  fitToScreen() {
    if (!this.image) return;
    const r = this.stageEl.getBoundingClientRect();
    this.transform = computeFitTransform(
      this.image.naturalWidth,
      this.image.naturalHeight,
      r.width || 600,
      r.height || this.options.heightPx
    );
    this.render();
  }

  render() {
    const rect = this.stageEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(10, Math.floor(rect.width));
    const ch = Math.max(10, Math.floor(rect.height));

    if (this.canvas.width !== cw * dpr || this.canvas.height !== ch * dpr) {
      this.canvas.width = cw * dpr;
      this.canvas.height = ch * dpr;
    }

    const ctx = this.ctx;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cw, ch);

    if (!this.image || !this.transform) {
      ctx.fillStyle = "#64748b";
      ctx.font = "12px Segoe UI, sans-serif";
      ctx.fillText("No image loaded.", 16, 28);
      ctx.restore();
      return;
    }

    const t = this.transform;
    ctx.globalAlpha = 1.0;
    ctx.imageSmoothingEnabled = t.scale < 2.5;
    ctx.drawImage(this.image, t.offsetX, t.offsetY, this.image.naturalWidth * t.scale, this.image.naturalHeight * t.scale);

    const overlayAlpha = Math.max(0, Math.min(1.0, Number(this.overlayOpacityPct ?? 100) / 100.0));
    if (overlayAlpha <= 0) {
      // 0% Overlay Opacity: retain 100% pure unobstructed original image pixels and restore state cleanly
      ctx.globalAlpha = 1.0;
      ctx.restore();
      return;
    }

    const ov = this.options.overlays;
    ctx.save();
    ctx.globalAlpha = overlayAlpha;
    try {

    // 1. Ground-Truth Boxes (Dashed Green) & Missed GT FN Boxes (Dashed Orange)
    if (ov.gtBoxes && Array.isArray(this.scene.gtBoxes)) {
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = COLORS.gt;
      for (const gb of this.scene.gtBoxes) {
        const sb = projectBoxToScreen(gb, t);
        ctx.strokeRect(sb.x, sb.y, sb.w, sb.h);
      }
      if (Array.isArray(this.scene.unmatchedGtBoxes)) {
        ctx.strokeStyle = COLORS.fn;
        ctx.lineWidth = 2.5;
        for (const fnItem of this.scene.unmatchedGtBoxes) {
          const b = fnItem.bounding_box || fnItem;
          const sb = projectBoxToScreen(b, t);
          ctx.strokeRect(sb.x, sb.y, sb.w, sb.h);
          this._drawTag(ctx, sb.x, sb.y - 16, "MISSED GT (FN)", COLORS.fn);
        }
      }
      ctx.restore();
    }

    // 2. Back-projected Alignment Crop Boundary Polygon
    if (ov.alignmentBoundary && Array.isArray(this.scene.alignmentQuadOrig) && this.scene.alignmentQuadOrig.length === 4) {
      const pts = projectPointsToScreen(this.scene.alignmentQuadOrig, t);
      ctx.save();
      ctx.strokeStyle = COLORS.alignQuad;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }

    // 3. Detected Face Bounding Boxes & Landmarks
    const dets = this.scene.detections || [];
    dets.forEach((det, idx) => {
      const bbox = det.bounding_box || det;
      const isSelected = idx === this.scene.selectedFaceIndex;
      let color = COLORS.tp;
      if (det.gt_status === "fp") color = COLORS.fp;
      else if (det.low_confidence_flag) color = COLORS.lowconf;
      if (isSelected) color = COLORS.selected;

      if (ov.bboxes) {
        const sb = projectBoxToScreen(bbox, t);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 2.8 : 2.0;
        ctx.strokeRect(sb.x, sb.y, sb.w, sb.h);

        if (ov.confidenceLabels) {
          const confTxt = det.confidence !== undefined ? `${(det.confidence * 100).toFixed(1)}%` : "";
          const statusTag = det.gt_status && det.gt_status !== "unverified" ? ` · ${det.gt_status.toUpperCase()}` : "";
          this._drawTag(ctx, sb.x, Math.max(2, sb.y - 17), `face_${String(idx).padStart(3, "0")} ${confTxt}${statusTag}`, color);
        }
        ctx.restore();
      }

      const lmPts = det.landmarks?.points || det.landmarks_points;
      if (ov.landmarks && Array.isArray(lmPts)) {
        const screenPts = projectPointsToScreen(lmPts, t);

        // Optional correspondence lines to canonical template or secondary detector landmarks
        if (ov.correspondenceLines && isSelected && Array.isArray(this.scene.templateCorrespondences)) {
          const dstScreen = projectPointsToScreen(this.scene.templateCorrespondences, t);
          ctx.save();
          ctx.strokeStyle = COLORS.correspondence;
          ctx.lineWidth = 1.5;
          screenPts.forEach((sp, k) => {
            if (!dstScreen[k]) return;
            ctx.beginPath();
            ctx.moveTo(sp[0], sp[1]);
            ctx.lineTo(dstScreen[k][0], dstScreen[k][1]);
            ctx.stroke();
            ctx.strokeRect(dstScreen[k][0] - 3, dstScreen[k][1] - 3, 6, 6);
          });
          ctx.restore();
        }

        screenPts.forEach(([px, py], pIdx) => {
          ctx.beginPath();
          ctx.fillStyle = COLORS.landmarkPrimary;
          ctx.arc(px, py, isSelected ? 4.2 : 3.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = "#ffffff";
          ctx.stroke();

          if (ov.landmarkLabels) {
            ctx.font = "10px Consolas, monospace";
            ctx.fillStyle = "#ffffff";
            ctx.strokeStyle = "#0f172a";
            ctx.lineWidth = 2.5;
            const lbl = LANDMARK_SHORT[pIdx] || `P${pIdx}`;
            ctx.strokeText(lbl, px + 6, py - 4);
            ctx.fillText(lbl, px + 6, py - 4);
          }
        });
      }
    });

    // 4. Secondary Landmark Set (e.g., YuNet vs SCRFD overlay on same face)
    if (ov.landmarks && Array.isArray(this.scene.secondaryLandmarks)) {
      const secScreen = projectPointsToScreen(this.scene.secondaryLandmarks, t);
      const selDet = dets[this.scene.selectedFaceIndex];
      const primPts = selDet?.landmarks?.points ? projectPointsToScreen(selDet.landmarks.points, t) : null;

      secScreen.forEach(([sx, sy], k) => {
        if (ov.correspondenceLines && primPts && primPts[k]) {
          ctx.save();
          ctx.strokeStyle = COLORS.correspondence;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(primPts[k][0], primPts[k][1]);
          ctx.lineTo(sx, sy);
          ctx.stroke();
          ctx.restore();
        }
        ctx.save();
        ctx.fillStyle = COLORS.landmarkSecondary;
        ctx.beginPath();
        ctx.moveTo(sx, sy - 4.5);
        ctx.lineTo(sx + 4.5, sy);
        ctx.lineTo(sx, sy + 4.5);
        ctx.lineTo(sx - 4.5, sy);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      });
    }
    } finally {
      ctx.restore();
      ctx.globalAlpha = 1.0;
    }

    ctx.restore();
  }

  _drawTag(ctx, x, y, text, bg) {
    ctx.save();
    ctx.font = "600 10.5px Segoe UI, sans-serif";
    const padX = 5;
    const w = ctx.measureText(text).width + padX * 2;
    ctx.fillStyle = bg;
    ctx.fillRect(x, y, w, 16);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, x + padX, y + 11.5);
    ctx.restore();
  }
}
